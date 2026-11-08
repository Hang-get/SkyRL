"""The colocated partial-rollout trainer.

The inference engines and the trainer share the GPUs, as in the synchronous trainer. The step does
not wait for every trajectory. The trainer keeps ``num_parallel_generation_workers`` groups in
flight. When ``policy_mini_batch_size`` of them have finished, the step runs in this order:

1. The inference engines abort the in-flight requests and sleep.
2. The trainer trains on the finished groups.
3. The inference engines wake with the new weights.
4. The aborted requests continue from the tokens they generated before the cut
   (see ``RemoteInferenceClient.resume_aborted_requests``).

Each step cuts the fraction ``1 - policy_mini_batch_size / num_parallel_generation_workers`` of the
in-flight groups.

Compared with the fully-async trainer, there is no GPU split and no generation during training. A
mini-batch is at most a few cuts behind the current policy. Compared with the synchronous trainer,
the step does not wait for the longest trajectory. Tokens generated before a cut carry the rollout
logprobs of the older policy. The loss must use rollout logprobs, and the base class asserts this.

``trainer.fully_async.stale_group_policy`` selects what happens to a group that reaches
``max_staleness_steps`` cuts. ``ignore`` trains on it and logs a warning. ``wait`` delays the next cut
until every such group has finished. ``drop`` does not resume such a group: the client returns its
request with stop reason ``abort``, and the trainer discards the group.
"""

from loguru import logger

from skyrl.train.streaming.admission import (
    AsyncStalenessManager,
    RolloutAdmissionController,
)
from skyrl.train.streaming.base import StreamingRayPPOTrainerBase
from skyrl.train.streaming.group import GeneratedOutputGroup

STALE_GROUP_POLICIES = ("ignore", "wait", "drop")


class PartialRolloutRayPPOTrainer(StreamingRayPPOTrainerBase):
    """The colocated streaming trainer. See the module docstring."""

    checkpoint_before_weight_sync = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The client holds each aborted request and resubmits it after the weight sync. The
        # generator receives one completion per trajectory, as if no cut happened.
        self.inference_engine_client.resume_aborted_requests = True
        self._stale_group_policy = self.cfg.trainer.fully_async.stale_group_policy
        self._stale_groups_dropped = 0
        if self._stale_group_policy == "drop":
            # A request that survives more than max_staleness_steps cuts comes back with stop reason
            # "abort", and `_accept_generated_group` drops its group.
            self.inference_engine_client.max_resumes_per_request = self.max_staleness_steps

    def _generation_buffer_maxsize(self) -> int:
        # Every in-flight group can finish before the training loop drains the buffer.
        return self.num_parallel_generation_workers

    def _build_admission_controller(self) -> RolloutAdmissionController:
        # No staleness term: the cut bounds the staleness, and a resumed group already holds its slot.
        return AsyncStalenessManager(
            max_concurrent_generation_groups=self.num_parallel_generation_workers,
            mini_batch_size=self.mini_batch_size,
            max_staleness_steps=None,
        )

    def _validate_mode_config(self) -> None:
        assert self.colocate_all, (
            "PartialRolloutRayPPOTrainer requires trainer.placement.colocate_all=true; with separate "
            "inference GPUs use FullyAsyncRayPPOTrainer instead."
        )
        assert self.num_parallel_generation_workers >= self.mini_batch_size, (
            "trainer.fully_async.num_parallel_generation_workers must be >= policy_mini_batch_size "
            "(equal means no partial rollout: every step waits for all its groups). Got: "
            f"{self.num_parallel_generation_workers=}, {self.mini_batch_size=}"
        )
        policy = self.cfg.trainer.fully_async.stale_group_policy
        assert (
            policy in STALE_GROUP_POLICIES
        ), f"trainer.fully_async.stale_group_policy must be one of {STALE_GROUP_POLICIES}. Got: {policy!r}"
        if policy != "ignore":
            assert self.max_staleness_steps >= 0, (
                f"trainer.fully_async.max_staleness_steps must be >= 0 under stale_group_policy={policy!r}. "
                f"Got: {self.max_staleness_steps}"
            )
        cut_fraction = 1 - self.mini_batch_size / self.num_parallel_generation_workers
        logger.info(
            f"Partial rollout: {self.num_parallel_generation_workers} groups in flight, train on every "
            f"{self.mini_batch_size} finished groups, {cut_fraction:.0%} of in-flight groups cut per step, "
            f"stale_group_policy={policy} at max_staleness_steps={self.max_staleness_steps}."
        )

    def _epoch_may_exhaust_early(self) -> bool:
        # Dropped stale groups do not reach the buffer, so an epoch can run out of prompts mid mini-batch.
        return super()._epoch_may_exhaust_early() or self.cfg.trainer.fully_async.stale_group_policy == "drop"

    def _accept_generated_group(self, group: GeneratedOutputGroup) -> bool:
        if self._stale_group_policy != "drop":
            return True
        stop_reasons = group.generator_output.get("stop_reasons") or []
        if "abort" not in stop_reasons:
            return True
        # The client did not resume a request of this group: it reached max_staleness_steps cuts.
        self._stale_groups_dropped += 1
        logger.debug(f"Dropping group {group.uid}: it reached {self.max_staleness_steps} cuts.")
        return False

    async def _wait_for_groups_at_staleness_threshold(self) -> None:
        """Delay the cut until every in-flight group at the staleness threshold has finished.

        A group that started at step ``s`` has staleness ``global_step - s`` when a step trains on it.
        A group with ``s <= global_step - max_staleness_steps`` would exceed the threshold after one
        more cut. The trainer lets it finish before the inference engines sleep."""
        threshold = self.global_step - self.max_staleness_steps
        gen = self._epoch_generation
        waited_groups = 0
        with self._phase_gauge.timed_phase("wait_for_stale_groups", self.all_timings):
            while True:
                stale = [uid for uid, start in self._in_flight_start_steps.items() if start <= threshold]
                if not stale:
                    break
                waited_groups = max(waited_groups, len(stale))
                self._in_flight_changed.clear()
                changed = self._in_flight_changed.wait()
                await (gen.failure.guard(changed) if gen is not None else changed)
        self.all_metrics["partial_rollout/stale_groups_waited"] = waited_groups

    async def _before_training_step(self) -> None:
        if self._stale_group_policy == "wait":
            await self._wait_for_groups_at_staleness_threshold()
        # The sleep aborts the in-flight requests, which the client holds, and frees the GPUs for training.
        await self.inference_engine_client.sleep()

    async def _sync_weights_after_training_step(self) -> None:
        # The inherited weight sync wakes the weights, broadcasts them, and wakes the key-value (KV)
        # cache. The KV cache wake resumes the inference engines, and the client resubmits the held
        # requests.
        await super()._sync_weights_after_training_step()
        # At this point the client holds every request that this step's cut aborted.
        stats = self.inference_engine_client.pop_resume_stats()
        self.all_metrics.update({f"partial_rollout/{k}": v for k, v in stats.items()})
        self.all_metrics["partial_rollout/stale_groups_dropped"] = self._stale_groups_dropped
        self._stale_groups_dropped = 0

    async def _after_training_loop(self) -> None:
        await self.inference_engine_client.sleep()
