"""Colocated partial rollout: cut generation when the mini-batch is full, train, resume the rest.

Inference engines and the trainer share the GPUs, as in the synchronous trainer, but the step does
not wait for every trajectory. ``num_parallel_generation_workers`` groups are kept in flight; once
``policy_mini_batch_size`` of them have finished, the in-flight requests are aborted, the engines
sleep, the trainer runs on the finished groups, the engines wake with the new weights, and the
aborted requests continue from the tokens they had produced (see
``RemoteInferenceClient.resume_aborted_requests``). The fraction of in-flight work that is cut
each step is therefore ``1 - policy_mini_batch_size / num_parallel_generation_workers``.

Compared with the fully-async trainer, there is no GPU split and no generation during training, so
every mini-batch is at most a few cuts behind the current policy; compared with the synchronous
trainer, the step no longer waits for the longest trajectory. Tokens produced before a cut carry the
rollout logprobs of the older policy, so the loss must use rollout logprobs, which the base class
asserts.
"""

from loguru import logger

from skyrl.train.streaming.admission import (
    ConcurrencyAdmissionController,
    RolloutAdmissionController,
)
from skyrl.train.streaming.base import StreamingRayPPOTrainerBase


class PartialRolloutRayPPOTrainer(StreamingRayPPOTrainerBase):
    """Colocated streaming trainer. See the module docstring."""

    checkpoint_before_weight_sync = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Aborted requests are held in the client and resubmitted after the weight sync, so the
        # generator sees one uninterrupted completion per trajectory.
        self.inference_engine_client.resume_aborted_requests = True

    def _generation_buffer_maxsize(self) -> int:
        # Every in-flight group can finish before the consumer drains the buffer.
        return self.num_parallel_generation_workers

    def _build_admission_controller(self) -> RolloutAdmissionController:
        return ConcurrencyAdmissionController(
            max_concurrent_groups=self.num_parallel_generation_workers,
            mini_batch_size=self.mini_batch_size,
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
        cut_fraction = 1 - self.mini_batch_size / self.num_parallel_generation_workers
        logger.info(
            f"Partial rollout: {self.num_parallel_generation_workers} groups in flight, train on every "
            f"{self.mini_batch_size} finished groups, {cut_fraction:.0%} of in-flight groups cut per step."
        )

    async def _before_training_step(self) -> None:
        # Abort the in-flight requests (the client holds them) and free the GPUs for training.
        await self.inference_engine_client.sleep()

    async def _sync_weights_after_training_step(self) -> None:
        # Inherited transfer: with colocate_all the dispatch wakes the weights, broadcasts, and wakes
        # the KV cache, which resumes the engines and releases the held requests.
        await super()._sync_weights_after_training_step()
        # By now every request this step's cut aborted has come back to the client and is held.
        stats = self.inference_engine_client.pop_resume_stats()
        self.all_metrics.update({f"partial_rollout/{k}": v for k, v in stats.items()})

    async def _after_training_loop(self) -> None:
        await self.inference_engine_client.sleep()
