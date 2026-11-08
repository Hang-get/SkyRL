"""The fully-async trainer. The inference engines run on their own GPUs and generation never stops.

For details, see https://docs.skyrl.ai/docs/tutorials/fully_async.
"""

from skyrl.train.streaming.admission import (
    AsyncStalenessManager,
    RolloutAdmissionController,
)
from skyrl.train.streaming.base import StreamingRayPPOTrainerBase


class FullyAsyncRayPPOTrainer(StreamingRayPPOTrainerBase):
    """The disaggregated fully-async trainer.

    Generation runs without pause on dedicated inference GPUs. :class:`AsyncStalenessManager` bounds
    admission, so no group is more than ``max_staleness_steps`` policy versions behind training. The
    weight sync pauses the inference engines, broadcasts the weights, and resumes the engines.
    """

    def _generation_buffer_maxsize(self) -> int:
        # The admission controller allows at most mini_batch_size * (max_staleness_steps + 1) finished
        # groups ahead of training.
        return self.mini_batch_size * (self.max_staleness_steps + 1)

    def _build_admission_controller(self) -> RolloutAdmissionController:
        return AsyncStalenessManager(
            max_concurrent_generation_groups=self.num_parallel_generation_workers,
            mini_batch_size=self.mini_batch_size,
            max_staleness_steps=self.max_staleness_steps,
        )

    def _validate_mode_config(self) -> None:
        assert (
            # otherwise wasted throughput
            self.mini_batch_size <= self.num_parallel_generation_workers
            and
            # otherwise would never use all workers due to capacity constraint
            self.num_parallel_generation_workers <= self.mini_batch_size * (self.max_staleness_steps + 1)
        ), (
            "Invalid num_parallel_generation_workers, the following must hold: "
            "mini_batch_size <= num_parallel_generation_workers <= mini_batch_size * (max_staleness_steps + 1). Got: "
            f"{self.mini_batch_size=}, {self.num_parallel_generation_workers=}, {self.max_staleness_steps=}"
        )
        # Generation and training overlap, so they need separate GPUs. A colocated trainer is a
        # separate subclass of StreamingRayPPOTrainerBase, not a mode of this one.
        assert not self.colocate_all, "colocate_all is not supported for async training yet."
