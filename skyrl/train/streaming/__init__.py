"""Trainers that generate rollouts one group at a time and train on each mini-batch of groups.

A group is one prompt generated ``n_samples_per_prompt`` times.

- :class:`StreamingRayPPOTrainerBase` holds the training loop, the generation workers, the buffer of
  finished groups, and the checkpoint of consumed UIDs. A subclass implements the hooks that depend
  on where the inference engines run.
- :class:`FullyAsyncRayPPOTrainer` runs the inference engines on their own GPUs. Generation never
  stops. :class:`AsyncStalenessManager` bounds how far generation runs ahead of training.
- :class:`PartialRolloutRayPPOTrainer` runs the inference engines on the trainer GPUs. It cuts
  generation when the mini-batch is full, trains, and resumes the cut groups under the new weights.
"""

from skyrl.train.streaming.admission import (
    AsyncStalenessManager,
    ConcurrencyAdmissionController,
    RolloutAdmissionController,
    RolloutStat,
)
from skyrl.train.streaming.base import StreamingRayPPOTrainerBase
from skyrl.train.streaming.dataloader import StreamingDataloader
from skyrl.train.streaming.group import GeneratedOutputGroup
from skyrl.train.streaming.partial_rollout import PartialRolloutRayPPOTrainer
from skyrl.train.streaming.trainer import FullyAsyncRayPPOTrainer

__all__ = [
    "StreamingDataloader",
    "AsyncStalenessManager",
    "ConcurrencyAdmissionController",
    "StreamingRayPPOTrainerBase",
    "FullyAsyncRayPPOTrainer",
    "GeneratedOutputGroup",
    "PartialRolloutRayPPOTrainer",
    "RolloutAdmissionController",
    "RolloutStat",
]
