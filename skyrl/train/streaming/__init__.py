"""Trainers that generate rollouts one group at a time and train on each mini-batch of groups.

A group is one prompt generated ``n_samples_per_prompt`` times.

- :class:`StreamingRayPPOTrainerBase` holds the training loop, the generation workers, the buffer of
  finished groups, and the checkpoint of consumed UIDs. A subclass implements the hooks that depend
  on where the inference engines run.
- :class:`FullyAsyncRayPPOTrainer` runs the inference engines on their own GPUs. Generation never
  stops. :class:`AsyncStalenessManager` bounds how far generation runs ahead of training.
- A colocated partial-rollout trainer is the planned second subclass. It cuts generation when the
  mini-batch is full and resumes the cut groups under the new weights.
"""

from skyrl.train.streaming.admission import (
    AsyncStalenessManager,
    RolloutAdmissionController,
    RolloutStat,
)
from skyrl.train.streaming.base import StreamingRayPPOTrainerBase
from skyrl.train.streaming.dataloader import StreamingDataloader
from skyrl.train.streaming.group import GeneratedOutputGroup
from skyrl.train.streaming.trainer import FullyAsyncRayPPOTrainer

__all__ = [
    "StreamingDataloader",
    "AsyncStalenessManager",
    "StreamingRayPPOTrainerBase",
    "FullyAsyncRayPPOTrainer",
    "GeneratedOutputGroup",
    "RolloutAdmissionController",
    "RolloutStat",
]
