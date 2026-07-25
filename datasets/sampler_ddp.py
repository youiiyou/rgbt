from __future__ import annotations

import torch.distributed as dist
from torch.utils.data import Sampler

from .sampler import RandomIdentitySampler


class RandomIdentitySampler_DDP(Sampler[int]):
    """Identity sampler whose global batches are split across distributed ranks."""

    def __init__(
        self,
        data_source,
        global_batch_size: int,
        num_instances: int,
        seed: int = 1,
    ):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DDP identity sampler requires an initialized process group")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        if global_batch_size % self.world_size != 0:
            raise ValueError("global batch size must divide evenly across ranks")
        self.global_batch_size = int(global_batch_size)
        self.local_batch_size = self.global_batch_size // self.world_size
        if self.local_batch_size % num_instances != 0:
            raise ValueError("per-rank batch size must be divisible by num_instances")
        self.seed = int(seed)
        self.epoch = 0
        self._base_sampler = RandomIdentitySampler(
            data_source,
            batch_size=self.global_batch_size,
            num_instances=num_instances,
            seed=self.seed,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._base_sampler.set_epoch(epoch)

    def __iter__(self):
        global_indices = list(iter(self._base_sampler))
        local_indices: list[int] = []
        for start in range(0, len(global_indices), self.global_batch_size):
            batch = global_indices[start : start + self.global_batch_size]
            if len(batch) != self.global_batch_size:
                continue
            local_start = self.rank * self.local_batch_size
            local_indices.extend(
                batch[local_start : local_start + self.local_batch_size]
            )
        return iter(local_indices)

    def __len__(self) -> int:
        return len(self._base_sampler) // self.world_size
