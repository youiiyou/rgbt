from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler[int]):
    """Sample identity-balanced mini-batches from five-field video samples."""

    def __init__(
        self,
        data_source: Sequence[tuple],
        batch_size: int,
        num_instances: int,
        seed: int = 1,
    ):
        if batch_size < 1 or num_instances < 1:
            raise ValueError("batch_size and num_instances must be positive")
        if batch_size < num_instances:
            raise ValueError("batch_size must be at least num_instances")
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.seed = int(seed)
        self.epoch = 0
        self.index_dic: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(data_source):
            if len(sample) < 1:
                raise ValueError("Every sample must contain a PID in field 0")
            self.index_dic[int(sample[0])].append(index)
        if len(self.index_dic) < self.num_pids_per_batch:
            raise ValueError(
                "Not enough identities for one identity-balanced batch: "
                f"need {self.num_pids_per_batch}, got {len(self.index_dic)}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_chunks(self, rng: random.Random) -> dict[int, list[list[int]]]:
        chunks: dict[int, list[list[int]]] = {}
        for pid, source_indices in self.index_dic.items():
            indices = list(source_indices)
            if len(indices) < self.num_instances:
                indices.extend(
                    rng.choices(indices, k=self.num_instances - len(indices))
                )
            rng.shuffle(indices)
            remainder = len(indices) % self.num_instances
            if remainder:
                indices.extend(rng.choices(indices, k=self.num_instances - remainder))
            chunks[pid] = [
                indices[start : start + self.num_instances]
                for start in range(0, len(indices), self.num_instances)
            ]
        return chunks

    def _generate_indices(self, epoch: int) -> list[int]:
        rng = random.Random(self.seed + int(epoch))
        chunks = self._build_chunks(rng)
        active = list(chunks)
        result: list[int] = []
        while len(active) >= self.num_pids_per_batch:
            rng.shuffle(active)
            active.sort(key=lambda pid: len(chunks[pid]), reverse=True)
            selected = active[: self.num_pids_per_batch]
            for pid in selected:
                result.extend(chunks[pid].pop(0))
                if not chunks[pid]:
                    active.remove(pid)
        return result

    def __iter__(self):
        return iter(self._generate_indices(self.epoch))

    def __len__(self) -> int:
        return len(self._generate_indices(self.epoch))
