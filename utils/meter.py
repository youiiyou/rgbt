from __future__ import annotations


class AverageMeter:
    """Track a sample-weighted running average."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.val = float(value)
        self.sum += self.val * int(count)
        self.count += int(count)
        self.avg = self.sum / self.count
