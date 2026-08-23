"""Learning rate schedulers."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWithWarmup(LRScheduler):
    """
    Cosine annealing LR scheduler with linear warmup.

    After warmup, follows cosine decay from base LR to ``min_lr``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ) -> None:
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


def create_scheduler(
    optimizer: Optimizer,
    config: dict,
    num_epochs: int,
) -> LRScheduler:
    """Create cosine scheduler from training config."""
    scheduler_cfg = config.get("scheduler", {})
    return CosineAnnealingWithWarmup(
        optimizer=optimizer,
        total_epochs=num_epochs,
        warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 0)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
    )
