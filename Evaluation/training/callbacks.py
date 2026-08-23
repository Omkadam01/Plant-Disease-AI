"""Training callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class EarlyStoppingState:
    """Tracks early stopping progress."""

    best_score: float = float("inf")
    counter: int = 0
    stopped: bool = False
    best_epoch: int = 0


class EarlyStopping:
    """
    Early stopping callback.

    Monitors a metric and stops training when no improvement is observed
    for ``patience`` epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.state = EarlyStoppingState()
        if mode == "min":
            self.state.best_score = float("inf")
        else:
            self.state.best_score = float("-inf")

    def step(self, score: float, epoch: int) -> bool:
        """
        Update early stopping state.

        Returns True if training should stop.
        """
        improved = False
        if self.mode == "min":
            improved = score < self.state.best_score - self.min_delta
        else:
            improved = score > self.state.best_score + self.min_delta

        if improved:
            self.state.best_score = score
            self.state.counter = 0
            self.state.best_epoch = epoch
        else:
            self.state.counter += 1
            if self.state.counter >= self.patience:
                self.state.stopped = True
                return True

        return False


class CheckpointManager:
    """Manages model checkpoint saving and retention."""

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last_n: int = 3,
        save_best: bool = True,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints: list[Path] = []
        self.best_score: float = float("inf")
        self.best_path: Path | None = None

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epoch: int,
        metrics: dict[str, float],
        class_mapping: dict,
        model_name: str,
        is_best: bool = False,
        scaler_state: dict | None = None,
    ) -> Path:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "class_mapping": class_mapping,
            "model_name": model_name,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if scaler_state is not None:
            checkpoint["scaler_state_dict"] = scaler_state

        filename = f"checkpoint_epoch_{epoch:03d}.pth"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.saved_checkpoints.append(path)

        if is_best and self.save_best:
            best_path = self.checkpoint_dir.parent / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.best_path = best_path
            self.best_score = metrics.get("val_loss", float("inf"))

        self._cleanup_old_checkpoints()
        return path

    def _cleanup_old_checkpoints(self) -> None:
        while len(self.saved_checkpoints) > self.keep_last_n:
            old = self.saved_checkpoints.pop(0)
            if old.exists():
                old.unlink()

    @staticmethod
    def load_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        device: torch.device | None = None,
    ) -> dict:
        """Load checkpoint and restore model/optimizer states."""
        map_location = device or "cpu"
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint
