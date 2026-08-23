"""Production training loop with mixed precision and early stopping."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.factory import ModelFactory
from training.callbacks import CheckpointManager, EarlyStopping
from training.dataset import create_dataloaders
from training.losses import create_loss_function
from training.scheduler import create_scheduler
from utils.config import AppConfig
from utils.device import get_device
from utils.paths import ProjectPaths
from utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Training run outcome."""

    success: bool
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0
    best_model_path: Path | None = None
    history: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Trainer:
    """
    Full-featured trainer for plant disease classification.

    Features:
    - Mixed precision training
    - Early stopping
    - Resume from checkpoint
    - Weighted cross-entropy
    - Gradient clipping
    - Cosine LR scheduler
    - TensorBoard logging
    - Checkpoint & best model saving
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.paths = ProjectPaths.from_config(config)
        self.device = get_device(config)
        self.paths.ensure_dirs()

    def _build_class_mapping(self) -> dict[str, int]:
        return {c.folder_name: c.id for c in self.config.class_configs}

    def _run_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: GradScaler | None,
        train: bool,
        grad_clip: float,
    ) -> tuple[float, float]:
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        use_amp = scaler is not None and self.device.type == "cuda"

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in tqdm(loader, desc="Train" if train else "Val", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    if train and optimizer is not None:
                        scaler.scale(loss).backward()
                        if grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if train and optimizer is not None:
                        loss.backward()
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def train(self) -> TrainingResult:
        """Execute full training pipeline."""
        result = TrainingResult(success=False)

        seed = int(self.config.get("project.seed", 42))
        set_seed(seed)

        train_dir = self.paths.train
        val_dir = self.paths.val

        if not train_dir.exists() or not any(train_dir.iterdir()):
            result.errors.append(
                f"Training directory empty or missing: {train_dir}. Run preprocessing first."
            )
            return result

        if not val_dir.exists() or not any(val_dir.iterdir()):
            result.errors.append(
                f"Validation directory empty or missing: {val_dir}. Run preprocessing first."
            )
            return result

        class_mapping = self._build_class_mapping()
        num_classes = self.config.num_classes
        model_name = str(self.config.get("training.model_name", "efficientnet_b0"))
        num_epochs = int(self.config.get("training.num_epochs", 50))
        batch_size = int(self.config.get("training.batch_size", 32))
        num_workers = int(self.config.get("training.num_workers", 4))
        image_size = int(self.config.get("training.image_size", 224))
        lr = float(self.config.get("training.learning_rate", 0.001))
        weight_decay = float(self.config.get("training.weight_decay", 1e-4))
        label_smoothing = float(self.config.get("training.label_smoothing", 0.0))
        mixed_precision = bool(self.config.get("training.mixed_precision", True))
        grad_clip = float(self.config.get("training.gradient_clip_norm", 1.0))
        use_class_weights = bool(self.config.get("training.use_class_weights", True))

        train_loader, val_loader, train_dataset = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            class_to_idx=class_mapping,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            pin_memory=bool(self.config.get("training.pin_memory", True)),
        )

        if len(train_dataset) == 0:
            result.errors.append("Training dataset contains no samples")
            return result

        model = ModelFactory.create(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            device=self.device,
        )

        class_weights = None
        if use_class_weights:
            class_weights = train_dataset.compute_class_weights()

        criterion = create_loss_function(
            num_classes=num_classes,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            device=self.device,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(
            optimizer,
            config=self.config.get("training", {}),
            num_epochs=num_epochs,
        )

        scaler = GradScaler() if mixed_precision and self.device.type == "cuda" else None

        checkpoint_cfg = self.config.get("training.checkpoint", {})
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.paths.weights / "checkpoints",
            keep_last_n=int(checkpoint_cfg.get("keep_last_n", 3)),
            save_best=bool(checkpoint_cfg.get("save_best", True)),
        )

        early_stop_cfg = self.config.get("training.early_stopping", {})
        early_stopping = None
        if early_stop_cfg.get("enabled", True):
            early_stopping = EarlyStopping(
                patience=int(early_stop_cfg.get("patience", 10)),
                min_delta=float(early_stop_cfg.get("min_delta", 0.001)),
                monitor=str(early_stop_cfg.get("monitor", "val_loss")),
                mode=str(early_stop_cfg.get("mode", "min")),
            )

        run_name = f"{self.config.crop_name}_{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        log_dir = self.paths.logs / "tensorboard" / run_name
        writer = None
        if self.config.get("training.tensorboard", True):
            writer = SummaryWriter(log_dir=str(log_dir))

        start_epoch = 0
        resume = bool(self.config.get("training.resume", False))
        resume_path = self.config.get("training.resume_checkpoint")

        if resume and resume_path:
            ckpt_path = Path(resume_path)
            if not ckpt_path.is_absolute():
                ckpt_path = self.config.project_root / ckpt_path
            if ckpt_path.exists():
                ckpt = CheckpointManager.load_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler, self.device
                )
                start_epoch = ckpt.get("epoch", 0) + 1
                logger.info("Resumed training from epoch %d", start_epoch)
            else:
                result.errors.append(f"Resume checkpoint not found: {ckpt_path}")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        max_val_accuracy = 0.0
        best_epoch = 0
        save_every = int(checkpoint_cfg.get("save_every_n_epochs", 5))
        last_checkpoint: dict | None = None

        logger.info(
            "Training %s on %s | epochs=%d batch=%d samples=%d",
            model_name,
            self.device,
            num_epochs,
            batch_size,
            len(train_dataset),
        )

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            train_loss, train_acc = self._run_epoch(
                model, train_loader, criterion, optimizer, scaler,
                train=True, grad_clip=grad_clip,
            )
            val_loss, val_acc = self._run_epoch(
                model, val_loader, criterion, None, None,
                train=False, grad_clip=0.0,
            )
            scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
                "epoch_time_sec": epoch_time,
            }
            result.history.append(epoch_metrics)

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | "
                "Train Acc: %.4f | Val Acc: %.4f | LR: %.6f | Time: %.1fs",
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                current_lr,
                epoch_time,
            )
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | Epoch Time: {epoch_time:.1f}s"
            )

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Accuracy/train", train_acc, epoch)
                writer.add_scalar("Accuracy/val", val_acc, epoch)
                writer.add_scalar("LearningRate", current_lr, epoch)

            max_val_accuracy = max(max_val_accuracy, val_acc)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch

            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_metrics,
                "class_mapping": {
                    "folder_to_id": class_mapping,
                    "id_to_display": {c.id: c.display_name for c in self.config.class_configs},
                    "model_name": model_name,
                    "image_size": image_size,
                },
                "model_name": model_name,
            }
            if scheduler is not None:
                checkpoint_payload["scheduler_state_dict"] = scheduler.state_dict()
            if scaler is not None:
                checkpoint_payload["scaler_state_dict"] = scaler.state_dict()
            last_checkpoint = checkpoint_payload

            if is_best or (epoch + 1) % save_every == 0:
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    class_mapping=checkpoint_payload["class_mapping"],
                    model_name=model_name,
                    is_best=is_best,
                    scaler_state=scaler.state_dict() if scaler else None,
                )

            if early_stopping and early_stopping.step(val_loss, epoch):
                logger.info(
                    "Early stopping triggered at epoch %d (best epoch: %d)",
                    epoch + 1,
                    early_stopping.state.best_epoch + 1,
                )
                break

        if writer:
            writer.close()

        last_model_path = self.paths.weights / "last_model.pth"
        if last_checkpoint is not None:
            torch.save(last_checkpoint, last_model_path)
            logger.info("Last model saved: %s", last_model_path)

        # Save training history
        history_path = self.paths.weights / "training_history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(result.history, f, indent=2)

        best_model_path = self.paths.weights / "best_model.pth"
        if not best_model_path.exists() and checkpoint_manager.best_path:
            best_model_path = checkpoint_manager.best_path

        result.success = True
        result.best_val_loss = best_val_loss
        result.best_val_accuracy = max_val_accuracy
        result.best_epoch = best_epoch
        result.total_epochs = len(result.history)
        result.best_model_path = best_model_path if best_model_path.exists() else None

        logger.info(
            "Training complete | best_val_loss=%.4f best_val_acc=%.4f epoch=%d",
            best_val_loss, best_val_acc, best_epoch + 1,
        )

        return result
