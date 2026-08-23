"""Tomato-specific EfficientNet-B0 training pipeline."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.factory import ModelFactory
from training.callbacks import EarlyStopping
from training.dataset import PlantDiseaseDataset, create_dataloaders
from training.losses import create_loss_function
from training.metrics_utils import compute_classification_metrics
from training.scheduler import create_scheduler
from utils.config import AppConfig
from utils.device import get_device
from utils.paths import ProjectPaths
from utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass
class TomatoTrainingResult:
    success: bool
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    best_val_loss: float = float("inf")
    total_training_time_sec: float = 0.0
    best_model_path: Path | None = None
    history: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class TomatoTrainer:
    """Train tomato EfficientNet-B0 on balanced_train with split/val validation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.paths = ProjectPaths.from_config(config)
        self.device = get_device(config)

    def _class_mapping(self) -> dict[str, int]:
        return {c.folder_name: c.id for c in self.config.class_configs}

    def _class_names(self) -> list[str]:
        ordered = sorted(self.config.class_configs, key=lambda c: c.id)
        return [c.display_name for c in ordered]

    def _compute_class_weights_from_original_train(self, class_mapping: dict[str, int]) -> torch.Tensor:
        source = self.config.path("paths.class_weights_source")
        dataset = PlantDiseaseDataset(root=source, class_to_idx=class_mapping, transform=None)
        logger.info("Class weights computed from original train split (%d samples)", len(dataset))
        return dataset.compute_class_weights()

    def _run_train_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler | None,
        grad_clip: float,
    ) -> tuple[float, float]:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        use_amp = scaler is not None and self.device.type == "cuda"

        for images, labels in tqdm(loader, desc="Train", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _run_val_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, dict]:
        model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Val", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        metrics = compute_classification_metrics(
            np.array(all_labels),
            np.array(all_preds),
            self._class_names(),
        )
        metrics["val_loss"] = avg_loss
        return avg_loss, metrics

    def _save_class_mapping(self, class_mapping: dict[str, int], model_name: str, image_size: int) -> None:
        report_dir = self.paths.reports
        report_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "crop": "tomato",
            "num_classes": self.config.num_classes,
            "classes": [
                {
                    "id": c.id,
                    "folder_name": c.folder_name,
                    "display_name": c.display_name,
                    "slug": c.slug,
                }
                for c in self.config.class_configs
            ],
            "folder_to_id": class_mapping,
            "id_to_display": {str(c.id): c.display_name for c in self.config.class_configs},
            "model_name": model_name,
            "image_size": image_size,
        }
        path = report_dir / "class_mapping.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Class mapping saved: %s", path)

    def _plot_training_curves(self, history: list[dict], output_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib unavailable — skipping training curves plot")
            return

        epochs = [h["epoch"] + 1 for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        train_acc = [h["train_accuracy"] for h in history]
        val_acc = [h["val_accuracy"] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, train_loss, label="Train")
        axes[0].plot(epochs, val_loss, label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, train_acc, label="Train")
        axes[1].plot(epochs, val_acc, label="Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_confusion_matrix(self, cm: list[list[int]], class_names: list[str], output_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib unavailable — skipping confusion matrix plot")
            return

        matrix = np.array(cm)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap="Greens")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Validation Confusion Matrix (Best Epoch)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def train(self) -> TomatoTrainingResult:
        result = TomatoTrainingResult(success=False)
        set_seed(int(self.config.get("project.seed", 42)))

        train_dir = self.paths.train
        val_dir = self.paths.val
        self.paths.weights.mkdir(parents=True, exist_ok=True)
        self.paths.logs.mkdir(parents=True, exist_ok=True)

        if not train_dir.exists():
            result.errors.append(f"Training directory missing: {train_dir}")
            return result
        if not val_dir.exists():
            result.errors.append(f"Validation directory missing: {val_dir}")
            return result

        class_mapping = self._class_mapping()
        model_name = str(self.config.get("training.model_name", "efficientnet_b0"))
        num_epochs = int(self.config.get("training.num_epochs", 50))
        batch_size = int(self.config.get("training.batch_size", 32))
        num_workers = int(self.config.get("training.num_workers", 4))
        image_size = int(self.config.get("training.image_size", 256))
        lr = float(self.config.get("training.learning_rate", 0.001))
        weight_decay = float(self.config.get("training.weight_decay", 1e-4))
        mixed_precision = bool(self.config.get("training.mixed_precision", True))
        grad_clip = float(self.config.get("training.gradient_clip_norm", 1.0))
        persistent_workers = bool(self.config.get("training.persistent_workers", True))

        train_loader, val_loader, train_dataset = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            class_to_idx=class_mapping,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            pin_memory=bool(self.config.get("training.pin_memory", True)),
            persistent_workers=persistent_workers,
        )

        if len(train_dataset) == 0:
            result.errors.append("Balanced training set is empty")
            return result

        class_weights = self._compute_class_weights_from_original_train(class_mapping)
        logger.info("Class weights: %s", class_weights.tolist())

        model = ModelFactory.create(
            model_name=model_name,
            num_classes=self.config.num_classes,
            pretrained=True,
            device=self.device,
        )
        criterion = create_loss_function(
            num_classes=self.config.num_classes,
            class_weights=class_weights,
            device=self.device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, config=self.config.get("training", {}), num_epochs=num_epochs)
        scaler = GradScaler() if mixed_precision and self.device.type == "cuda" else None

        early_stop_cfg = self.config.get("training.early_stopping", {})
        early_stopping = EarlyStopping(
            patience=int(early_stop_cfg.get("patience", 8)),
            min_delta=float(early_stop_cfg.get("min_delta", 0.001)),
        )

        run_name = f"tomato_{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        tb_dir = self.paths.logs / "tensorboard" / run_name
        writer = SummaryWriter(log_dir=str(tb_dir)) if self.config.get("training.tensorboard", True) else None

        csv_path = self.paths.logs / "tomato_training_log.csv"
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                "precision_macro", "recall_macro", "f1_macro", "learning_rate", "epoch_time_sec",
            ],
        )
        csv_writer.writeheader()

        self._save_class_mapping(class_mapping, model_name, image_size)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        best_cm: list[list[int]] | None = None
        last_checkpoint: dict | None = None

        training_start = time.time()
        logger.info(
            "Tomato training | train=%s val=%s | samples=%d device=%s",
            train_dir, val_dir, len(train_dataset), self.device,
        )

        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss, train_acc = self._run_train_epoch(
                model, train_loader, criterion, optimizer, scaler, grad_clip,
            )
            val_loss, val_metrics = self._run_val_epoch(model, val_loader, criterion)
            scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            val_acc = float(val_metrics["accuracy"])

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "precision_macro": val_metrics["precision_macro"],
                "recall_macro": val_metrics["recall_macro"],
                "f1_macro": val_metrics["f1_macro"],
                "confusion_matrix": val_metrics["confusion_matrix"],
                "learning_rate": current_lr,
                "epoch_time_sec": epoch_time,
            }
            result.history.append(epoch_record)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | train_acc=%.4f val_acc=%.4f | "
                "P=%.4f R=%.4f F1=%.4f | %.1fs",
                epoch + 1, num_epochs, train_loss, val_loss, train_acc, val_acc,
                val_metrics["precision_macro"], val_metrics["recall_macro"],
                val_metrics["f1_macro"], epoch_time,
            )
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | F1: {val_metrics['f1_macro']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            csv_writer.writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{val_acc:.6f}",
                "precision_macro": f"{val_metrics['precision_macro']:.6f}",
                "recall_macro": f"{val_metrics['recall_macro']:.6f}",
                "f1_macro": f"{val_metrics['f1_macro']:.6f}",
                "learning_rate": f"{current_lr:.8f}",
                "epoch_time_sec": f"{epoch_time:.2f}",
            })
            csv_file.flush()

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Accuracy/train", train_acc, epoch)
                writer.add_scalar("Accuracy/val", val_acc, epoch)
                writer.add_scalar("F1/val_macro", val_metrics["f1_macro"], epoch)
                writer.add_scalar("LearningRate", current_lr, epoch)

            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_record,
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_cm = val_metrics["confusion_matrix"]
                best_path = self.paths.weights / "best_model.pth"
                torch.save(checkpoint_payload, best_path)
                logger.info("New best model saved (val_loss=%.4f)", val_loss)

            if early_stopping.step(val_loss, epoch):
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        csv_file.close()
        if writer:
            writer.close()

        total_time = time.time() - training_start
        last_path = self.paths.weights / "last_model.pth"
        if last_checkpoint is not None:
            torch.save(last_checkpoint, last_path)

        history_path = self.paths.logs / "tomato_training_history.json"
        history_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")

        curves_path = self.paths.weights / "training_curves.png"
        self._plot_training_curves(result.history, curves_path)

        if best_cm is not None:
            cm_path = self.paths.weights / "confusion_matrix.png"
            self._plot_confusion_matrix(best_cm, self._class_names(), cm_path)

        result.success = True
        result.best_epoch = best_epoch
        result.best_val_loss = best_val_loss
        result.best_val_accuracy = best_val_acc
        result.total_training_time_sec = total_time
        result.best_model_path = self.paths.weights / "best_model.pth"

        return result
