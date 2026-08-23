"""Model evaluation on test set."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import MetricsResult, compute_metrics
from evaluation.plots import plot_confusion_matrix, plot_precision_recall_curves, plot_roc_curves
from models.factory import ModelFactory
from training.dataset import PlantDiseaseDataset
from training.transforms import get_val_transforms
from utils.config import AppConfig
from utils.device import get_device
from utils.paths import ProjectPaths

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation run outcome."""

    success: bool
    metrics: MetricsResult | None = None
    metrics_path: Path | None = None
    errors: list[str] = field(default_factory=list)


class Evaluator:
    """Evaluate trained model on test dataset."""

    def __init__(self, config: AppConfig, weights_path: Path | None = None) -> None:
        self.config = config
        self.paths = ProjectPaths.from_config(config)
        self.device = get_device(config)

        if weights_path:
            self.weights_path = weights_path
        else:
            default = self.config.get(
                "inference.weights_path", "Models/weights/grape/best_model.pth"
            )
            self.weights_path = config.project_root / default

    def _load_model(self) -> tuple[nn.Module, dict]:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        class_mapping = checkpoint.get("class_mapping", {})
        model_name = checkpoint.get("model_name") or class_mapping.get(
            "model_name", self.config.get("training.model_name", "efficientnet_b0")
        )
        image_size = class_mapping.get("image_size", int(self.config.get("training.image_size", 224)))

        model = ModelFactory.create(
            model_name=str(model_name),
            num_classes=self.config.num_classes,
            pretrained=False,
            device=self.device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, {"image_size": image_size, "class_mapping": class_mapping}

    @torch.no_grad()
    def _predict(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_labels: list[int] = []
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []

        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(self.device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )

    def evaluate(self) -> EvaluationResult:
        """Run full evaluation on test set."""
        result = EvaluationResult(success=False)
        test_dir = self.paths.test

        if not test_dir.exists() or not any(test_dir.iterdir()):
            result.errors.append(
                f"Test directory empty or missing: {test_dir}. Run preprocessing first."
            )
            return result

        try:
            model, meta = self._load_model()
        except FileNotFoundError as exc:
            result.errors.append(str(exc))
            return result

        class_mapping = {c.folder_name: c.id for c in self.config.class_configs}
        image_size = meta["image_size"]

        test_dataset = PlantDiseaseDataset(
            root=test_dir,
            class_to_idx=class_mapping,
            transform=get_val_transforms(image_size),
        )

        if len(test_dataset) == 0:
            result.errors.append("Test dataset contains no samples")
            return result

        batch_size = int(self.config.get("evaluation.batch_size", 32))
        num_workers = int(self.config.get("evaluation.num_workers", 4))

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        y_true, y_pred, y_prob = self._predict(model, test_loader)
        class_names = self.config.class_names
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

        output_dir = Path(self.config.get("evaluation.output_dir", f"evaluation/{self.config.crop_name}"))
        if not output_dir.is_absolute():
            output_dir = self.config.project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        if self.config.get("evaluation.generate_plots", True):
            plot_confusion_matrix(metrics, class_names, output_dir / "confusion_matrix.png")
            plot_roc_curves(metrics, output_dir / "roc_curves.png")
            plot_precision_recall_curves(metrics, output_dir / "precision_recall_curves.png")

        report_path = output_dir / "classification_report.txt"
        report_path.write_text(metrics.classification_report, encoding="utf-8")

        if self.config.get("evaluation.save_predictions", True):
            predictions = {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist(),
            }
            with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
                json.dump(predictions, f)

        result.success = True
        result.metrics = metrics
        result.metrics_path = metrics_path

        logger.info(
            "Evaluation complete | accuracy=%.4f f1_macro=%.4f",
            metrics.accuracy,
            metrics.f1_macro,
        )

        return result
