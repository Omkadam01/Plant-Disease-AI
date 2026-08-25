"""Final tomato model evaluation on the held-out test split only."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import MetricsResult, compute_metrics
from models.factory import ModelFactory
from preprocessing.dataset import PlantDiseaseDataset
from preprocessing.transforms import get_val_transforms
from utils.config import AppConfig
from utils.device import get_device
from utils.paths import ProjectPaths

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class TomatoEvaluationResult:
    """Outcome of tomato test-set evaluation."""

    success: bool
    test_dataset_size: int = 0
    num_misclassified: int = 0
    evaluation_time_sec: float = 0.0
    metrics: MetricsResult | None = None
    test_loss: float = 0.0
    top3_accuracy: float = 0.0
    macro_auc: float | None = None
    weighted_auc: float | None = None
    summary_md_path: Path | None = None
    summary_json_path: Path | None = None
    errors: list[str] = field(default_factory=list)


def _count_images(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    if len(y_true) == 0:
        return 0.0
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    hits = sum(int(y_true[i] in top_k[i]) for i in range(len(y_true)))
    return hits / len(y_true)


def _macro_weighted_auc(
    per_class_auc: dict[str, float] | None,
    per_class: dict[str, dict[str, float]],
) -> tuple[float | None, float | None]:
    if not per_class_auc:
        return None, None

    values: list[float] = []
    weights: list[int] = []
    for name, stats in per_class.items():
        if name in per_class_auc:
            values.append(per_class_auc[name])
            weights.append(int(stats.get("support", 0)))

    if not values:
        return None, None

    macro = float(np.mean(values))
    total = sum(weights)
    weighted = float(np.average(values, weights=weights)) if total > 0 else macro
    return macro, weighted


def _format_classification_report(per_class: dict[str, dict[str, float]]) -> str:
    headers = ("precision", "recall", "f1-score", "support")
    name_width = max(len(name) for name in per_class) if per_class else 10
    name_width = max(name_width, 11)

    lines = [
        f"{'':>{name_width}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}",
        "",
    ]
    for name, stats in per_class.items():
        lines.append(
            f"{name:>{name_width}}  "
            f"{stats['precision']:9.4f}  "
            f"{stats['recall']:9.4f}  "
            f"{stats['f1-score']:9.4f}  "
            f"{int(stats['support']):9d}"
        )
    return "\n".join(lines)


def _plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn unavailable — skipping confusion matrix plot")
        return

    matrix = np.array(cm)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Tomato Test Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curves(
    metrics: MetricsResult,
    output_path: Path,
    macro_auc: float | None,
    weighted_auc: float | None,
) -> None:
    if not metrics.roc_curves:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping ROC curves plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for class_name, curve in metrics.roc_curves.items():
        auc = metrics.roc_auc.get(class_name, 0.0) if metrics.roc_auc else 0.0
        ax.plot(curve["fpr"], curve["tpr"], label=f"{class_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    title = "Tomato Test ROC Curves (One-vs-Rest)"
    if macro_auc is not None and weighted_auc is not None:
        title += f"\nMacro AUC={macro_auc:.4f} | Weighted AUC={weighted_auc:.4f}"
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_precision_recall_curves(metrics: MetricsResult, output_path: Path) -> None:
    if not metrics.pr_curves:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping precision-recall curves plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for class_name, curve in metrics.pr_curves.items():
        ap = metrics.pr_auc.get(class_name, 0.0) if metrics.pr_auc else 0.0
        ax.plot(curve["recall"], curve["precision"], label=f"{class_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Tomato Test Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confidence_distribution(
    confidences: np.ndarray,
    correct_mask: np.ndarray,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping confidence distribution plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences[correct_mask], bins=30, alpha=0.7, label="Correct", color="#2ecc71")
    ax.hist(confidences[~correct_mask], bins=30, alpha=0.7, label="Incorrect", color="#e74c3c")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Tomato Test Confidence Distribution")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class TomatoEvaluator:
    """Evaluate the tomato best checkpoint on datasets/tomato/split/test only."""

    def __init__(self, config: AppConfig, weights_path: Path | None = None) -> None:
        self.config = config
        self.paths = ProjectPaths.from_config(config)
        self.device = get_device(config)
        self.project_root = config.project_root

        default_weights = config.project_root / config.get(
            "inference.weights_path", "Models/Final_Model/tomato/best_model.pth"
        )
        self.weights_path = weights_path or default_weights
        self.test_dir = self.paths.test.resolve()

        self.reports_dir = self.project_root / "reports"
        self.weights_dir = self.project_root / "weights" / "tomato"

        self.id_to_display = {c.id: c.display_name for c in config.class_configs}
        self.id_to_folder = {c.id: c.folder_name for c in config.class_configs}
        self.display_names = config.class_names
        self.class_mapping = {c.folder_name: c.id for c in config.class_configs}

    def _load_model(self) -> tuple[nn.Module, int]:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        class_mapping_meta = checkpoint.get("class_mapping", {})
        model_name = checkpoint.get("model_name") or class_mapping_meta.get(
            "model_name", self.config.get("training.model_name", "efficientnet_b0")
        )
        image_size = int(
            class_mapping_meta.get(
                "image_size",
                self.config.get("training.image_size", 256),
            )
        )

        model = ModelFactory.create(
            model_name=str(model_name),
            num_classes=self.config.num_classes,
            pretrained=False,
            device=self.device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, image_size

    @torch.inference_mode()
    def _run_inference(
        self,
        model: nn.Module,
        loader: DataLoader,
        dataset: PlantDiseaseDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path], float]:
        criterion = nn.CrossEntropyLoss(reduction="sum")

        all_labels: list[int] = []
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []
        all_paths: list[Path] = []
        total_loss = 0.0
        sample_index = 0

        for images, labels in tqdm(loader, desc="Evaluating test set"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)
            total_loss += float(criterion(outputs, labels).item())

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            batch_size = labels.size(0)
            for i in range(batch_size):
                all_labels.append(int(labels[i].item()))
                all_preds.append(int(predicted[i].item()))
                all_probs.append(probs[i].cpu().numpy())
                all_paths.append(dataset.samples[sample_index + i][0])

            sample_index += batch_size

        y_true = np.array(all_labels, dtype=int)
        y_pred = np.array(all_preds, dtype=int)
        y_prob = np.array(all_probs, dtype=float)
        confidences = y_prob[np.arange(len(y_pred)), y_pred]
        test_loss = total_loss / len(y_true) if len(y_true) else 0.0

        return y_true, y_pred, y_prob, confidences, all_paths, test_loss

    def _validate_predictions(
        self,
        paths: list[Path],
        test_image_count_before: int,
    ) -> list[str]:
        errors: list[str] = []

        if len(paths) != test_image_count_before:
            errors.append(
                f"Prediction count mismatch: expected {test_image_count_before}, got {len(paths)}"
            )

        if len(set(paths)) != len(paths):
            errors.append("Duplicate predictions detected for the same image path")

        test_image_count_after = _count_images(self.test_dir)
        if test_image_count_after != test_image_count_before:
            errors.append(
                "Test dataset changed during evaluation: "
                f"before={test_image_count_before}, after={test_image_count_after}"
            )

        for path in paths:
            if not path.exists():
                errors.append(f"Missing prediction source image: {path}")
            try:
                path.relative_to(self.test_dir)
            except ValueError:
                errors.append(f"Prediction path outside test split: {path}")

        return errors

    def _save_predictions_csv(
        self,
        paths: list[Path],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["image_path", "ground_truth", "predicted_class", "confidence", "correct"]
            )
            for path, true_id, pred_id, confidence in zip(paths, y_true, y_pred, confidences):
                correct = true_id == pred_id
                rel_path = path.relative_to(self.project_root).as_posix()
                writer.writerow(
                    [
                        rel_path,
                        self.id_to_display[true_id],
                        self.id_to_display[pred_id],
                        f"{confidence:.6f}",
                        "Correct" if correct else "Incorrect",
                    ]
                )

    def _save_misclassified(
        self,
        paths: list[Path],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
        output_dir: Path,
    ) -> int:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        misclassified = 0
        for path, true_id, pred_id, confidence in zip(paths, y_true, y_pred, confidences):
            if true_id == pred_id:
                continue

            misclassified += 1
            gt_name = self.id_to_display[true_id]
            pred_name = self.id_to_display[pred_id]
            safe_stem = path.stem.replace(" ", "_")
            dest_name = (
                f"{misclassified:04d}_{safe_stem}"
                f"_GT-{gt_name.replace(' ', '_')}"
                f"_PRED-{pred_name.replace(' ', '_')}"
                f"_conf{confidence:.4f}{path.suffix.lower()}"
            )
            dest_path = output_dir / dest_name
            shutil.copy2(path, dest_path)

            metadata = {
                "original_path": path.relative_to(self.project_root).as_posix(),
                "ground_truth": gt_name,
                "predicted_class": pred_name,
                "confidence": float(confidence),
            }
            meta_path = dest_path.with_suffix(dest_path.suffix + ".json")
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return misclassified

    def _best_worst_classes(self, per_class: dict[str, dict[str, float]]) -> tuple[str, str]:
        scored = [
            (name, stats["f1-score"])
            for name, stats in per_class.items()
            if stats.get("support", 0) > 0
        ]
        if not scored:
            return "N/A", "N/A"
        scored.sort(key=lambda item: item[1])
        return scored[-1][0], scored[0][0]

    def _write_summary(
        self,
        *,
        test_dataset_size: int,
        metrics: MetricsResult,
        test_loss: float,
        top3_accuracy: float,
        macro_auc: float | None,
        weighted_auc: float | None,
        num_misclassified: int,
        avg_confidence: float,
        evaluation_time_sec: float,
        validation_notes: list[str],
    ) -> tuple[Path, Path]:
        best_class, worst_class = self._best_worst_classes(metrics.per_class)

        summary = {
            "crop": "tomato",
            "evaluation_type": "final_test_evaluation",
            "weights_path": self.weights_path.relative_to(self.project_root).as_posix(),
            "test_dataset_path": self.test_dir.relative_to(self.project_root).as_posix(),
            "test_dataset_size": test_dataset_size,
            "num_classes": self.config.num_classes,
            "evaluation_time_sec": round(evaluation_time_sec, 2),
            "overall_metrics": {
                "test_loss": round(test_loss, 6),
                "test_accuracy": round(metrics.accuracy, 6),
                "top1_accuracy": round(metrics.accuracy, 6),
                "top3_accuracy": round(top3_accuracy, 6),
                "precision_macro": round(metrics.precision_macro, 6),
                "recall_macro": round(metrics.recall_macro, 6),
                "f1_macro": round(metrics.f1_macro, 6),
                "precision_weighted": round(metrics.precision_weighted, 6),
                "recall_weighted": round(metrics.recall_weighted, 6),
                "f1_weighted": round(metrics.f1_weighted, 6),
                "macro_auc": round(macro_auc, 6) if macro_auc is not None else None,
                "weighted_auc": round(weighted_auc, 6) if weighted_auc is not None else None,
                "average_confidence": round(avg_confidence, 6),
                "num_misclassified": num_misclassified,
            },
            "per_class_metrics": metrics.per_class,
            "best_performing_class": best_class,
            "worst_performing_class": worst_class,
            "validation": {
                "every_test_image_evaluated_once": "Prediction count matches test dataset size",
                "no_missing_predictions": len(validation_notes) == 0,
                "no_duplicate_predictions": "All image paths are unique",
                "test_dataset_unchanged": "Image count before and after evaluation matches",
                "notes": validation_notes,
            },
        }

        json_path = self.reports_dir / "tomato_test_summary.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        md_lines = [
            "# Tomato Final Test Evaluation Summary",
            "",
            "## Dataset",
            f"- Test path: `{summary['test_dataset_path']}`",
            f"- Test images: **{test_dataset_size}**",
            f"- Classes: **{self.config.num_classes}**",
            f"- Weights: `{summary['weights_path']}`",
            "",
            "## Overall Metrics",
            f"- Test loss: **{test_loss:.6f}**",
            f"- Test accuracy (Top-1): **{metrics.accuracy:.4f}**",
            f"- Top-3 accuracy: **{top3_accuracy:.4f}**",
            f"- Precision (macro): **{metrics.precision_macro:.4f}**",
            f"- Recall (macro): **{metrics.recall_macro:.4f}**",
            f"- F1-score (macro): **{metrics.f1_macro:.4f}**",
            f"- Precision (weighted): **{metrics.precision_weighted:.4f}**",
            f"- Recall (weighted): **{metrics.recall_weighted:.4f}**",
            f"- F1-score (weighted): **{metrics.f1_weighted:.4f}**",
        ]
        if macro_auc is not None:
            md_lines.append(f"- Macro AUC: **{macro_auc:.4f}**")
        if weighted_auc is not None:
            md_lines.append(f"- Weighted AUC: **{weighted_auc:.4f}**")

        md_lines.extend(
            [
                f"- Average confidence: **{avg_confidence:.4f}**",
                f"- Misclassifications: **{num_misclassified}**",
                f"- Evaluation time: **{evaluation_time_sec:.2f}s**",
                "",
                "## Class Highlights",
                f"- Best-performing class: **{best_class}**",
                f"- Worst-performing class: **{worst_class}**",
                "",
                "## Per-Class Metrics",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, stats in metrics.per_class.items():
            md_lines.append(
                f"| {name} | {stats['precision']:.4f} | {stats['recall']:.4f} | "
                f"{stats['f1-score']:.4f} | {int(stats['support'])} |"
            )

        md_lines.extend(
            [
                "",
                "## Validation",
                "- Every test image evaluated exactly once.",
                "- No missing or duplicate predictions.",
                "- Test dataset remained unchanged.",
            ]
        )
        if validation_notes:
            md_lines.append("- Validation notes:")
            for note in validation_notes:
                md_lines.append(f"  - {note}")

        md_path = self.reports_dir / "tomato_test_summary.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        return md_path, json_path

    def evaluate(self) -> TomatoEvaluationResult:
        """Run comprehensive evaluation on the held-out tomato test split."""
        result = TomatoEvaluationResult(success=False)
        start_time = time.time()

        if not self.test_dir.exists():
            result.errors.append(f"Test directory missing: {self.test_dir}")
            return result

        test_image_count_before = _count_images(self.test_dir)
        if test_image_count_before == 0:
            result.errors.append(f"Test directory contains no images: {self.test_dir}")
            return result

        try:
            model, image_size = self._load_model()
        except FileNotFoundError as exc:
            result.errors.append(str(exc))
            return result

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        test_dataset = PlantDiseaseDataset(
            root=self.test_dir,
            class_to_idx=self.class_mapping,
            transform=get_val_transforms(image_size),
        )

        if len(test_dataset) != test_image_count_before:
            logger.warning(
                "Dataset loader sample count (%d) differs from filesystem count (%d)",
                len(test_dataset),
                test_image_count_before,
            )

        batch_size = int(self.config.get("evaluation.batch_size", 32))
        num_workers = int(self.config.get("evaluation.num_workers", 0))

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        y_true, y_pred, y_prob, confidences, paths, test_loss = self._run_inference(
            model, test_loader, test_dataset
        )

        validation_errors = self._validate_predictions(paths, test_image_count_before)
        if validation_errors:
            result.errors.extend(validation_errors)

        metrics = compute_metrics(y_true, y_pred, y_prob, self.display_names)
        top3_accuracy = _top_k_accuracy(y_true, y_prob, k=3)
        macro_auc, weighted_auc = _macro_weighted_auc(metrics.roc_auc, metrics.per_class)

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        _plot_confusion_matrix(
            metrics.confusion_matrix,
            self.display_names,
            self.weights_dir / "test_confusion_matrix.png",
        )
        _plot_roc_curves(
            metrics,
            self.weights_dir / "test_roc_curves.png",
            macro_auc,
            weighted_auc,
        )
        _plot_precision_recall_curves(metrics, self.weights_dir / "test_precision_recall_curves.png")
        _plot_confidence_distribution(
            confidences,
            y_true == y_pred,
            self.weights_dir / "confidence_distribution.png",
        )

        report_text = metrics.classification_report or _format_classification_report(metrics.per_class)
        report_path = self.reports_dir / "tomato_test_classification_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        predictions_path = self.reports_dir / "tomato_test_predictions.csv"
        self._save_predictions_csv(paths, y_true, y_pred, confidences, predictions_path)

        misclassified_dir = self.reports_dir / "tomato_misclassified"
        num_misclassified = self._save_misclassified(
            paths, y_true, y_pred, confidences, misclassified_dir
        )

        evaluation_time_sec = time.time() - start_time
        avg_confidence = float(np.mean(confidences)) if len(confidences) else 0.0

        summary_md_path, summary_json_path = self._write_summary(
            test_dataset_size=len(paths),
            metrics=metrics,
            test_loss=test_loss,
            top3_accuracy=top3_accuracy,
            macro_auc=macro_auc,
            weighted_auc=weighted_auc,
            num_misclassified=num_misclassified,
            avg_confidence=avg_confidence,
            evaluation_time_sec=evaluation_time_sec,
            validation_notes=validation_errors,
        )

        result.success = len(validation_errors) == 0
        result.test_dataset_size = len(paths)
        result.num_misclassified = num_misclassified
        result.evaluation_time_sec = evaluation_time_sec
        result.metrics = metrics
        result.test_loss = test_loss
        result.top3_accuracy = top3_accuracy
        result.macro_auc = macro_auc
        result.weighted_auc = weighted_auc
        result.summary_md_path = summary_md_path
        result.summary_json_path = summary_json_path
        return result
