#!/usr/bin/env python3
"""Generate confusion matrix plots for grape and tomato models."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sanitize_import_path() -> None:
    vendor_root = (PROJECT_ROOT / ".vendor").resolve()
    sys.path = [e for e in sys.path if e and Path(e).resolve() != vendor_root]


_sanitize_import_path()
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _load_class_names(crop: str) -> list[str]:
    mapping_path = PROJECT_ROOT / f"Dataset/datasets/{crop}/reports/class_mapping.json"
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    id_to_display = data["id_to_display"]
    return [id_to_display[str(i)] for i in range(len(id_to_display))]


def plot_cm(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_grape() -> Path:
    class_names = _load_class_names("grape")
    metrics_path = PROJECT_ROOT / "evaluation/grape/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cm = np.array(metrics["confusion_matrix"], dtype=int)

    out = PROJECT_ROOT / "Outputs/Grape/confusion_matrix.png"
    plot_cm(cm, class_names, "Grape Test Confusion Matrix", out, figsize=(8, 6))

    canonical = PROJECT_ROOT / "evaluation/grape/confusion_matrix.png"
    plot_cm(cm, class_names, "Grape Test Confusion Matrix", canonical, figsize=(8, 6))
    return out


def generate_tomato() -> Path:
    class_names = _load_class_names("tomato")
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    csv_path = PROJECT_ROOT / "Outputs/Tomato/tomato_test_predictions.csv"
    y_true: list[int] = []
    y_pred: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            y_true.append(name_to_idx[row["ground_truth"]])
            y_pred.append(name_to_idx[row["predicted_class"]])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    out = PROJECT_ROOT / "Outputs/Tomato/confusion_matrix.png"
    plot_cm(cm, class_names, "Tomato Test Confusion Matrix", out, figsize=(12, 10))

    canonical = PROJECT_ROOT / "Outputs/Tomato/confusion_matrix.png"
    plot_cm(cm, class_names, "Tomato Test Confusion Matrix", canonical, figsize=(12, 10))
    return out


def main() -> int:
    grape_path = generate_grape()
    tomato_path = generate_tomato()
    print(f"Grape confusion matrix: {grape_path}")
    print(f"Tomato confusion matrix: {tomato_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
