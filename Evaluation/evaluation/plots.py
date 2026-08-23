"""Evaluation visualization plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evaluation.metrics import MetricsResult


def plot_confusion_matrix(
    metrics: MetricsResult,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save confusion matrix heatmap."""
    cm = np.array(metrics.confusion_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
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
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    metrics: MetricsResult,
    output_path: Path,
) -> None:
    """Save ROC curves for all classes."""
    if not metrics.roc_curves:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for class_name, curve in metrics.roc_curves.items():
        auc = metrics.roc_auc.get(class_name, 0.0) if metrics.roc_auc else 0.0
        ax.plot(curve["fpr"], curve["tpr"], label=f"{class_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curves(
    metrics: MetricsResult,
    output_path: Path,
) -> None:
    """Save per-class precision-recall curves."""
    if not metrics.pr_curves:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for class_name, curve in metrics.pr_curves.items():
        ap = metrics.pr_auc.get(class_name, 0.0) if metrics.pr_auc else 0.0
        ax.plot(curve["recall"], curve["precision"], label=f"{class_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
