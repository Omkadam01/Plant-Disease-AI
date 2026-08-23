"""Evaluation metrics computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import auc as sk_auc


@dataclass
class MetricsResult:
    """Computed evaluation metrics."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    classification_report: str
    roc_auc: dict[str, float] | None = None
    roc_curves: dict[str, dict[str, list[float]]] | None = None
    pr_curves: dict[str, dict[str, list[float]]] | None = None
    pr_auc: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "classification_report": self.classification_report,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    class_names: list[str],
) -> MetricsResult:
    """
    Compute full classification metrics.

    Parameters
    ----------
    y_true:
        Ground truth labels.
    y_pred:
        Predicted labels.
    y_prob:
        Predicted probabilities (N, num_classes) for ROC curves.
    class_names:
        Human-readable class names.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    per_class: dict[str, dict[str, float]] = {}
    for name in class_names:
        if name in report_dict:
            per_class[name] = {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1-score": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"]),
            }

    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    roc_curves = None
    pr_curves = None
    pr_auc = None
    if y_prob is not None and y_prob.ndim == 2:
        num_classes = y_prob.shape[1]
        roc_auc = {}
        roc_curves = {}
        pr_curves = {}
        pr_auc = {}
        for i, name in enumerate(class_names[:num_classes]):
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                continue
            try:
                fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
                roc_score = float(roc_auc_score(y_binary, y_prob[:, i]))
                roc_auc[name] = roc_score
                roc_curves[name] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }
                precision, recall, _ = precision_recall_curve(y_binary, y_prob[:, i])
                pr_curves[name] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                }
                pr_auc[name] = float(sk_auc(recall, precision))
            except ValueError:
                continue

    return MetricsResult(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
        per_class=per_class,
        confusion_matrix=cm,
        classification_report=report_str,
        roc_auc=roc_auc,
        roc_curves=roc_curves,
        pr_curves=pr_curves,
        pr_auc=pr_auc,
    )
