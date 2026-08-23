"""Lightweight classification metrics without sklearn dependency."""

from __future__ import annotations

import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1

    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []

    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        if support > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
