"""Loss functions for training."""

from __future__ import annotations

import torch
import torch.nn as nn


def create_loss_function(
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Create weighted cross-entropy loss.

    Parameters
    ----------
    class_weights:
        Per-class weights tensor. If None, uses uniform weights.
    """
    weight = None
    if class_weights is not None:
        weight = class_weights.to(device) if device else class_weights

    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
