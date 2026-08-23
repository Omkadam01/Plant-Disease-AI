"""Model factory for EfficientNet-B0 and MobileNetV3-Large."""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
)


class SupportedModel(str, Enum):
    """Supported backbone architectures."""

    EFFICIENTNET_B0 = "efficientnet_b0"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"


def _replace_classifier(model: nn.Module, num_classes: int, in_features: int) -> nn.Module:
    """Replace the final classification head."""
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            layers = list(model.classifier.children())
            # Find last Linear layer input features
            for i in range(len(layers) - 1, -1, -1):
                if isinstance(layers[i], nn.Linear):
                    layers[i] = nn.Linear(layers[i].in_features, num_classes)
                    break
            model.classifier = nn.Sequential(*layers)
        elif isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 with custom classification head."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v3_large(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build MobileNetV3-Large with custom classification head."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a classification model by name.

    Parameters
    ----------
    model_name:
        One of ``efficientnet_b0`` or ``mobilenet_v3_large``.
    num_classes:
        Number of output classes.
    pretrained:
        Use ImageNet pretrained weights.
    """
    name = model_name.lower().replace("-", "_")
    if name == SupportedModel.EFFICIENTNET_B0.value:
        return build_efficientnet_b0(num_classes, pretrained=pretrained)
    if name == SupportedModel.MOBILENET_V3_LARGE.value:
        return build_mobilenet_v3_large(num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unsupported model: {model_name}. "
        f"Supported: {[m.value for m in SupportedModel]}"
    )


class ModelFactory:
    """Factory for creating and inspecting models."""

    @staticmethod
    def create(
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        device: torch.device | None = None,
    ) -> nn.Module:
        model = build_model(model_name, num_classes, pretrained=pretrained)
        if device is not None:
            model = model.to(device)
        return model

    @staticmethod
    def supported_models() -> list[str]:
        return [m.value for m in SupportedModel]

    @staticmethod
    def get_model_info(model_name: str) -> dict[str, Any]:
        name = model_name.lower().replace("-", "_")
        info = {
            SupportedModel.EFFICIENTNET_B0.value: {
                "name": "EfficientNet-B0",
                "params_approx": "5.3M",
                "role": "primary",
            },
            SupportedModel.MOBILENET_V3_LARGE.value: {
                "name": "MobileNetV3-Large",
                "params_approx": "5.4M",
                "role": "baseline",
            },
        }
        if name not in info:
            raise ValueError(f"Unknown model: {model_name}")
        return info[name]
