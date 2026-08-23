"""Grad-CAM explainability for EfficientNet-B0 inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradCAMOutputs:
    """Grad-CAM visualization artifacts."""

    heatmap: np.ndarray
    overlay: np.ndarray
    heatmap_path: Path
    overlay_path: Path
    original_path: Path


def get_efficientnet_b0_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the final convolutional layer of EfficientNet-B0.

    Targets the 1x1 conv inside the last feature block (1280 channels).
    """
    if not hasattr(model, "features"):
        raise ValueError("Model does not expose a 'features' module for Grad-CAM.")
    last_block = model.features[-1]
    if isinstance(last_block, nn.Sequential):
        return last_block[0]
    if hasattr(last_block, "__getitem__"):
        return last_block[0]
    return last_block


class GradCAM:
    """
    Native Grad-CAM implementation for PyTorch classification models.

    Computes class-discriminative localization maps by weighting feature-map
    activations with global-average-pooled gradients of the target class score.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            self._activations = output

        def backward_hook(_module: nn.Module, _grad_input: tuple, grad_output: tuple) -> None:
            self._gradients = grad_output[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def close(self) -> None:
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate a normalized Grad-CAM heatmap for the given class.

        Returns a 2D float array in [0, 1].
        """
        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None

        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad(set_to_none=True)
        score = output[0, target_class]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        gradients = self._gradients[0]
        activations = self._activations[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_np = _normalize_cam(cam_np)
        return cam_np


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def apply_colormap(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Apply OpenCV colormap to a normalized heatmap. Returns RGB uint8 image."""
    heatmap_uint8 = (cam * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def create_overlay(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend Grad-CAM heatmap onto the original image.

    Parameters
    ----------
    original_rgb:
        Original image in RGB, any spatial size.
    cam:
        Normalized 2D heatmap from Grad-CAM (model input resolution).
    alpha:
        Heatmap blend weight.
    """
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_rgb = apply_colormap(cam_resized)

    original = original_rgb.astype(np.float32)
    heatmap = heatmap_rgb.astype(np.float32)
    overlay = (1.0 - alpha) * original + alpha * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_gradcam_outputs(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    output_dir: Path,
    alpha: float = 0.45,
) -> GradCAMOutputs:
    """
    Save original, heatmap, and overlay images to ``output_dir``.

    Creates ``original.jpg``, ``heatmap.png``, and ``overlay.png``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / "original.jpg"
    heatmap_path = output_dir / "heatmap.png"
    overlay_path = output_dir / "overlay.png"

    Image.fromarray(original_rgb).save(original_path, quality=95)

    heatmap_rgb = apply_colormap(cv2.resize(cam, (original_rgb.shape[1], original_rgb.shape[0])))
    Image.fromarray(heatmap_rgb).save(heatmap_path)

    overlay = create_overlay(original_rgb, cam, alpha=alpha)
    Image.fromarray(overlay).save(overlay_path, quality=95)

    return GradCAMOutputs(
        heatmap=cam,
        overlay=overlay,
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
        original_path=original_path,
    )


def try_pytorch_grad_cam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    rgb_image: np.ndarray,
) -> np.ndarray | None:
    """
    Optional wrapper around ``pytorch-grad-cam`` if installed and compatible.

    Returns normalized CAM or None if the library is unavailable.
    """
    try:
        from pytorch_grad_cam import GradCAM as LibGradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None

    try:
        cam_extractor = LibGradCAM(model=model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(target_class)]
        tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        grayscale_cam = cam_extractor(input_tensor=tensor, targets=targets)
        return _normalize_cam(grayscale_cam[0])
    except Exception as exc:
        logger.debug("pytorch-grad-cam unavailable, using native Grad-CAM: %s", exc)
        return None


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_rgb: np.ndarray,
    output_dir: Path,
    device: torch.device,
    target_layer_fn: Callable[[nn.Module], nn.Module] | None = None,
    alpha: float = 0.45,
    display_max_side: int | None = None,
    *,
    use_external_lib: bool = True,
) -> GradCAMOutputs:
    """
    End-to-end Grad-CAM generation and save.

    Attempts ``pytorch-grad-cam`` first (unless disabled), falls back to native.
    ``display_max_side`` downscales the saved overlay on low-RAM devices (Pi).
    """
    import gc

    layer_fn = target_layer_fn or get_efficientnet_b0_target_layer
    target_layer = layer_fn(model)

    display_rgb = original_rgb
    if display_max_side is not None:
        h, w = original_rgb.shape[:2]
        longest = max(h, w)
        if longest > display_max_side:
            scale = display_max_side / float(longest)
            display_rgb = cv2.resize(
                original_rgb,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

    cam = None
    if use_external_lib:
        cam = try_pytorch_grad_cam(model, target_layer, input_tensor, target_class, display_rgb)

    if cam is None:
        model.eval()
        with GradCAM(model=model, target_layer=target_layer, device=device) as grad_cam:
            cam = grad_cam.generate(input_tensor, target_class)

    model.zero_grad(set_to_none=True)
    outputs = save_gradcam_outputs(display_rgb, cam, output_dir, alpha=alpha)
    del cam, display_rgb
    gc.collect()
    return outputs
