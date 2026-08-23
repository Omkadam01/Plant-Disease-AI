"""Explainable inference with integrated Grad-CAM."""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from inference.gradcam import GradCAMOutputs, generate_gradcam
from inference.predictor import Predictor, PredictionResult, TopPrediction
from training.transforms import IMAGENET_MEAN, IMAGENET_STD
from utils.image_utils import read_image_rgb
from utils.platform import is_raspberry_pi

logger = logging.getLogger(__name__)


@dataclass
class ExplainablePredictionResult(PredictionResult):
    """Prediction result extended with Grad-CAM explainability artifacts."""

    heatmap_path: str | None = None
    overlay_path: str | None = None
    original_output_path: str | None = None
    gradcam_output_dir: str | None = None
    heatmap: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "heatmap_path": self.heatmap_path,
            "overlay_path": self.overlay_path,
            "original_output_path": self.original_output_path,
            "gradcam_output_dir": self.gradcam_output_dir,
        })
        return base


class ExplainablePredictor(Predictor):
    """
    PyTorch inference predictor with automatic Grad-CAM generation.

    Loads crop-specific weights and class mapping from configuration.
    Every call to ``predict_with_gradcam`` runs standard prediction first,
    then generates a Top-1 Grad-CAM overlay without altering predictions.
    """

    def __init__(
        self,
        config,
        weights_path: Path | None = None,
        gradcam_output_root: Path | None = None,
        overlay_alpha: float = 0.45,
    ) -> None:
        super().__init__(
            config,
            weights_path=weights_path,
            backend="pytorch",
        )
        self.gradcam_output_root = gradcam_output_root or (
            config.project_root / "Outputs" / "GradCAM" / config.crop_name
        )
        self.overlay_alpha = overlay_alpha

    def _gradcam_enabled(self) -> bool:
        env = os.getenv("PLANT_DISEASE_ENABLE_GRADCAM", "").strip().lower()
        if env in {"0", "false", "no", "off"}:
            return False
        if env in {"1", "true", "yes", "on"}:
            return True
        return bool(self.config.get("inference.enable_gradcam", True))

    def _resolve_output_dir(self, image_path: Path, output_dir: Path | None) -> Path:
        if output_dir is not None:
            return output_dir
        return self.gradcam_output_root / image_path.stem

    def _prediction_only_result(
        self,
        path: Path,
        base_result: PredictionResult,
        out_dir: Path,
        *,
        original_path: Path | None = None,
    ) -> ExplainablePredictionResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_original = original_path or path
        if original_path is None:
            try:
                import cv2

                rgb = read_image_rgb(path)
                saved_original = out_dir / "original.jpg"
                cv2.imwrite(str(saved_original), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                del rgb
            except Exception:
                saved_original = path

        return ExplainablePredictionResult(
            predicted_class=base_result.predicted_class,
            predicted_class_id=base_result.predicted_class_id,
            confidence=base_result.confidence,
            top_predictions=list(base_result.top_predictions),
            inference_time_ms=base_result.inference_time_ms,
            image_path=str(path),
            heatmap_path=None,
            overlay_path=None,
            original_output_path=str(saved_original),
            gradcam_output_dir=str(out_dir),
            heatmap=None,
        )

    @staticmethod
    def _tensor_to_display_rgb(tensor: torch.Tensor) -> np.ndarray:
        """Rebuild a small RGB preview from the model input tensor (Pi-safe)."""
        arr = tensor.detach().float().cpu()
        if arr.dim() == 4:
            arr = arr[0]
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        arr = arr * std + mean
        arr = arr.clamp(0, 1).permute(1, 2, 0).numpy()
        return (arr * 255.0).astype(np.uint8)

    def predict_with_gradcam(
        self,
        image_path: Path | str,
        output_dir: Path | str | None = None,
    ) -> ExplainablePredictionResult:
        """
        Run prediction and generate Grad-CAM for the Top-1 predicted class.

        On Raspberry Pi, Grad-CAM uses a low-RAM path (model-size overlay only)
        and falls back to prediction-only if Grad-CAM fails.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        base_result = self.predict(path)
        out_dir = self._resolve_output_dir(path, Path(output_dir) if output_dir else None)

        if not self._gradcam_enabled():
            logger.info("Grad-CAM disabled; returning prediction only")
            return self._prediction_only_result(path, base_result, out_dir)

        if self._model is None or not isinstance(self._model, torch.nn.Module):
            raise RuntimeError("Grad-CAM requires the PyTorch backend and a loaded nn.Module.")

        on_pi = is_raspberry_pi()
        try:
            tensor = self._preprocess(path)
            if on_pi:
                # Avoid decoding a second full-resolution copy for overlay.
                display_rgb = self._tensor_to_display_rgb(tensor)
                display_max_side = 224
                use_external_lib = False
            else:
                display_rgb = read_image_rgb(path)
                raw_max = self.config.get("inference.pi_gradcam_max_side")
                try:
                    display_max_side = int(raw_max) if raw_max is not None else None
                except (TypeError, ValueError):
                    display_max_side = None
                use_external_lib = True

            gc.collect()
            gradcam_start = time.perf_counter()
            gradcam_outputs: GradCAMOutputs = generate_gradcam(
                model=self._model,
                input_tensor=tensor,
                target_class=base_result.predicted_class_id,
                original_rgb=display_rgb,
                output_dir=out_dir,
                device=self.device,
                alpha=self.overlay_alpha,
                display_max_side=display_max_side,
                use_external_lib=use_external_lib,
            )
            gradcam_ms = (time.perf_counter() - gradcam_start) * 1000

            del display_rgb, tensor
            gc.collect()

            return ExplainablePredictionResult(
                predicted_class=base_result.predicted_class,
                predicted_class_id=base_result.predicted_class_id,
                confidence=base_result.confidence,
                top_predictions=list(base_result.top_predictions),
                inference_time_ms=base_result.inference_time_ms + gradcam_ms,
                image_path=str(path),
                heatmap_path=str(gradcam_outputs.heatmap_path),
                overlay_path=str(gradcam_outputs.overlay_path),
                original_output_path=str(gradcam_outputs.original_path),
                gradcam_output_dir=str(out_dir),
                heatmap=None,
            )
        except Exception as exc:
            logger.exception("Grad-CAM failed; returning prediction only | error=%s", exc)
            gc.collect()
            return self._prediction_only_result(path, base_result, out_dir)

    def predict(self, image_path: Path | str) -> PredictionResult:
        """Standard prediction without Grad-CAM (backward compatible)."""
        return super().predict(image_path)
