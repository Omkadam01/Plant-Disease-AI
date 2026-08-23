"""Inference service — UI-facing wrapper around the AI pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from inference.explainable_predictor import ExplainablePredictionResult, ExplainablePredictor
from utils.config import AppConfig

logger = logging.getLogger(__name__)

INFERENCE_DEBUG = os.getenv("PLANT_DISEASE_INFERENCE_DEBUG", "").lower() in ("1", "true", "yes")


class InferenceService:
    """
    Thin service layer exposing ``predict(image)`` to the desktop UI.

    Delegates to ``ExplainablePredictor`` without modifying inference code.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._predictor: ExplainablePredictor | None = None

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def model_name(self) -> str:
        if self._predictor is None:
            return "Not loaded"
        return str(self.config.get("training.model_name", "efficientnet_b0"))

    @property
    def device_label(self) -> str:
        if self._predictor is None:
            return "—"
        device = self._predictor.device
        if device.type == "cuda":
            return "GPU"
        if device.type == "mps":
            return "Apple GPU"
        return "CPU"

    @property
    def crop_name(self) -> str:
        return self.config.crop_name

    @property
    def weights_path(self) -> Path | None:
        if self._predictor is None:
            return None
        return self._predictor.weights_path

    @property
    def class_names_path(self) -> Path | None:
        if self._predictor is None:
            return None
        return getattr(self._predictor, "class_names_path", None)

    def load_model(self) -> None:
        """Load EfficientNet-B0 weights and prepare the predictor."""
        self._predictor = ExplainablePredictor(self.config)

    def predict(self, image_path: Path | str) -> ExplainablePredictionResult:
        """
        Run prediction with Grad-CAM on the given image.

        Raises
        ------
        RuntimeError
            If the model has not been loaded yet.
        """
        if self._predictor is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if INFERENCE_DEBUG:
            logger.info(
                "[INFERENCE_DEBUG] desktop_predict crop=%s weights=%s class_mapping=%s image=%s",
                self.crop_name,
                self.weights_path,
                self.class_names_path,
                image_path,
            )

        return self._predictor.predict_with_gradcam(image_path)
