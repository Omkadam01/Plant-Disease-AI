"""Background inference worker thread."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from inference.predictor import PredictionResult, Predictor
from utils.config import AppConfig


class InferenceWorker(QThread):
    """Run model inference in a background thread."""

    finished = Signal(object)  # PredictionResult
    error = Signal(str)

    def __init__(
        self,
        predictor: Predictor,
        image_path: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.predictor = predictor
        self.image_path = image_path

    def run(self) -> None:
        try:
            result = self.predictor.predict(self.image_path)
            self.finished.emit(result)
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n{tb}")


class PredictorLoader(QThread):
    """Load predictor model in background to avoid blocking UI startup."""

    loaded = Signal(object)  # Predictor
    error = Signal(str)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config

    def run(self) -> None:
        try:
            backend = self.config.get("desktop_app.inference_backend") or self.config.get(
                "inference.backend", "pytorch"
            )
            predictor = Predictor(self.config, backend=backend)
            self.loaded.emit(predictor)
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"{exc}\n{tb}")
