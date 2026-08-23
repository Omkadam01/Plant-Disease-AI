"""Background workers for the desktop application."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from desktop_app.services.groq_service import GroqExplanation, GroqExplanationService
from desktop_app.services.model_manager import ModelManager
from desktop_app.services.inference_service import InferenceService
from inference.explainable_predictor import ExplainablePredictionResult

class ModelLoadWorker(QThread):
    """Load the inference model in a background thread."""

    loaded = Signal(str, object)
    error = Signal(str)

    def __init__(
        self,
        crop: str,
        model_manager: ModelManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.crop = crop
        self.model_manager = model_manager

    def run(self) -> None:
        try:
            service = self.model_manager.get_or_load(self.crop)
            self.loaded.emit(self.crop, service)
        except Exception as exc:
            self.error.emit(str(exc))


class InferenceWorker(QThread):
    """Run predict(image) in a background thread."""

    finished = Signal(int, object)
    error = Signal(str)

    def __init__(
        self,
        service: InferenceService,
        image_path: Path,
        generation: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.image_path = image_path
        self.generation = generation

    def run(self) -> None:
        try:
            result: ExplainablePredictionResult = self.service.predict(self.image_path)
            self.finished.emit(self.generation, result)
        except Exception as exc:
            self.error.emit(str(exc))


class GroqWorker(QThread):
    """Fetch Groq AI explanation in a background thread."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        service: GroqExplanationService,
        image_path: Path,
        crop_name: str,
        crop_slug: str,
        result: ExplainablePredictionResult,
        language: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.image_path = image_path
        self.crop_name = crop_name
        self.crop_slug = crop_slug
        self.result = result
        self.language = language

    def run(self) -> None:
        try:
            explanation = self.service.get_explanation(
                image_path=self.image_path,
                crop_name=self.crop_name,
                crop_slug=self.crop_slug,
                predicted_class=self.result.predicted_class,
                confidence=self.result.confidence,
                top_predictions=self.result.top_predictions,
                language=self.language,
            )
            self.finished.emit(explanation)
        except Exception as exc:
            self.error.emit(str(exc))
