"""Desktop application services."""

from desktop_app.services.inference_service import InferenceService
from desktop_app.services.groq_service import GroqExplanation, GroqExplanationService

__all__ = ["InferenceService", "GroqExplanationService", "GroqExplanation"]
