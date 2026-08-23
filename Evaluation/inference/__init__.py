"""Inference module."""

__all__ = [
    "Predictor",
    "PredictionResult",
    "ExplainablePredictor",
    "ExplainablePredictionResult",
    "GradCAM",
]


def __getattr__(name: str):
    if name == "Predictor":
        from inference.predictor import Predictor

        return Predictor
    if name == "PredictionResult":
        from inference.predictor import PredictionResult

        return PredictionResult
    if name == "ExplainablePredictor":
        from inference.explainable_predictor import ExplainablePredictor

        return ExplainablePredictor
    if name == "ExplainablePredictionResult":
        from inference.explainable_predictor import ExplainablePredictionResult

        return ExplainablePredictionResult
    if name == "GradCAM":
        from inference.gradcam import GradCAM

        return GradCAM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
