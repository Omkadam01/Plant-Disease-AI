"""Model evaluation module."""

__all__ = ["Evaluator", "EvaluationResult"]


def __getattr__(name: str):
    if name == "Evaluator":
        from evaluation.evaluator import Evaluator

        return Evaluator
    if name == "EvaluationResult":
        from evaluation.evaluator import EvaluationResult

        return EvaluationResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
