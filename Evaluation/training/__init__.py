"""Training module for PlantDiseaseAI."""

__all__ = ["Trainer", "TrainingResult"]


def __getattr__(name: str):
    if name == "Trainer":
        from training.trainer import Trainer

        return Trainer
    if name == "TrainingResult":
        from training.trainer import TrainingResult

        return TrainingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
