"""Dataset preprocessing package with lazy access to the full pipeline."""

__all__ = ["PreprocessingPipeline", "PreprocessingResult"]


def __getattr__(name: str):
    if name in __all__:
        from preprocessing.pipeline import PreprocessingPipeline, PreprocessingResult

        return {
            "PreprocessingPipeline": PreprocessingPipeline,
            "PreprocessingResult": PreprocessingResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
