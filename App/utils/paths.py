"""Project path utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.config import AppConfig


def resolve_path(base: Path, *parts: str) -> Path:
    """Resolve and create parent directories if needed."""
    path = (base / Path(*parts)).resolve()
    return path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project directory paths."""

    root: Path
    logs: Path
    weights: Path
    exports: Path
    dataset_root: Path
    raw: Path
    processed: Path
    train: Path
    val: Path
    test: Path
    reports: Path
    model_dir: Path

    @classmethod
    def from_config(cls, config: AppConfig) -> ProjectPaths:
        return cls(
            root=config.project_root,
            logs=config.path("paths.logs"),
            weights=config.path("paths.weights") / config.crop_name,
            exports=config.path("paths.exports") / config.crop_name,
            dataset_root=config.path("paths.dataset_root"),
            raw=config.path("paths.raw"),
            processed=config.path("paths.processed"),
            train=config.path("paths.train"),
            val=config.path("paths.val"),
            test=config.path("paths.test"),
            reports=config.path("paths.reports"),
            model_dir=config.path("paths.model_dir"),
        )

    def ensure_dirs(self) -> None:
        """Create all standard output directories."""
        for path in (
            self.logs,
            self.weights,
            self.exports,
            self.processed,
            self.train,
            self.val,
            self.test,
            self.reports,
            self.model_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
