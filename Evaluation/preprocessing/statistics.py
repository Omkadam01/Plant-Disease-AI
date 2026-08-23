"""Dataset image statistics computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.image_utils import read_image_rgb


@dataclass
class ClassStatistics:
    """Per-class image statistics."""

    class_name: str
    count: int = 0
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    mean_rgb: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std_rgb: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "count": self.count,
            "width": {
                "min": int(min(self.widths)) if self.widths else 0,
                "max": int(max(self.widths)) if self.widths else 0,
                "mean": float(np.mean(self.widths)) if self.widths else 0.0,
            },
            "height": {
                "min": int(min(self.heights)) if self.heights else 0,
                "max": int(max(self.heights)) if self.heights else 0,
                "mean": float(np.mean(self.heights)) if self.heights else 0.0,
            },
            "mean_rgb": self.mean_rgb,
            "std_rgb": self.std_rgb,
        }


@dataclass
class DatasetStatistics:
    """Aggregate dataset statistics."""

    total_images: int = 0
    class_stats: dict[str, ClassStatistics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "classes": {name: stats.to_dict() for name, stats in self.class_stats.items()},
        }


def compute_statistics(
    labeled_paths: dict[str, list[Path]],
    sample_limit_per_class: int | None = None,
) -> DatasetStatistics:
    """
    Compute per-class and aggregate image statistics.

    Parameters
    ----------
    labeled_paths:
        Mapping of class folder name to list of image paths.
    sample_limit_per_class:
        Optional limit for RGB mean/std computation (performance).
    """
    result = DatasetStatistics()

    for class_name, paths in labeled_paths.items():
        stats = ClassStatistics(class_name=class_name, count=len(paths))
        result.total_images += len(paths)

        rgb_values: list[np.ndarray] = []
        sample_paths = paths[:sample_limit_per_class] if sample_limit_per_class else paths

        for path in paths:
            try:
                image = read_image_rgb(path)
                h, w = image.shape[:2]
                stats.widths.append(w)
                stats.heights.append(h)
            except Exception:
                continue

        for path in sample_paths:
            try:
                image = read_image_rgb(path)
                rgb_values.append(image.reshape(-1, 3).astype(np.float32))
            except Exception:
                continue

        if rgb_values:
            stacked = np.concatenate(rgb_values, axis=0)
            stats.mean_rgb = stacked.mean(axis=0).tolist()
            stats.std_rgb = stacked.std(axis=0).tolist()

        result.class_stats[class_name] = stats

    return result
