"""Corrupted image detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from utils.config import AppConfig
from utils.image_utils import ImageValidationResult, collect_image_paths, validate_image

logger = logging.getLogger(__name__)


@dataclass
class CorruptionCheckResult:
    """Results from corrupted image scan."""

    total_checked: int = 0
    valid_count: int = 0
    corrupted: list[ImageValidationResult] = field(default_factory=list)
    valid_images: list[Path] = field(default_factory=list)

    @property
    def corrupted_count(self) -> int:
        return len(self.corrupted)


def detect_corrupted_images(
    image_paths: list[Path],
    config: AppConfig,
) -> CorruptionCheckResult:
    """Scan images and identify corrupted or invalid files."""
    min_size = int(config.get("preprocessing.min_image_size", 32))
    max_size = int(config.get("preprocessing.max_image_size", 4096))

    result = CorruptionCheckResult(total_checked=len(image_paths))

    for path in image_paths:
        validation = validate_image(path, min_size=min_size, max_size=max_size)
        if validation.is_valid:
            result.valid_count += 1
            result.valid_images.append(path)
        else:
            result.corrupted.append(validation)
            logger.warning("Corrupted image: %s - %s", path, validation.error)

    return result
