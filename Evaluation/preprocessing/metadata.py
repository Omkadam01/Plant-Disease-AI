"""Dataset metadata generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from preprocessing.duplicate_detector import DuplicateDetectionResult
from preprocessing.image_checker import CorruptionCheckResult
from preprocessing.splitter import SplitResult
from preprocessing.statistics import DatasetStatistics
from preprocessing.validators import LabelValidationResult
from utils.config import AppConfig


@dataclass
class DatasetMetadata:
    """Complete dataset metadata document."""

    crop: str
    dataset_source: str
    created_at: str
    total_images: int
    num_classes: int
    classes: list[dict]
    class_mapping: dict[str, int]
    split_counts: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    statistics: dict
    validation: dict
    corruption: dict
    duplicates: dict

    def to_dict(self) -> dict:
        return asdict(self)


def generate_class_mapping(config: AppConfig) -> dict[str, int]:
    """Build folder-name to class-id mapping."""
    return {c.folder_name: c.id for c in config.class_configs}


def save_class_mapping(config: AppConfig, output_path: Path) -> None:
    """Save class mapping and display names to JSON."""
    data = {
        "crop": config.crop_name,
        "num_classes": config.num_classes,
        "classes": [
            {
                "id": c.id,
                "folder_name": c.folder_name,
                "display_name": c.display_name,
                "slug": c.slug,
            }
            for c in config.class_configs
        ],
        "folder_to_id": generate_class_mapping(config),
        "id_to_display": {c.id: c.display_name for c in config.class_configs},
        "id_to_slug": {c.id: c.slug for c in config.class_configs},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_metadata(
    config: AppConfig,
    label_result: LabelValidationResult,
    corruption_result: CorruptionCheckResult,
    duplicate_result: DuplicateDetectionResult,
    statistics: DatasetStatistics,
    split_result: SplitResult,
) -> DatasetMetadata:
    """Assemble full dataset metadata from pipeline stage results."""
    return DatasetMetadata(
        crop=config.crop_name,
        dataset_source=str(config.get("crop.dataset_source", "")),
        created_at=datetime.now(timezone.utc).isoformat(),
        total_images=statistics.total_images,
        num_classes=config.num_classes,
        classes=[
            {
                "id": c.id,
                "folder_name": c.folder_name,
                "display_name": c.display_name,
                "slug": c.slug,
            }
            for c in config.class_configs
        ],
        class_mapping=generate_class_mapping(config),
        split_counts={
            "train": split_result.train_count,
            "val": split_result.val_count,
            "test": split_result.test_count,
        },
        class_distribution=split_result.class_distribution(),
        statistics=statistics.to_dict(),
        validation={
            "is_valid": label_result.is_valid,
            "found_folders": label_result.found_folders,
            "missing_folders": label_result.missing_folders,
            "unexpected_folders": label_result.unexpected_folders,
            "empty_folders": label_result.empty_folders,
            "alias_mappings": label_result.alias_mappings,
            "errors": label_result.errors,
            "warnings": label_result.warnings,
        },
        corruption={
            "total_checked": corruption_result.total_checked,
            "valid_count": corruption_result.valid_count,
            "corrupted_count": corruption_result.corrupted_count,
            "corrupted_files": [
                {"path": str(v.path), "error": v.error}
                for v in corruption_result.corrupted
            ],
        },
        duplicates={
            "exact_duplicate_groups": len(duplicate_result.exact_duplicates),
            "exact_duplicate_count": duplicate_result.exact_duplicate_count,
            "near_duplicate_groups": len(duplicate_result.near_duplicates),
            "near_duplicate_count": duplicate_result.near_duplicate_count,
            "unique_images": len(duplicate_result.unique_paths),
        },
    )


def save_metadata(metadata: DatasetMetadata, output_path: Path) -> None:
    """Persist metadata JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)
