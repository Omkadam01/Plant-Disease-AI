"""End-to-end preprocessing pipeline orchestrator."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from preprocessing.duplicate_detector import detect_duplicates
from preprocessing.image_checker import detect_corrupted_images
from preprocessing.metadata import (
    build_metadata,
    save_class_mapping,
    save_metadata,
)
from preprocessing.report import save_audit_report
from preprocessing.splitter import materialize_split, save_split_manifest, stratified_split
from preprocessing.statistics import compute_statistics
from preprocessing.validators import validate_labels_and_folders
from utils.config import AppConfig
from utils.image_utils import collect_image_paths
from utils.paths import ProjectPaths

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Final preprocessing pipeline result."""

    success: bool
    metadata_path: Path | None = None
    report_paths: tuple[Path, Path] | None = None
    total_unique_images: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PreprocessingPipeline:
    """
    Production preprocessing pipeline for grape leaf disease datasets.

    Stages:
    1. Label & folder validation
    2. Corrupted image detection
    3. Exact & near duplicate removal
    4. Image statistics
    5. Stratified train/val/test split
    6. Metadata & audit report generation
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.paths = ProjectPaths.from_config(config)
        self.aliases: dict[str, str] = dict(config.get("class_aliases", {}))

    def _resolve_class_name(self, folder_name: str) -> str:
        return self.aliases.get(folder_name, folder_name)

    def _collect_labeled_paths(self, base_dir: Path) -> dict[str, list[Path]]:
        """Collect images grouped by canonical class folder name."""
        extensions = tuple(self.config.get("preprocessing.image_extensions", [".jpg", ".jpeg", ".png"]))
        labeled: dict[str, list[Path]] = {c.folder_name: [] for c in self.config.class_configs}

        if not base_dir.exists():
            return labeled

        for class_dir in base_dir.iterdir():
            if not class_dir.is_dir():
                continue
            canonical = self._resolve_class_name(class_dir.name)
            if canonical not in labeled:
                continue
            images = collect_image_paths(class_dir, extensions)
            labeled[canonical].extend(images)

        return labeled

    def _copy_to_processed(
        self,
        labeled_paths: dict[str, list[Path]],
    ) -> dict[str, list[Path]]:
        """Copy deduplicated images to processed/ with canonical folder names."""
        processed_paths: dict[str, list[Path]] = {k: [] for k in labeled_paths}

        for class_name, paths in labeled_paths.items():
            dest_dir = self.paths.processed / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for src in paths:
                dest = dest_dir / src.name
                if dest.exists():
                    stem, suffix = dest.stem, dest.suffix
                    counter = 1
                    while dest.exists():
                        dest = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.copy2(src, dest)
                processed_paths[class_name].append(dest)

        return processed_paths

    def run(self) -> PreprocessingResult:
        """Execute the full preprocessing pipeline."""
        result = PreprocessingResult(success=False)
        self.paths.ensure_dirs()

        logger.info("Starting preprocessing pipeline for crop: %s", self.config.crop_name)

        # Stage 1: Label validation
        logger.info("Stage 1: Validating labels and folder structure")
        label_result = validate_labels_and_folders(self.paths.raw, self.config)
        result.warnings.extend(label_result.warnings)

        if not label_result.is_valid:
            result.errors.extend(label_result.errors)
            logger.error("Label validation failed: %s", label_result.errors)
            return result

        # Collect all raw images
        raw_labeled = self._collect_labeled_paths(self.paths.raw)
        all_paths: list[Path] = []
        for paths in raw_labeled.values():
            all_paths.extend(paths)

        if not all_paths:
            result.errors.append("No images found in raw dataset directory")
            return result

        logger.info("Found %d images across %d classes", len(all_paths), len(raw_labeled))

        # Stage 2: Corruption detection
        logger.info("Stage 2: Detecting corrupted images")
        corruption_result = detect_corrupted_images(all_paths, self.config)
        if corruption_result.corrupted_count > 0:
            result.warnings.append(
                f"Removed {corruption_result.corrupted_count} corrupted images"
            )

        valid_paths = corruption_result.valid_images

        # Rebuild labeled dict with valid paths only
        valid_set = set(valid_paths)
        valid_labeled: dict[str, list[Path]] = {}
        for class_name, paths in raw_labeled.items():
            valid_labeled[class_name] = [p for p in paths if p in valid_set]

        # Stage 3: Duplicate detection
        logger.info("Stage 3: Detecting duplicates")
        flat_valid = [p for paths in valid_labeled.values() for p in paths]
        duplicate_result = detect_duplicates(flat_valid, self.config)

        unique_set = set(duplicate_result.unique_paths)
        dedup_labeled: dict[str, list[Path]] = {}
        for class_name, paths in valid_labeled.items():
            dedup_labeled[class_name] = [p for p in paths if p in unique_set]

        if duplicate_result.exact_duplicate_count or duplicate_result.near_duplicate_count:
            result.warnings.append(
                f"Removed {duplicate_result.exact_duplicate_count} exact and "
                f"{duplicate_result.near_duplicate_count} near duplicates"
            )

        # Copy to processed/
        logger.info("Copying deduplicated images to processed/")
        processed_labeled = self._copy_to_processed(dedup_labeled)
        result.total_unique_images = sum(len(v) for v in processed_labeled.values())

        # Stage 4: Statistics
        logger.info("Stage 4: Computing image statistics")
        statistics = compute_statistics(processed_labeled)

        # Stage 5: Stratified split
        logger.info("Stage 5: Performing stratified split")
        split_result = stratified_split(processed_labeled, self.config)

        copy_files = bool(self.config.get("preprocessing.copy_files", True))
        materialize_split(
            split_result,
            output_dirs={
                "train": self.paths.train,
                "val": self.paths.val,
                "test": self.paths.test,
            },
            copy_files=copy_files,
        )

        save_split_manifest(split_result, self.paths.reports / "split_manifest.json")

        # Stage 6: Metadata & reports
        logger.info("Stage 6: Generating metadata and audit report")
        metadata = build_metadata(
            config=self.config,
            label_result=label_result,
            corruption_result=corruption_result,
            duplicate_result=duplicate_result,
            statistics=statistics,
            split_result=split_result,
        )

        metadata_path = self.paths.reports / "dataset_metadata.json"
        save_metadata(metadata, metadata_path)
        save_class_mapping(self.config, self.paths.reports / "class_mapping.json")

        report_paths = save_audit_report(metadata, self.paths.reports)

        result.success = True
        result.metadata_path = metadata_path
        result.report_paths = report_paths

        logger.info(
            "Preprocessing complete: %d unique images | train=%d val=%d test=%d",
            result.total_unique_images,
            split_result.train_count,
            split_result.val_count,
            split_result.test_count,
        )

        return result
