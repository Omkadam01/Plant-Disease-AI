#!/usr/bin/env python3
"""Run dataset preprocessing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from preprocessing.pipeline import PreprocessingPipeline
from utils.config import load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess PlantVillage grape dataset for training",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default="grape",
        help="Crop identifier (default: grape)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to base config YAML",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config = Path(args.config) if args.config else None

    config = load_config(crop=args.crop, base_config_path=base_config, project_root=PROJECT_ROOT)
    logger = setup_logging(config, log_name="preprocess", log_subdir="preprocessing")

    logger.info("PlantDiseaseAI Preprocessing Pipeline")
    logger.info("Crop: %s", config.crop_name)
    logger.info("Raw data path: %s", config.path("paths.raw"))

    pipeline = PreprocessingPipeline(config)
    result = pipeline.run()

    if result.success:
        logger.info("Preprocessing completed successfully")
        logger.info("Unique images: %d", result.total_unique_images)
        if result.metadata_path:
            logger.info("Metadata: %s", result.metadata_path)
        if result.report_paths:
            logger.info("Audit report: %s", result.report_paths[0])
        return 0

    for error in result.errors:
        logger.error("Error: %s", error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
