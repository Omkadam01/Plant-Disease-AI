#!/usr/bin/env python3
"""Evaluate trained model on test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from evaluation.evaluator import Evaluator
from utils.config import load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model on test dataset",
    )
    parser.add_argument("--crop", type=str, default="grape", help="Crop identifier")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: from config)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    logger = setup_logging(config, log_name="evaluate", log_subdir="evaluation")

    weights_path = Path(args.weights) if args.weights else None
    if weights_path and not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    logger.info("PlantDiseaseAI Evaluation")
    evaluator = Evaluator(config, weights_path=weights_path)
    result = evaluator.evaluate()

    if result.success and result.metrics:
        logger.info("Evaluation completed successfully")
        logger.info("Accuracy: %.4f", result.metrics.accuracy)
        logger.info("F1 (macro): %.4f", result.metrics.f1_macro)
        logger.info("Metrics saved: %s", result.metrics_path)
        return 0

    for error in result.errors:
        logger.error("Error: %s", error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
