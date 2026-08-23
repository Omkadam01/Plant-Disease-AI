#!/usr/bin/env python3
"""Final tomato model evaluation on the held-out test split only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sanitize_import_path() -> None:
    """Remove bundled .vendor packages so system numpy/sklearn/matplotlib are used."""
    vendor_root = (PROJECT_ROOT / ".vendor").resolve()
    sys.path = [
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() != vendor_root
    ]


_sanitize_import_path()

for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from evaluation.tomato_evaluator import TomatoEvaluator
from utils.config import load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tomato best_model.pth on Dataset/datasets/tomato/split/test only",
    )
    parser.add_argument("--crop", type=str, default="tomato")
    parser.add_argument(
        "--weights",
        type=str,
        default="Models/weights/tomato/best_model.pth",
        help="Path to tomato checkpoint (default: Models/weights/tomato/best_model.pth)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    logger = setup_logging(config, log_name="evaluate_tomato", log_subdir="evaluation")

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    logger.info("Tomato Final Test Evaluation (test split only)")
    logger.info("Weights: %s", weights_path)
    logger.info("Test dir: %s", config.path("paths.test"))

    evaluator = TomatoEvaluator(config, weights_path=weights_path)
    result = evaluator.evaluate()

    if result.metrics:
        print("\n=== Tomato Final Test Evaluation ===")
        print(f"Test Accuracy:        {result.metrics.accuracy:.4f}")
        print(f"Test Loss:            {result.test_loss:.6f}")
        print(f"Macro F1:             {result.metrics.f1_macro:.4f}")
        print(f"Weighted F1:          {result.metrics.f1_weighted:.4f}")
        print(f"Top-3 Accuracy:       {result.top3_accuracy:.4f}")
        print(f"Misclassified Images: {result.num_misclassified}")
        print(f"Evaluation Time:      {result.evaluation_time_sec:.2f}s")
        print("====================================\n")

    if result.success:
        logger.info("Evaluation completed successfully")
        logger.info("Summary JSON: %s", result.summary_json_path)
        logger.info("Summary MD: %s", result.summary_md_path)
        return 0

    for error in result.errors:
        logger.error("Error: %s", error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
