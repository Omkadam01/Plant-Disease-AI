#!/usr/bin/env python3
"""Create stratified train/val/test split for tomato dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from importlib.util import module_from_spec, spec_from_file_location

_split_spec = spec_from_file_location(
    "tomato_split_pipeline",
    PROJECT_ROOT / "preprocessing" / "tomato_split_pipeline.py",
)
_split_module = module_from_spec(_split_spec)
sys.modules["tomato_split_pipeline"] = _split_module
assert _split_spec.loader is not None
_split_spec.loader.exec_module(_split_module)
TomatoSplitPipeline = _split_module.TomatoSplitPipeline

from utils.config import load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified tomato train/val/test split")
    parser.add_argument("--crop", type=str, default="tomato")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    logger = setup_logging(config, log_name="split_tomato", log_subdir="preprocessing")

    logger.info("Tomato split — stratified 70/15/15, seed=42 (no augment/balance)")
    pipeline = TomatoSplitPipeline(config, PROJECT_ROOT)
    result = pipeline.run()

    print("\n=== Tomato Split Summary ===")
    print(f"Total images:  {result.total_images:,}")
    print(f"Train:         {result.train_count:,}")
    print(f"Validation:    {result.val_count:,}")
    print(f"Test:          {result.test_count:,}")
    print("Per-class counts:")
    for class_name, dist in sorted(result.per_class_distribution.items()):
        print(
            f"  {class_name}: train={dist.get('train', 0)}, "
            f"val={dist.get('val', 0)}, test={dist.get('test', 0)}"
        )
    print(f"Validation:    {'PASSED' if result.validation.get('passed') else 'FAILED'}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.report_md_path:
        print(f"Report: {result.report_md_path}")
    if result.report_json_path:
        print(f"Report: {result.report_json_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
