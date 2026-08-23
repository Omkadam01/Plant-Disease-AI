#!/usr/bin/env python3
"""Preprocess PlantVillage tomato dataset into Dataset/datasets/tomato/raw/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from importlib.util import module_from_spec, spec_from_file_location

_tomato_spec = spec_from_file_location(
    "tomato_raw_pipeline",
    PROJECT_ROOT / "preprocessing" / "tomato_raw_pipeline.py",
)
_tomato_module = module_from_spec(_tomato_spec)
sys.modules["tomato_raw_pipeline"] = _tomato_module
assert _tomato_spec.loader is not None
_tomato_spec.loader.exec_module(_tomato_module)
TomatoRawPreprocessor = _tomato_module.TomatoRawPreprocessor
from utils.config import load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy and clean PlantVillage tomato dataset (raw stage only)",
    )
    parser.add_argument("--crop", type=str, default="tomato", help="Crop identifier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    logger = setup_logging(config, log_name="preprocess_tomato", log_subdir="preprocessing")

    logger.info("Tomato raw preprocessing — copy & clean only (no split/balance/augment)")
    preprocessor = TomatoRawPreprocessor(config, PROJECT_ROOT)
    result = preprocessor.run()

    print("\n=== Tomato Preprocessing Summary ===")
    print(f"Images copied:            {result.images_copied:,}")
    print(f"Exact duplicates removed: {result.exact_duplicates_removed}")
    print(f"Near duplicates removed:  {result.near_duplicates_removed}")
    print(f"Files skipped:            {result.files_skipped}")
    print(f"Final image count:        {result.final_image_count:,}")
    print("Images per class:")
    for name, count in sorted(result.images_per_class.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count:,}")
    print(f"Validation passed:        {result.validation.get('passed', False)}")
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
