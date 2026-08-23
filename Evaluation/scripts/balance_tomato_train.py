#!/usr/bin/env python3
"""Balance tomato training split to 1500 images per class."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from importlib.util import module_from_spec, spec_from_file_location

_balance_spec = spec_from_file_location(
    "tomato_balance_pipeline",
    PROJECT_ROOT / "preprocessing" / "tomato_balance_pipeline.py",
)
_balance_module = module_from_spec(_balance_spec)
sys.modules["tomato_balance_pipeline"] = _balance_module
assert _balance_spec.loader is not None
_balance_spec.loader.exec_module(_balance_module)
TomatoTrainBalancer = _balance_module.TomatoTrainBalancer

from utils.config import load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balance tomato train split via offline augmentation")
    parser.add_argument("--crop", type=str, default="tomato")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    logger = setup_logging(config, log_name="balance_tomato", log_subdir="preprocessing")

    logger.info("Tomato train balancing — target 1500/class, seed=%d", args.seed)
    balancer = TomatoTrainBalancer(config, PROJECT_ROOT, seed=args.seed)
    result = balancer.run()

    print("\n=== Tomato Training Balancing Summary ===")
    print(f"Original training images: {result.original_training_images:,}")
    print(f"Generated images:         {result.generated_images:,}")
    print(f"Final balanced size:      {result.final_balanced_size:,}")
    print("Images per class:")
    for stats in result.per_class:
        print(
            f"  {stats.class_name}: original={stats.original_count}, "
            f"aug={stats.augmented_count}, final={stats.final_count}"
        )
    print(f"Validation:               {'PASSED' if result.validation.get('passed') else 'FAILED'}")
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
