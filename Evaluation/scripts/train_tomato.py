#!/usr/bin/env python3
"""Train tomato EfficientNet-B0 disease classification model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from training.tomato_trainer import TomatoTrainer
from utils.config import load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tomato EfficientNet-B0 model")
    parser.add_argument("--crop", type=str, default="tomato")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    if args.epochs:
        config.raw.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size:
        config.raw["training"]["batch_size"] = args.batch_size

    logger = setup_logging(config, log_name="train_tomato", log_subdir="training")
    logger.info("Tomato EfficientNet-B0 Training (test set excluded)")

    trainer = TomatoTrainer(config)
    result = trainer.train()

    if not result.success:
        for error in result.errors:
            logger.error(error)
            print(f"ERROR: {error}")
        return 1

    hours = result.total_training_time_sec / 3600
    print("\n=== Tomato Training Complete ===")
    print(f"Best epoch:            {result.best_epoch + 1}")
    print(f"Best validation acc:   {result.best_val_accuracy:.4f}")
    print(f"Best validation loss:  {result.best_val_loss:.4f}")
    print(f"Total training time:   {result.total_training_time_sec:.1f}s ({hours:.2f}h)")
    if result.best_model_path:
        print(f"Best model:            {result.best_model_path}")
    print(f"Last model:            Models/weights/tomato/last_model.pth")
    print(f"History:               logs/tomato_training_history.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
