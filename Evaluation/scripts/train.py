#!/usr/bin/env python3
"""Train plant disease classification model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from training.trainer import Trainer
from utils.config import AppConfig, load_config
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train grape leaf disease classification model",
    )
    parser.add_argument("--crop", type=str, default="grape", help="Crop identifier")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["efficientnet_b0", "mobilenet_v3_large"],
        help="Model architecture override",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for resuming",
    )
    return parser.parse_args()


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.model:
        config.raw.setdefault("training", {})["model_name"] = args.model
    if args.epochs:
        config.raw["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config.raw["training"]["batch_size"] = args.batch_size
    if args.lr:
        config.raw["training"]["learning_rate"] = args.lr
    if args.resume:
        config.raw["training"]["resume"] = True
    if args.resume_checkpoint:
        config.raw["training"]["resume_checkpoint"] = args.resume_checkpoint
    return config


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    config = _apply_overrides(config, args)

    logger = setup_logging(config, log_name="train", log_subdir="training")
    logger.info("PlantDiseaseAI Training")
    logger.info("Model: %s", config.get("training.model_name"))
    logger.info("Epochs: %s", config.get("training.num_epochs"))

    trainer = Trainer(config)
    result = trainer.train()

    if result.success:
        logger.info("Training completed successfully")
        logger.info("Best val loss: %.4f", result.best_val_loss)
        logger.info("Best val accuracy: %.4f", result.best_val_accuracy)
        if result.best_model_path:
            logger.info("Best model: %s", result.best_model_path)
        return 0

    for error in result.errors:
        logger.error("Error: %s", error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
