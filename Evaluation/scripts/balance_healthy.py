#!/usr/bin/env python3
"""
Offline augmentation to balance the Healthy class.

Generates realistic augmented Healthy images until the target count
is reached. Does not modify preprocessing pipeline or model code.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from preprocessing.metadata import DatasetMetadata, save_metadata
from preprocessing.report import save_audit_report
from preprocessing.statistics import compute_statistics
from utils.config import load_config
from utils.image_utils import read_image_rgb
from utils.logging import setup_logging
from utils.paths import ProjectPaths

HEALTHY_FOLDER = "Healthy"
TARGET_COUNT = 1000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_healthy_augmentation_pipeline(image_size: int = 256) -> A.Compose:
    """Realistic augmentations for grape leaf Healthy class balancing."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.5,
        ),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.6,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.15),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.Resize(image_size, image_size),
    ])


def collect_images(folder: Path) -> list[Path]:
    """Collect existing image paths in folder."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_image_rgb(path: Path, image: np.ndarray) -> None:
    """Save RGB image as JPEG."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def generate_augmented_images(
    source_images: list[Path],
    output_dir: Path,
    target_count: int,
    pipeline: A.Compose,
    seed: int = 42,
) -> list[Path]:
    """
    Generate augmented images until target_count is reached.

    Returns list of newly created file paths.
    """
    existing = collect_images(output_dir)
    if len(existing) >= target_count:
        return []

    needed = target_count - len(existing)
    rng = np.random.default_rng(seed)
    created: list[Path] = []
    source_idx = 0

    while len(created) < needed:
        src = source_images[source_idx % len(source_images)]
        source_idx += 1

        image = read_image_rgb(src)
        augmented = pipeline(image=image)["image"]

        filename = f"healthy_aug_{uuid.uuid4().hex[:12]}.jpg"
        dest = output_dir / filename

        while dest.exists():
            filename = f"healthy_aug_{uuid.uuid4().hex[:12]}.jpg"
            dest = output_dir / filename

        save_image_rgb(dest, augmented)
        created.append(dest)

    return created


def count_class_images(base: Path, class_folders: list[str]) -> dict[str, int]:
    """Count images per canonical class folder."""
    counts: dict[str, int] = {}
    for name in class_folders:
        folder = base / name
        counts[name] = len(collect_images(folder)) if folder.exists() else 0
    return counts


def load_existing_metadata(metadata_path: Path) -> dict:
    """Load existing dataset metadata JSON."""
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_updated_metadata(
    config,
    paths: ProjectPaths,
    prior: dict,
    augmented_count: int,
) -> DatasetMetadata:
    """Rebuild metadata reflecting post-balance dataset state."""
    class_folders = [c.folder_name for c in config.class_configs]
    processed_labeled = {
        name: collect_images(paths.processed / name) for name in class_folders
    }
    statistics = compute_statistics(
        {k: [Path(p) for p in v] for k, v in processed_labeled.items()}
    )

    train_counts = count_class_images(paths.train, class_folders)
    val_counts = count_class_images(paths.val, class_folders)
    test_counts = count_class_images(paths.test, class_folders)

    class_distribution = {
        name: {
            "train": train_counts[name],
            "val": val_counts[name],
            "test": test_counts[name],
        }
        for name in class_folders
    }

    total_processed = sum(len(v) for v in processed_labeled.values())

    balance_info = prior.get("balance", {})
    balance_info.update({
        "healthy_target": TARGET_COUNT,
        "healthy_augmented_added": augmented_count,
        "healthy_augmentation_method": "offline_realistic",
        "balanced_at": datetime.now(timezone.utc).isoformat(),
    })

    return DatasetMetadata(
        crop=config.crop_name,
        dataset_source=str(config.get("crop.dataset_source", "")),
        created_at=datetime.now(timezone.utc).isoformat(),
        total_images=total_processed,
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
        class_mapping={c.folder_name: c.id for c in config.class_configs},
        split_counts={
            "train": sum(train_counts.values()),
            "val": sum(val_counts.values()),
            "test": sum(test_counts.values()),
        },
        class_distribution=class_distribution,
        statistics=statistics.to_dict(),
        validation=prior.get("validation", {}),
        corruption=prior.get("corruption", {}),
        duplicates=prior.get("duplicates", {}),
    )


def distribute_to_splits(
    new_images: list[Path],
    paths: ProjectPaths,
    seed: int = 42,
) -> dict[str, int]:
    """
    Distribute newly augmented Healthy images into train/val/test (70/15/15).

    Enables balanced training without re-running the preprocessing pipeline.
    """
    if not new_images:
        return {"train": 0, "val": 0, "test": 0}

    rng = np.random.default_rng(seed)
    shuffled = list(new_images)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    import shutil

    counts = {"train": 0, "val": 0, "test": 0}
    for split_name, images in splits.items():
        dest_dir = getattr(paths, split_name) / HEALTHY_FOLDER
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in images:
            dest = dest_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
                counts[split_name] += 1

    return counts


def print_distribution(paths: ProjectPaths, class_configs) -> None:
    """Print final class distribution across processed and splits."""
    print()
    print("=" * 65)
    print("FINAL CLASS DISTRIBUTION")
    print("=" * 65)
    print(f"{'Class':<40} {'Processed':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 65)

    totals = {"processed": 0, "train": 0, "val": 0, "test": 0}
    for c in class_configs:
        counts = {
            split: count_class_images(getattr(paths, split), [c.folder_name])[c.folder_name]
            for split in ("processed", "train", "val", "test")
        }
        print(
            f"{c.display_name:<40} {counts['processed']:>8} "
            f"{counts['train']:>8} {counts['val']:>8} {counts['test']:>8}"
        )
        for k, v in counts.items():
            totals[k] += v

    print("-" * 65)
    print(
        f"{'TOTAL':<40} {totals['processed']:>8} "
        f"{totals['train']:>8} {totals['val']:>8} {totals['test']:>8}"
    )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balance Healthy class via offline augmentation")
    parser.add_argument("--crop", default="grape")
    parser.add_argument("--target", type=int, default=TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    logger = setup_logging(config, log_name="balance", log_subdir="balance")
    paths = ProjectPaths.from_config(config)

    healthy_dir = paths.processed / HEALTHY_FOLDER
    if not healthy_dir.exists():
        logger.error("Healthy processed directory not found: %s", healthy_dir)
        return 1

    source_images = collect_images(healthy_dir)
    if not source_images:
        logger.error("No source Healthy images found in %s", healthy_dir)
        return 1

    current_count = len(source_images)
    target = args.target
    logger.info("Healthy class: %d images (target: %d)", current_count, target)

    if current_count >= target:
        logger.info("Healthy class already meets target. No augmentation needed.")
        print_distribution(paths, config.class_configs)
        return 0

    image_size = int(config.get("training.image_size", 224))
    # PlantVillage images are 256x256; use native size for augmentation
    if source_images:
        sample = read_image_rgb(source_images[0])
        image_size = sample.shape[0]

    pipeline = build_healthy_augmentation_pipeline(image_size=image_size)
    new_images = generate_augmented_images(
        source_images=source_images,
        output_dir=healthy_dir,
        target_count=target,
        pipeline=pipeline,
        seed=args.seed,
    )

    logger.info("Generated %d augmented Healthy images", len(new_images))

    split_distribution = distribute_to_splits(new_images, paths, seed=args.seed)
    logger.info(
        "Distributed to splits — train: %d, val: %d, test: %d",
        split_distribution["train"],
        split_distribution["val"],
        split_distribution["test"],
    )

    metadata_path = paths.reports / "dataset_metadata.json"
    prior = load_existing_metadata(metadata_path) if metadata_path.exists() else {}

    metadata = build_updated_metadata(config, paths, prior, len(new_images))
    metadata_dict = metadata.to_dict()
    metadata_dict["balance"] = {
        "healthy_target": target,
        "healthy_before": current_count,
        "healthy_after": current_count + len(new_images),
        "healthy_augmented_added": len(new_images),
        "augmentations": [
            "HorizontalFlip",
            "VerticalFlip",
            "Rotation (+/-20)",
            "RandomBrightnessContrast",
            "HueSaturationValue",
            "RandomResizedCrop + Resize",
            "GaussianBlur (low probability)",
            "Affine",
        ],
        "split_distribution_of_new": split_distribution,
        "balanced_at": datetime.now(timezone.utc).isoformat(),
    }

    save_metadata(metadata, metadata_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)

    report_paths = save_audit_report(metadata, paths.reports)
    logger.info("Updated metadata: %s", metadata_path)
    logger.info("Updated audit report: %s", report_paths[0])

    print_distribution(paths, config.class_configs)

    final_healthy = count_class_images(paths.processed, [HEALTHY_FOLDER])[HEALTHY_FOLDER]
    print(f"Healthy class balanced: {current_count} -> {final_healthy} images")
    print(f"Reports updated at: {paths.reports}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
