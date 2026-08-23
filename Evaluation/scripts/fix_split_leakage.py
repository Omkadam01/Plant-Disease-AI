#!/usr/bin/env python3
"""
Fix data leakage: remove augmented Healthy images from val/test,
ensure all augmentations exist only in train, update reports.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from preprocessing.metadata import DatasetMetadata, save_metadata
from preprocessing.report import save_audit_report
from preprocessing.statistics import compute_statistics
from utils.config import load_config
from utils.paths import ProjectPaths

AUGMENTED_PREFIX = "healthy_aug_"
HEALTHY_FOLDER = "Healthy"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def is_augmented(path: Path) -> bool:
    return path.name.startswith(AUGMENTED_PREFIX)


def count_split_images(base: Path, class_folders: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for name in class_folders:
            counts[split][name] = len(collect_images(base / split / name))
    return counts


def remove_augmented_from_split(split_dir: Path) -> list[Path]:
    """Remove augmented Healthy images from a split. Returns removed paths."""
    healthy_dir = split_dir / HEALTHY_FOLDER
    removed: list[Path] = []
    for path in collect_images(healthy_dir):
        if is_augmented(path):
            path.unlink()
            removed.append(path)
    return removed


def ensure_augmented_in_train(processed_healthy: Path, train_healthy: Path) -> list[Path]:
    """Copy any augmented images from processed to train if missing."""
    train_healthy.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in collect_images(train_healthy)}
    added: list[Path] = []

    for src in collect_images(processed_healthy):
        if not is_augmented(src):
            continue
        if src.name not in existing:
            dest = train_healthy / src.name
            shutil.copy2(src, dest)
            added.append(dest)
            existing.add(src.name)

    return added


def update_split_manifest(
    manifest_path: Path,
    augmented_train_files: list[Path],
    class_counts: dict[str, dict[str, int]],
) -> None:
    """Update split manifest with corrected counts and augmented train entries."""
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove any augmented entries incorrectly assigned to val/test
    original_entries = [
        e for e in data["entries"]
        if not (e.get("filename", "").startswith(AUGMENTED_PREFIX) and e.get("split") in ("val", "test"))
    ]

    existing_aug_in_manifest = {
        e["filename"] for e in original_entries
        if e.get("filename", "").startswith(AUGMENTED_PREFIX) and e.get("split") == "train"
    }

    for path in augmented_train_files:
        if path.name in existing_aug_in_manifest:
            continue
        original_entries.append({
            "source_path": str(path),
            "class_name": HEALTHY_FOLDER,
            "class_id": 3,
            "split": "train",
            "filename": path.name,
            "augmented": True,
        })

    train_count = sum(class_counts["train"].values())
    val_count = sum(class_counts["val"].values())
    test_count = sum(class_counts["test"].values())

    data["train_count"] = train_count
    data["val_count"] = val_count
    data["test_count"] = test_count
    data["class_distribution"] = {
        name: {
            "train": class_counts["train"][name],
            "val": class_counts["val"][name],
            "test": class_counts["test"][name],
        }
        for name in class_counts["train"]
    }
    data["entries"] = original_entries
    data["split_corrected_at"] = datetime.now(timezone.utc).isoformat()
    data["data_leakage_fix"] = {
        "augmented_images_train_only": True,
        "augmented_removed_from_val_test": True,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_metadata(
    config,
    paths: ProjectPaths,
    prior: dict,
    class_counts: dict[str, dict[str, int]],
) -> DatasetMetadata:
    class_folders = [c.folder_name for c in config.class_configs]
    processed_labeled = {name: collect_images(paths.processed / name) for name in class_folders}
    statistics = compute_statistics({k: v for k, v in processed_labeled.items()})

    return DatasetMetadata(
        crop=config.crop_name,
        dataset_source=str(config.get("crop.dataset_source", "")),
        created_at=datetime.now(timezone.utc).isoformat(),
        total_images=sum(len(v) for v in processed_labeled.values()),
        num_classes=config.num_classes,
        classes=[
            {"id": c.id, "folder_name": c.folder_name, "display_name": c.display_name, "slug": c.slug}
            for c in config.class_configs
        ],
        class_mapping={c.folder_name: c.id for c in config.class_configs},
        split_counts={
            "train": sum(class_counts["train"].values()),
            "val": sum(class_counts["val"].values()),
            "test": sum(class_counts["test"].values()),
        },
        class_distribution={
            name: {
                "train": class_counts["train"][name],
                "val": class_counts["val"][name],
                "test": class_counts["test"][name],
            }
            for name in class_folders
        },
        statistics=statistics.to_dict(),
        validation=prior.get("validation", {}),
        corruption=prior.get("corruption", {}),
        duplicates=prior.get("duplicates", {}),
    )


def verify_no_leakage(paths: ProjectPaths) -> dict[str, bool | int]:
    val_aug = [p for p in collect_images(paths.val / HEALTHY_FOLDER) if is_augmented(p)]
    test_aug = [p for p in collect_images(paths.test / HEALTHY_FOLDER) if is_augmented(p)]
    train_aug = [p for p in collect_images(paths.train / HEALTHY_FOLDER) if is_augmented(p)]
    processed_aug = [p for p in collect_images(paths.processed / HEALTHY_FOLDER) if is_augmented(p)]

    return {
        "val_augmented_count": len(val_aug),
        "test_augmented_count": len(test_aug),
        "train_augmented_count": len(train_aug),
        "processed_augmented_count": len(processed_aug),
        "no_aug_in_val": len(val_aug) == 0,
        "no_aug_in_test": len(test_aug) == 0,
        "all_aug_in_train": len(train_aug) == len(processed_aug),
        "no_leakage": len(val_aug) == 0 and len(test_aug) == 0 and len(train_aug) == len(processed_aug),
    }


def main() -> int:
    config = load_config(crop="grape", project_root=PROJECT_ROOT)
    paths = ProjectPaths.from_config(config)
    class_folders = [c.folder_name for c in config.class_configs]

    print("Fixing dataset split — removing augmented images from val/test...")
    print()

    removed_val = remove_augmented_from_split(paths.val)
    removed_test = remove_augmented_from_split(paths.test)
    print(f"Removed {len(removed_val)} augmented images from val/Healthy/")
    print(f"Removed {len(removed_test)} augmented images from test/Healthy/")

    added_train = ensure_augmented_in_train(
        paths.processed / HEALTHY_FOLDER,
        paths.train / HEALTHY_FOLDER,
    )
    print(f"Added {len(added_train)} missing augmented images to train/Healthy/")

    class_counts = count_split_images(paths.dataset_root, class_folders)
    verification = verify_no_leakage(paths)

    # Load prior metadata
    metadata_path = paths.reports / "dataset_metadata.json"
    prior = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            prior = json.load(f)

    metadata = build_metadata(config, paths, prior, class_counts)
    metadata_dict = metadata.to_dict()
    metadata_dict["balance"] = prior.get("balance", {})
    metadata_dict["split_correction"] = {
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "augmented_images_train_only": True,
        "removed_from_val": len(removed_val),
        "removed_from_test": len(removed_test),
        "added_to_train": len(added_train),
        "verification": verification,
    }
    save_metadata(metadata, metadata_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)

    augmented_train = [p for p in collect_images(paths.train / HEALTHY_FOLDER) if is_augmented(p)]
    update_split_manifest(
        paths.reports / "split_manifest.json",
        augmented_train,
        class_counts,
    )
    save_audit_report(metadata, paths.reports)

    print()
    print("=" * 65)
    print("FINAL DATASET STATISTICS")
    print("=" * 65)
    print(f"{'Class':<40} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 65)
    for c in config.class_configs:
        print(
            f"{c.display_name:<40} "
            f"{class_counts['train'][c.folder_name]:>8} "
            f"{class_counts['val'][c.folder_name]:>8} "
            f"{class_counts['test'][c.folder_name]:>8}"
        )
    print("-" * 65)
    totals = {s: sum(class_counts[s].values()) for s in ("train", "val", "test")}
    print(f"{'TOTAL':<40} {totals['train']:>8} {totals['val']:>8} {totals['test']:>8}")
    print()

    train_healthy = class_counts["train"][HEALTHY_FOLDER]
    train_aug = verification["train_augmented_count"]
    train_orig = train_healthy - train_aug
    print(f"Healthy Train:  {train_healthy} ({train_orig} original + {train_aug} augmented)")
    print(f"Healthy Val:    {class_counts['val'][HEALTHY_FOLDER]} (original only)")
    print(f"Healthy Test:   {class_counts['test'][HEALTHY_FOLDER]} (original only)")
    print()
    print("Verification:")
    print(f"  No augmented in val:   {'PASS' if verification['no_aug_in_val'] else 'FAIL'}")
    print(f"  No augmented in test:  {'PASS' if verification['no_aug_in_test'] else 'FAIL'}")
    print(f"  All augmented in train: {'PASS' if verification['all_aug_in_train'] else 'FAIL'}")
    print(f"  No data leakage:        {'PASS' if verification['no_leakage'] else 'FAIL'}")
    print()

    if verification["no_leakage"]:
        print("Dataset split verified. No data leakage detected. Ready for EfficientNet-B0 training.")
        return 0

    print("ERROR: Data leakage verification failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
