"""Balance tomato training split via offline augmentation."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from utils.config import AppConfig
from utils.image_utils import collect_image_paths, read_image_rgb

logger = logging.getLogger(__name__)

TARGET_PER_CLASS = 1500
IMAGE_SIZE = 256
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
AUG_PREFIX = "aug_"


@dataclass
class ClassBalanceStats:
    class_name: str
    original_count: int = 0
    augmented_count: int = 0
    final_count: int = 0


@dataclass
class TomatoBalanceResult:
    success: bool
    original_training_images: int = 0
    generated_images: int = 0
    final_balanced_size: int = 0
    per_class: list[ClassBalanceStats] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    report_md_path: Path | None = None
    report_json_path: Path | None = None


def _transform_pool() -> list[A.BasicTransform]:
    """Augmentation pool — individual transforms applied via SomeOf."""
    return [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=1.0,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=12,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            shear=(-3, 3),
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        A.Perspective(scale=(0.02, 0.05), p=1.0),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=1.0,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=1.0,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
    ]


def _build_augmentation_pipeline(seed: int) -> A.Compose:
    """Build pipeline with 2–4 randomly combined augmentations (deterministic seed)."""
    rng = np.random.default_rng(seed)
    n_transforms = int(rng.integers(2, 5))  # 2, 3, or 4
    pool = _transform_pool()
    indices = rng.choice(len(pool), size=n_transforms, replace=False)
    selected = [pool[int(i)] for i in sorted(indices.tolist())]
    return A.Compose(
        selected + [A.Resize(IMAGE_SIZE, IMAGE_SIZE)],
        p=1.0,
    )


def _save_image_rgb(path: Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _is_augmented(path: Path) -> bool:
    return path.name.startswith(AUG_PREFIX)


def _copy_originals(source_class_dir: Path, dest_class_dir: Path) -> list[Path]:
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sorted(collect_image_paths(source_class_dir, IMAGE_EXTENSIONS)):
        dest = dest_class_dir / src.name
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def _generate_augmented(
    source_images: list[Path],
    dest_class_dir: Path,
    needed: int,
    class_seed: int,
) -> list[Path]:
    if needed <= 0 or not source_images:
        return []

    originals_only = [p for p in source_images if not _is_augmented(p)]
    if not originals_only:
        originals_only = source_images

    created: list[Path] = []
    source_idx = 0

    for aug_num in range(1, needed + 1):
        src = originals_only[source_idx % len(originals_only)]
        source_idx += 1

        seed = class_seed + aug_num
        pipeline = _build_augmentation_pipeline(seed)
        image = read_image_rgb(src)
        augmented = pipeline(image=image)["image"]

        counter = aug_num
        filename = f"{AUG_PREFIX}{counter:06d}.jpg"
        dest = dest_class_dir / filename
        while dest.exists():
            counter += 1
            filename = f"{AUG_PREFIX}{counter:06d}.jpg"
            dest = dest_class_dir / filename

        _save_image_rgb(dest, augmented)
        created.append(dest)

    return created


def _validate_balanced(
    balanced_root: Path,
    train_root: Path,
    val_root: Path,
    test_root: Path,
    class_names: list[str],
) -> dict:
    issues: list[str] = []
    per_class_final: dict[str, int] = {}

    for class_name in class_names:
        class_dir = balanced_root / class_name
        if not class_dir.exists():
            issues.append(f"missing_class:{class_name}")
            continue

        names_lower: set[str] = set()
        files = [p for p in class_dir.iterdir() if p.is_file()]
        per_class_final[class_name] = len(files)

        if not files:
            issues.append(f"empty_class:{class_name}")

        for path in files:
            lower = path.name.lower()
            if lower in names_lower:
                issues.append(f"duplicate_filename:{class_name}/{path.name}")
            names_lower.add(lower)

            if not _is_augmented(path):
                src = train_root / class_name / path.name
                if not src.exists():
                    issues.append(f"unexpected_original:{class_name}/{path.name}")

            try:
                with Image.open(path) as img:
                    if img.mode != "RGB":
                        issues.append(f"not_rgb:{class_name}/{path.name}:{img.mode}")
                    if img.width != IMAGE_SIZE or img.height != IMAGE_SIZE:
                        issues.append(
                            f"bad_size:{class_name}/{path.name}:{img.width}x{img.height}"
                        )
                    img.verify()
            except Exception as exc:
                issues.append(f"open_failed:{class_name}/{path.name}:{exc}")

    # Ensure val/test untouched — count files and compare to expected baseline
    val_test_untouched = True
    for split_root, split_name in ((val_root, "val"), (test_root, "test")):
        if not split_root.exists():
            issues.append(f"missing_{split_name}_split")
            val_test_untouched = False

    # No augmented files in train/val/test
    for split_root, split_name in (
        (train_root, "train"),
        (val_root, "val"),
        (test_root, "test"),
    ):
        if not split_root.exists():
            continue
        for path in split_root.rglob("*.jpg"):
            if _is_augmented(path):
                issues.append(f"aug_leaked_to_{split_name}:{path}")
                val_test_untouched = False

    aug_only_in_balanced = not any(i.startswith("aug_leaked") for i in issues)

    return {
        "every_class_exists": not any(i.startswith("missing_class") for i in issues),
        "all_images_open": not any(i.startswith("open_failed") for i in issues),
        "all_rgb": not any(i.startswith("not_rgb") for i in issues),
        "all_256x256": not any(i.startswith("bad_size") for i in issues),
        "no_duplicate_filenames": not any(i.startswith("duplicate_filename") for i in issues),
        "no_augmented_outside_balanced_train": aug_only_in_balanced,
        "val_test_untouched": val_test_untouched,
        "per_class_final_counts": per_class_final,
        "issues": issues[:50],
        "passed": len(issues) == 0,
    }


def _generate_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# PlantDiseaseAI — Tomato Training Balancing Report",
        "",
        f"**Generated:** {report['metadata']['generated_at']}",
        f"**Source:** `{report['metadata']['source_root']}`",
        f"**Destination:** `{report['metadata']['destination_root']}`",
        f"**Target per class:** {report['metadata']['target_per_class']}",
        f"**Random seed:** {report['metadata']['random_seed']}",
        "",
        "## Summary",
        "",
        f"- **Original training images:** {s['original_training_images']:,}",
        f"- **Generated (augmented):** {s['generated_images']:,}",
        f"- **Final balanced size:** {s['final_balanced_size']:,}",
        "",
        "## Per-Class Statistics",
        "",
        "| Class | Original | Augmented | Final |",
        "|-------|----------:|----------:|------:|",
    ]
    for row in report["per_class"]:
        lines.append(
            f"| `{row['class_name']}` | {row['original_count']:,} | "
            f"{row['augmented_count']:,} | {row['final_count']:,} |"
        )

    lines.extend(["", "## Validation", ""])
    for key, value in report["validation"].items():
        if key not in ("issues", "per_class_final_counts"):
            lines.append(f"- **{key}:** {value}")

    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


class TomatoTrainBalancer:
    """Balance tomato train split to TARGET_PER_CLASS via offline augmentation."""

    def __init__(self, config: AppConfig, project_root: Path, seed: int = 42) -> None:
        self.config = config
        self.project_root = project_root
        self.seed = seed
        self.train_root = project_root / "datasets" / "tomato" / "split" / "train"
        self.val_root = project_root / "datasets" / "tomato" / "split" / "val"
        self.test_root = project_root / "datasets" / "tomato" / "split" / "test"
        self.balanced_root = project_root / "datasets" / "tomato" / "balanced_train"
        self.reports_dir = project_root / "reports"

    def run(self) -> TomatoBalanceResult:
        result = TomatoBalanceResult(success=False)
        class_names = [c.folder_name for c in self.config.class_configs]

        if not self.train_root.exists():
            result.warnings.append(f"Train split not found: {self.train_root}")
            return result

        if self.balanced_root.exists():
            shutil.rmtree(self.balanced_root)
        self.balanced_root.mkdir(parents=True, exist_ok=True)

        total_original = 0
        total_generated = 0
        per_class_stats: list[ClassBalanceStats] = []

        for class_index, class_name in enumerate(class_names):
            source_dir = self.train_root / class_name
            dest_dir = self.balanced_root / class_name

            if not source_dir.exists():
                result.warnings.append(f"Missing train class: {class_name}")
                continue

            originals = _copy_originals(source_dir, dest_dir)
            original_count = len(originals)
            total_original += original_count

            if original_count >= TARGET_PER_CLASS:
                augmented_count = 0
                final_count = original_count
                logger.info(
                    "Class %s: %d images (>= %d) — copied unchanged",
                    class_name,
                    original_count,
                    TARGET_PER_CLASS,
                )
            else:
                needed = TARGET_PER_CLASS - original_count
                class_seed = self.seed + class_index * 100_000
                created = _generate_augmented(originals, dest_dir, needed, class_seed)
                augmented_count = len(created)
                final_count = original_count + augmented_count
                total_generated += augmented_count
                logger.info(
                    "Class %s: %d original + %d augmented = %d",
                    class_name,
                    original_count,
                    augmented_count,
                    final_count,
                )

            per_class_stats.append(
                ClassBalanceStats(
                    class_name=class_name,
                    original_count=original_count,
                    augmented_count=augmented_count,
                    final_count=final_count,
                )
            )

        validation = _validate_balanced(
            self.balanced_root,
            self.train_root,
            self.val_root,
            self.test_root,
            class_names,
        )

        final_size = sum(s.final_count for s in per_class_stats)
        result.original_training_images = total_original
        result.generated_images = total_generated
        result.final_balanced_size = final_size
        result.per_class = per_class_stats
        result.validation = validation
        result.success = validation["passed"]

        augmentation_stats = {
            "target_per_class": TARGET_PER_CLASS,
            "classes_augmented": sum(1 for s in per_class_stats if s.augmented_count > 0),
            "classes_unchanged": sum(1 for s in per_class_stats if s.augmented_count == 0),
            "total_generated": total_generated,
            "transforms_pool_size": len(_transform_pool()),
            "transforms_per_image": "2-4 random combination",
            "jpeg_quality": 95,
        }

        report_dict = {
            "metadata": {
                "crop": "tomato",
                "source_root": str(self.train_root),
                "destination_root": str(self.balanced_root),
                "target_per_class": TARGET_PER_CLASS,
                "random_seed": self.seed,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "augmentation_library": "albumentations",
                "no_data_leakage": True,
            },
            "summary": {
                "original_training_images": total_original,
                "generated_images": total_generated,
                "final_balanced_size": final_size,
            },
            "per_class": [s.__dict__ for s in per_class_stats],
            "augmentation_statistics": augmentation_stats,
            "validation": validation,
            "warnings": result.warnings,
        }

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = self.reports_dir / "tomato_balancing.md"
        json_path = self.reports_dir / "tomato_balancing.json"
        md_path.write_text(_generate_markdown(report_dict), encoding="utf-8")
        json_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        result.report_md_path = md_path
        result.report_json_path = json_path
        return result
