"""Tomato dataset train/val/test split — stratified, reproducible."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from utils.config import AppConfig
from utils.image_utils import collect_image_paths

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SplitAssignment:
    source_path: Path
    class_name: str
    class_id: int
    split: str
    filename: str


@dataclass
class TomatoSplitResult:
    success: bool
    total_images: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    per_class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    assignments: list[SplitAssignment] = field(default_factory=list)
    report_md_path: Path | None = None
    report_json_path: Path | None = None


def _train_test_split(
    items: list[Path],
    test_size: float,
    random_state: int,
) -> tuple[list[Path], list[Path]]:
    """Numpy-only split mirroring sklearn train_test_split (shuffle, seed)."""
    n = len(items)
    if n == 0:
        return [], []
    if n == 1:
        return items, []

    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_test = max(1, min(n - 1, n_test)) if test_size > 0 else 0

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    train = [items[i] for i in train_indices]
    test = [items[i] for i in test_indices]
    return train, test


def stratified_split_class(
    paths: list[Path],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Per-class stratified split preserving class ratio across splits."""
    paths = sorted(paths, key=lambda p: p.name.lower())
    if not paths:
        return [], [], []

    val_test_ratio = val_ratio + test_ratio
    train_paths, val_test_paths = _train_test_split(paths, val_test_ratio, seed)

    relative_test = test_ratio / val_test_ratio if val_test_ratio > 0 else 0.5
    if len(val_test_paths) >= 2:
        val_paths, test_paths = _train_test_split(val_test_paths, relative_test, seed)
    else:
        val_paths = val_test_paths
        test_paths = []

    return train_paths, val_paths, test_paths


def build_split_assignments(
    labeled_paths: dict[str, list[Path]],
    class_to_id: dict[str, int],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> list[SplitAssignment]:
    assignments: list[SplitAssignment] = []

    for class_name, paths in sorted(labeled_paths.items()):
        class_id = class_to_id[class_name]
        train_paths, val_paths, test_paths = stratified_split_class(
            paths,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        for split_name, split_paths in (
            ("train", train_paths),
            ("val", val_paths),
            ("test", test_paths),
        ):
            for path in split_paths:
                assignments.append(
                    SplitAssignment(
                        source_path=path,
                        class_name=class_name,
                        class_id=class_id,
                        split=split_name,
                        filename=path.name,
                    )
                )

    return assignments


def materialize_split(
    assignments: list[SplitAssignment],
    output_root: Path,
) -> None:
    """Copy files into split/train|val|test/<class>/ preserving filenames."""
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in SPLIT_NAMES:
        (output_root / split).mkdir(parents=True, exist_ok=True)

    for entry in assignments:
        dest_dir = output_root / entry.split / entry.class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / entry.filename
        shutil.copy2(entry.source_path, dest)


def _class_distribution(assignments: list[SplitAssignment]) -> dict[str, dict[str, int]]:
    dist: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for entry in assignments:
        dist[entry.class_name][entry.split] += 1
    return {k: dict(v) for k, v in dist.items()}


def validate_split(
    source_root: Path,
    split_root: Path,
    assignments: list[SplitAssignment],
    class_names: list[str],
) -> dict:
    source_files: set[tuple[str, str]] = set()
    for class_name in class_names:
        class_dir = source_root / class_name
        for path in collect_image_paths(class_dir, IMAGE_EXTENSIONS):
            source_files.add((class_name, path.name))

    assigned_sources: set[tuple[str, str]] = set()
    duplicate_assignment: list[str] = []
    for entry in assignments:
        key = (entry.class_name, entry.filename)
        if key in assigned_sources:
            duplicate_assignment.append(f"{entry.class_name}/{entry.filename}")
        assigned_sources.add(key)

    missing = sorted(source_files - assigned_sources)
    extra = sorted(assigned_sources - source_files)

    split_file_counts: dict[str, set[tuple[str, str]]] = {s: set() for s in SPLIT_NAMES}
    split_dup_names: list[str] = []
    image_issues: list[str] = []

    for split in SPLIT_NAMES:
        per_class_names: dict[str, set[str]] = defaultdict(set)
        for class_name in class_names:
            class_dir = split_root / split / class_name
            if not class_dir.exists():
                image_issues.append(f"missing_folder:{split}/{class_name}")
                continue
            files = list(class_dir.iterdir())
            if not files:
                image_issues.append(f"empty_folder:{split}/{class_name}")
            for path in files:
                if not path.is_file():
                    continue
                key = (class_name, path.name)
                split_file_counts[split].add(key)
                if path.name.lower() in per_class_names[class_name]:
                    split_dup_names.append(f"{split}/{class_name}/{path.name}")
                per_class_names[class_name].add(path.name.lower())
                try:
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            image_issues.append(f"not_rgb:{split}/{class_name}/{path.name}:{img.mode}")
                        if img.width != 256 or img.height != 256:
                            image_issues.append(
                                f"bad_size:{split}/{class_name}/{path.name}:{img.width}x{img.height}"
                            )
                        img.verify()
                except Exception as exc:
                    image_issues.append(f"open_failed:{split}/{class_name}/{path.name}:{exc}")

    all_split_files = set().union(*split_file_counts.values())
    overlap_train_val = split_file_counts["train"] & split_file_counts["val"]
    overlap_train_test = split_file_counts["train"] & split_file_counts["test"]
    overlap_val_test = split_file_counts["val"] & split_file_counts["test"]

    classes_missing_split: list[str] = []
    for class_name in class_names:
        dist = _class_distribution(assignments).get(class_name, {})
        for split in SPLIT_NAMES:
            if dist.get(split, 0) == 0:
                classes_missing_split.append(f"{class_name}:{split}")

    passed = (
        not missing
        and not extra
        and not duplicate_assignment
        and not overlap_train_val
        and not overlap_train_test
        and not overlap_val_test
        and not split_dup_names
        and not image_issues
        and len(all_split_files) == len(source_files)
    )

    return {
        "every_image_assigned_once": not missing and not extra and not duplicate_assignment,
        "no_cross_split_duplicates": not (overlap_train_val or overlap_train_test or overlap_val_test),
        "no_duplicate_filenames_in_class": not split_dup_names,
        "all_images_open": not any(i.startswith("open_failed") for i in image_issues),
        "all_rgb": not any(i.startswith("not_rgb") for i in image_issues),
        "all_256x256": not any(i.startswith("bad_size") for i in image_issues),
        "all_classes_in_all_splits": not classes_missing_split,
        "total_source_images": len(source_files),
        "total_split_images": len(all_split_files),
        "missing_from_assignment": missing[:20],
        "extra_in_assignment": extra[:20],
        "cross_split_overlaps": len(overlap_train_val | overlap_train_test | overlap_val_test),
        "classes_missing_split": classes_missing_split,
        "issues": image_issues[:30],
        "passed": passed,
    }


def _split_percentages(counts: dict[str, int], total: int) -> dict[str, float]:
    if total == 0:
        return {"train": 0.0, "val": 0.0, "test": 0.0}
    return {
        split: round(counts.get(split, 0) / total * 100, 2)
        for split in SPLIT_NAMES
    }


def generate_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# PlantDiseaseAI — Tomato Dataset Split Report",
        "",
        f"**Generated:** {report['metadata']['generated_at']}",
        f"**Source:** `{report['metadata']['source_root']}`",
        f"**Destination:** `{report['metadata']['destination_root']}`",
        f"**Random seed:** {report['metadata']['random_seed']}",
        "",
        "## Summary",
        "",
        f"- **Total images:** {s['total_images']:,}",
        f"- **Train:** {s['train_count']:,} ({s['split_percentages']['train']}%)",
        f"- **Validation:** {s['val_count']:,} ({s['split_percentages']['val']}%)",
        f"- **Test:** {s['test_count']:,} ({s['split_percentages']['test']}%)",
        "",
        "## Per-Class Distribution",
        "",
        "| Class | Train | Val | Test | Total | Train % | Val % | Test % |",
        "|-------|------:|----:|-----:|------:|--------:|------:|-------:|",
    ]

    for class_name, dist in sorted(report["per_class_distribution"].items()):
        total = sum(dist.values())
        pcts = _split_percentages(dist, total)
        lines.append(
            f"| `{class_name}` | {dist.get('train', 0)} | {dist.get('val', 0)} | "
            f"{dist.get('test', 0)} | {total} | {pcts['train']} | {pcts['val']} | {pcts['test']} |"
        )

    lines.extend(["", "## Validation", ""])
    for key, value in report["validation"].items():
        if key not in ("issues", "missing_from_assignment", "extra_in_assignment"):
            lines.append(f"- **{key}:** {value}")

    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


class TomatoSplitPipeline:
    """Create stratified train/val/test split for tomato raw dataset."""

    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.source_root = project_root / config.path("paths.raw")
        self.split_root = project_root / "datasets" / "tomato" / "split"
        self.reports_dir = project_root / "reports"
        self.seed = int(config.get("project.seed", 42))
        self.train_ratio = float(config.get("preprocessing.train_ratio", 0.70))
        self.val_ratio = float(config.get("preprocessing.val_ratio", 0.15))
        self.test_ratio = float(config.get("preprocessing.test_ratio", 0.15))

    def run(self) -> TomatoSplitResult:
        result = TomatoSplitResult(success=False)

        if not self.source_root.exists():
            result.warnings.append(f"Source not found: {self.source_root}")
            return result

        class_to_id = {c.folder_name: c.id for c in self.config.class_configs}
        class_names = [c.folder_name for c in self.config.class_configs]

        labeled_paths: dict[str, list[Path]] = {}
        for class_name in class_names:
            class_dir = self.source_root / class_name
            if not class_dir.exists():
                result.warnings.append(f"Missing class folder in raw: {class_name}")
                labeled_paths[class_name] = []
                continue
            labeled_paths[class_name] = collect_image_paths(class_dir, IMAGE_EXTENSIONS)

        total = sum(len(v) for v in labeled_paths.values())
        if total == 0:
            result.warnings.append("No images found in source dataset")
            return result

        logger.info("Splitting %d images across %d classes", total, len(class_names))
        assignments = build_split_assignments(
            labeled_paths,
            class_to_id,
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

        materialize_split(assignments, self.split_root)

        per_class = _class_distribution(assignments)
        train_count = sum(1 for a in assignments if a.split == "train")
        val_count = sum(1 for a in assignments if a.split == "val")
        test_count = sum(1 for a in assignments if a.split == "test")

        validation = validate_split(
            self.source_root,
            self.split_root,
            assignments,
            class_names,
        )

        result.assignments = assignments
        result.total_images = total
        result.train_count = train_count
        result.val_count = val_count
        result.test_count = test_count
        result.per_class_distribution = per_class
        result.validation = validation
        result.success = validation["passed"]

        report_dict = {
            "metadata": {
                "crop": "tomato",
                "source_root": str(self.source_root),
                "destination_root": str(self.split_root),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": self.seed,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "stratified": True,
                "deterministic": True,
            },
            "summary": {
                "total_images": total,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "split_percentages": _split_percentages(
                    {"train": train_count, "val": val_count, "test": test_count},
                    total,
                ),
            },
            "per_class_distribution": per_class,
            "validation": validation,
            "warnings": result.warnings,
        }

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = self.reports_dir / "tomato_split.md"
        json_path = self.reports_dir / "tomato_split.json"
        md_path.write_text(generate_markdown(report_dict), encoding="utf-8")
        json_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        result.report_md_path = md_path
        result.report_json_path = json_path
        return result
