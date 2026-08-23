"""Tomato raw dataset preprocessing — copy, clean, deduplicate, validate."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from PIL import Image

from utils.config import AppConfig
from utils.image_utils import validate_image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HIDDEN_NAMES = {".ds_store", "thumbs.db", "desktop.ini", ".gitkeep", ".gitignore"}
PLANTVILLAGE_TOMATO_ROOT = Path("PlantVillage-Dataset") / "raw" / "color"


@dataclass
class SkippedFile:
    source: str
    class_name: str
    reason: str


@dataclass
class TomatoRawPreprocessResult:
    success: bool
    images_copied: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    files_skipped: int = 0
    final_image_count: int = 0
    images_per_class: dict[str, int] = field(default_factory=dict)
    skipped_files: list[SkippedFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    report_md_path: Path | None = None
    report_json_path: Path | None = None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _quality_score(path: Path) -> tuple[int, int]:
    """Higher is better: (file_size, negative name length for tie-break)."""
    return (path.stat().st_size, -len(path.name))


def _is_hidden(path: Path) -> bool:
    name = path.name.lower()
    return name in HIDDEN_NAMES or name.startswith(".")


def _discover_source_classes(source_root: Path) -> list[Path]:
    return sorted(p for p in source_root.iterdir() if p.is_dir() and p.name.startswith("Tomato"))


def _collect_valid_sources(class_dir: Path, class_name: str, skipped: list[SkippedFile]) -> list[Path]:
    valid: list[Path] = []
    for path in sorted(class_dir.iterdir()):
        if not path.is_file():
            continue
        if _is_hidden(path):
            skipped.append(SkippedFile(str(path), class_name, "hidden_or_system_file"))
            continue
        if path.stat().st_size == 0:
            skipped.append(SkippedFile(str(path), class_name, "zero_byte"))
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append(SkippedFile(str(path), class_name, f"unsupported_format:{ext}"))
            continue
        validation = validate_image(path)
        if not validation.is_valid:
            skipped.append(
                SkippedFile(str(path), class_name, f"corrupted:{validation.error}")
            )
            continue
        valid.append(path)
    return valid


def _select_exact_duplicates(paths: list[Path]) -> tuple[list[Path], int]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[_sha256_file(path)].append(path)

    kept: list[Path] = []
    removed = 0
    for group in groups.values():
        best = max(group, key=_quality_score)
        kept.append(best)
        removed += len(group) - 1
    return sorted(kept, key=lambda p: p.name.lower()), removed


def _perceptual_hash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


def _select_near_duplicates(paths: list[Path], threshold: int) -> tuple[list[Path], int]:
    if len(paths) <= 1:
        return paths, 0

    hashed = [(path, _perceptual_hash(path)) for path in paths]
    removed: set[Path] = set()
    removed_count = 0

    for i, (path_a, hash_a) in enumerate(hashed):
        if path_a in removed:
            continue
        group = [path_a]
        for path_b, hash_b in hashed[i + 1 :]:
            if path_b in removed:
                continue
            if hash_a - hash_b <= threshold:
                group.append(path_b)

        if len(group) > 1:
            best = max(group, key=_quality_score)
            for path in group:
                if path != best:
                    removed.add(path)
                    removed_count += 1

    kept = [path for path in paths if path not in removed]
    return kept, removed_count


def _dest_filename(src: Path, used_names: set[str]) -> str:
    stem = src.stem
    candidate = f"{stem}.jpg"
    if candidate.lower() not in used_names:
        used_names.add(candidate.lower())
        return candidate
    counter = 1
    while True:
        candidate = f"{stem}_{counter}.jpg"
        if candidate.lower() not in used_names:
            used_names.add(candidate.lower())
            return candidate
        counter += 1


def _copy_as_jpg(src: Path, dest: Path) -> None:
    ext = src.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        shutil.copy2(src, dest)
        return
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        rgb.save(dest, format="JPEG", quality=95, subsampling=0)


def _validate_destination(raw_root: Path, class_names: list[str]) -> dict:
    results = {
        "all_images_open": True,
        "all_rgb": True,
        "all_256x256": True,
        "single_class_assignment": True,
        "no_duplicate_filenames": True,
        "no_empty_folders": True,
        "issues": [],
    }

    all_names: dict[str, list[str]] = defaultdict(list)

    for class_name in class_names:
        class_dir = raw_root / class_name
        if not class_dir.exists():
            results["no_empty_folders"] = False
            results["issues"].append(f"missing_class_folder:{class_name}")
            continue

        files = [p for p in class_dir.iterdir() if p.is_file()]
        if not files:
            results["no_empty_folders"] = False
            results["issues"].append(f"empty_folder:{class_name}")

        for path in files:
            all_names[path.name.lower()].append(class_name)
            try:
                with Image.open(path) as img:
                    if img.mode not in ("RGB",):
                        results["all_rgb"] = False
                        results["issues"].append(f"not_rgb:{class_name}/{path.name}:{img.mode}")
                    if img.width != 256 or img.height != 256:
                        results["all_256x256"] = False
                        results["issues"].append(
                            f"bad_resolution:{class_name}/{path.name}:{img.width}x{img.height}"
                        )
                    img.verify()
            except Exception as exc:
                results["all_images_open"] = False
                results["issues"].append(f"open_failed:{class_name}/{path.name}:{exc}")

    dup_names = {name: classes for name, classes in all_names.items() if len(classes) > 1}
    if dup_names:
        results["no_duplicate_filenames"] = False
        for name, classes in list(dup_names.items())[:10]:
            results["issues"].append(f"duplicate_filename:{name}:{'|'.join(classes)}")

    if dup_names:
        results["single_class_assignment"] = False

    results["passed"] = all(
        results[key]
        for key in (
            "all_images_open",
            "all_rgb",
            "all_256x256",
            "single_class_assignment",
            "no_duplicate_filenames",
            "no_empty_folders",
        )
    )
    return results


def _generate_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# PlantDiseaseAI — Tomato Raw Preprocessing Report",
        "",
        f"**Generated:** {report['metadata']['generated_at']}",
        f"**Source:** `{report['metadata']['source_root']}`",
        f"**Destination:** `{report['metadata']['destination_root']}`",
        "",
        "## Summary",
        "",
        f"- **Images copied:** {s['images_copied']:,}",
        f"- **Exact duplicates removed:** {s['exact_duplicates_removed']}",
        f"- **Near duplicates removed:** {s['near_duplicates_removed']}",
        f"- **Files skipped:** {s['files_skipped']}",
        f"- **Final image count:** {s['final_image_count']:,}",
        "",
        "## Images Per Class",
        "",
        "| Class | Count |",
        "|-------|------:|",
    ]
    for name, count in sorted(s["images_per_class"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{name}` | {count:,} |")

    lines.extend([
        "",
        "## Validation",
        "",
    ])
    for key, value in report["validation"].items():
        if key != "issues":
            lines.append(f"- **{key}:** {value}")

    if report["validation"].get("issues"):
        lines.extend(["", "### Issues", ""])
        for issue in report["validation"]["issues"][:30]:
            lines.append(f"- {issue}")

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    if report["skipped_samples"]:
        lines.extend(["", "## Skipped Files (sample)", ""])
        for item in report["skipped_samples"][:20]:
            lines.append(f"- `{item['class_name']}`: {item['source']} — {item['reason']}")

    lines.append("")
    return "\n".join(lines)


class TomatoRawPreprocessor:
    """Copy and clean PlantVillage tomato images into datasets/tomato/raw/."""

    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.source_root = project_root / PLANTVILLAGE_TOMATO_ROOT
        self.dest_root = project_root / config.path("paths.raw")
        self.reports_dir = project_root / "reports"
        self.threshold = int(config.get("preprocessing.near_duplicate_threshold", 5))

    def run(self) -> TomatoRawPreprocessResult:
        result = TomatoRawPreprocessResult(success=False)

        if not self.source_root.exists():
            result.warnings.append(f"Source not found: {self.source_root}")
            return result

        if self.dest_root.exists():
            logger.info("Clearing existing destination: %s", self.dest_root)
            shutil.rmtree(self.dest_root)
        self.dest_root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        class_dirs = _discover_source_classes(self.source_root)
        expected_classes = [c.folder_name for c in self.config.class_configs]
        found_names = [d.name for d in class_dirs]

        missing = set(expected_classes) - set(found_names)
        extra = set(found_names) - set(expected_classes)
        if missing:
            result.warnings.append(f"Missing expected classes in source: {sorted(missing)}")
        if extra:
            result.warnings.append(f"Unexpected classes in source: {sorted(extra)}")

        total_exact_removed = 0
        total_near_removed = 0
        total_copied = 0
        images_per_class: dict[str, int] = {}

        for class_dir in class_dirs:
            class_name = class_dir.name
            logger.info("Processing class: %s", class_name)

            valid_paths = _collect_valid_sources(class_dir, class_name, result.skipped_files)
            after_exact, exact_removed = _select_exact_duplicates(valid_paths)
            after_near, near_removed = _select_near_duplicates(after_exact, self.threshold)

            total_exact_removed += exact_removed
            total_near_removed += near_removed

            dest_class_dir = self.dest_root / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            used_names: set[str] = set()

            for src in after_near:
                filename = _dest_filename(src, used_names)
                dest = dest_class_dir / filename
                _copy_as_jpg(src, dest)
                total_copied += 1

            images_per_class[class_name] = len(after_near)
            logger.info(
                "  valid=%d exact_removed=%d near_removed=%d copied=%d",
                len(valid_paths),
                exact_removed,
                near_removed,
                len(after_near),
            )

        validation = _validate_destination(self.dest_root, found_names)

        result.success = validation["passed"]
        result.images_copied = total_copied
        result.exact_duplicates_removed = total_exact_removed
        result.near_duplicates_removed = total_near_removed
        result.files_skipped = len(result.skipped_files)
        result.final_image_count = total_copied
        result.images_per_class = images_per_class
        result.validation = validation

        report_dict = {
            "metadata": {
                "crop": "tomato",
                "source_root": str(self.source_root),
                "destination_root": str(self.dest_root),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "near_duplicate_threshold": self.threshold,
                "hash_algorithm_exact": "sha256",
                "hash_algorithm_near": "phash",
                "deterministic": True,
            },
            "summary": {
                "images_copied": total_copied,
                "exact_duplicates_removed": total_exact_removed,
                "near_duplicates_removed": total_near_removed,
                "files_skipped": len(result.skipped_files),
                "final_image_count": total_copied,
                "images_per_class": images_per_class,
            },
            "validation": validation,
            "warnings": result.warnings,
            "skipped_samples": [asdict(s) for s in result.skipped_files[:50]],
            "skipped_by_reason": _count_skip_reasons(result.skipped_files),
        }

        md_path = self.reports_dir / "tomato_preprocessing.md"
        json_path = self.reports_dir / "tomato_preprocessing.json"
        md_path.write_text(_generate_markdown(report_dict), encoding="utf-8")
        json_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        result.report_md_path = md_path
        result.report_json_path = json_path
        return result


def _count_skip_reasons(skipped: list[SkippedFile]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in skipped:
        reason_key = item.reason.split(":")[0]
        counts[reason_key] += 1
    return dict(counts)
