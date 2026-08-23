#!/usr/bin/env python3
"""Read-only audit of the official PlantVillage Tomato dataset."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import imagehash
from PIL import Image

from utils.image_utils import collect_image_paths, validate_image

DATASET_ROOT = PROJECT_ROOT / "PlantVillage-Dataset" / "raw" / "color"
LEAFMAPS_DIR = PROJECT_ROOT / "PlantVillage-Dataset" / "leaf_grouping" / "filtered_leafmaps"
REPORTS_DIR = PROJECT_ROOT / "reports"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HIDDEN_PATTERNS = {
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
    ".gitkeep",
    ".gitignore",
}
INVALID_FILENAME_PATTERN = re.compile(r'[<>:"|?*\x00-\x1f]')


@dataclass
class ClassAudit:
    folder_name: str
    image_count: int = 0
    formats: dict[str, int] = field(default_factory=dict)
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    color_modes: dict[str, int] = field(default_factory=dict)
    zero_byte_files: list[str] = field(default_factory=list)
    corrupted: list[dict] = field(default_factory=list)
    invalid_filenames: list[str] = field(default_factory=list)
    unsupported_files: list[str] = field(default_factory=list)
    hidden_files: list[str] = field(default_factory=list)


def _file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _discover_tomato_classes(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Tomato"))


def _scan_hidden_and_non_images(class_dir: Path) -> tuple[list[str], list[str], list[str]]:
    hidden: list[str] = []
    unsupported: list[str] = []
    for path in class_dir.rglob("*"):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if name_lower in HIDDEN_PATTERNS or name_lower.startswith("."):
            hidden.append(str(path.relative_to(class_dir)))
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            unsupported.append(str(path.relative_to(class_dir)))
    return hidden, unsupported, []


def _audit_class(class_dir: Path) -> ClassAudit:
    audit = ClassAudit(folder_name=class_dir.name)
    hidden, unsupported, _ = _scan_hidden_and_non_images(class_dir)
    audit.hidden_files = hidden
    audit.unsupported_files = unsupported

    for path in sorted(class_dir.iterdir()):
        if not path.is_file():
            continue

        rel = path.name
        if INVALID_FILENAME_PATTERN.search(path.name):
            audit.invalid_filenames.append(rel)

        ext = path.suffix.lower()
        audit.formats[ext] = audit.formats.get(ext, 0) + 1

        if path.stat().st_size == 0:
            audit.zero_byte_files.append(rel)
            continue

        if ext not in SUPPORTED_EXTENSIONS:
            continue

        audit.image_count += 1

        try:
            with Image.open(path) as img:
                audit.widths.append(img.width)
                audit.heights.append(img.height)
                mode = img.mode
                if mode in ("L", "LA"):
                    color_label = "Grayscale"
                elif mode in ("RGB", "RGBA", "CMYK", "P"):
                    color_label = "RGB"
                else:
                    color_label = mode
                audit.color_modes[color_label] = audit.color_modes.get(color_label, 0) + 1
        except Exception as exc:
            audit.corrupted.append({"file": rel, "error": str(exc)})

        validation = validate_image(path)
        if not validation.is_valid and rel not in {c["file"] for c in audit.corrupted}:
            audit.corrupted.append({"file": rel, "error": validation.error or "validation failed"})

    return audit


def _resolution_stats(widths: list[int], heights: list[int]) -> dict:
    if not widths:
        return {
            "min_width": 0,
            "max_width": 0,
            "avg_width": 0.0,
            "min_height": 0,
            "max_height": 0,
            "avg_height": 0.0,
            "min_resolution": "0x0",
            "max_resolution": "0x0",
            "avg_resolution": "0x0",
        }
    avg_w = statistics.mean(widths)
    avg_h = statistics.mean(heights)
    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": round(avg_w, 2),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": round(avg_h, 2),
        "min_resolution": f"{min(widths)}x{min(heights)}",
        "max_resolution": f"{max(widths)}x{max(heights)}",
        "avg_resolution": f"{round(avg_w)}x{round(avg_h)}",
    }


def _class_imbalance(counts: dict[str, int]) -> dict:
    if not counts:
        return {}
    values = list(counts.values())
    total = sum(values)
    min_c = min(values)
    max_c = max(values)
    mean_c = total / len(values)
    return {
        "min_per_class": min_c,
        "max_per_class": max_c,
        "mean_per_class": round(mean_c, 2),
        "imbalance_ratio_max_to_min": round(max_c / min_c, 3) if min_c else None,
        "coefficient_of_variation": round(statistics.pstdev(values) / mean_c, 3) if mean_c else None,
        "majority_class": max(counts, key=counts.get),
        "minority_class": min(counts, key=counts.get),
    }


def _cross_class_duplicates(class_paths: dict[str, list[Path]]) -> dict:
    md5_to_classes: dict[str, list[str]] = defaultdict(list)
    name_to_classes: dict[str, list[str]] = defaultdict(list)

    for class_name, paths in class_paths.items():
        for path in paths:
            digest = _file_md5(path)
            md5_to_classes[digest].append(class_name)
            name_to_classes[path.name.lower()].append(class_name)

    cross_md5 = [
        {"md5": digest, "classes": sorted(set(classes)), "count": len(classes)}
        for digest, classes in md5_to_classes.items()
        if len(set(classes)) > 1
    ]
    cross_name = [
        {"filename": name, "classes": sorted(set(classes))}
        for name, classes in name_to_classes.items()
        if len(set(classes)) > 1
    ]
    return {
        "cross_class_exact_hash_groups": len(cross_md5),
        "cross_class_exact_hash_details": cross_md5[:20],
        "cross_class_same_filename_groups": len(cross_name),
        "cross_class_same_filename_details": cross_name[:20],
        "single_class_assignment_verified": len(cross_md5) == 0,
    }


def _leafmap_missing(class_dirs: list[Path]) -> dict:
    results: dict[str, dict] = {}
    for class_dir in class_dirs:
        csv_path = LEAFMAPS_DIR / f"{class_dir.name}.csv"
        if not csv_path.exists():
            results[class_dir.name] = {
                "leafmap_exists": False,
                "expected_from_leafmap": None,
                "actual_on_disk": len(list(class_dir.glob("*.*"))),
                "missing_from_disk": [],
                "extra_on_disk": [],
            }
            continue

        import csv

        expected: set[str] = set()
        with csv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                expected.add(row["File Name"].strip())

        actual = {p.name for p in class_dir.iterdir() if p.is_file()}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)

        results[class_dir.name] = {
            "leafmap_exists": True,
            "expected_from_leafmap": len(expected),
            "actual_on_disk": len(actual),
            "missing_from_disk_count": len(missing),
            "missing_from_disk_sample": missing[:10],
            "extra_on_disk_count": len(extra),
            "extra_on_disk_sample": extra[:10],
        }
    return results


def _near_duplicates_in_paths(
    paths: list[Path],
    duplicate_paths: set[Path],
    threshold: int,
    start_group_id: int,
) -> tuple[list[dict], int]:
    """Detect near-duplicates within a single class folder."""
    unique_paths = [p for p in paths if p not in duplicate_paths]
    all_hashes: list[tuple[Path, imagehash.ImageHash]] = []
    for path in unique_paths:
        try:
            with Image.open(path) as img:
                all_hashes.append((path, imagehash.phash(img.convert("RGB"))))
        except Exception:
            continue

    near_groups: list[dict] = []
    near_seen: set[Path] = set()
    group_id = start_group_id
    for i, (path_a, hash_a) in enumerate(all_hashes):
        if path_a in near_seen:
            continue
        group_paths = [path_a]
        for path_b, hash_b in all_hashes[i + 1 :]:
            if path_b in near_seen:
                continue
            if hash_a - hash_b <= threshold:
                group_paths.append(path_b)
                near_seen.add(path_b)
        if len(group_paths) > 1:
            near_groups.append(
                {"group_id": group_id, "phash": str(hash_a), "count": len(group_paths), "paths": group_paths}
            )
            group_id += 1
    return near_groups, group_id


def _detect_duplicates(class_paths: dict[str, list[Path]], threshold: int = 5) -> dict:
    all_image_paths = [p for paths in class_paths.values() for p in paths]
    md5_map: dict[str, list[Path]] = defaultdict(list)
    for path in all_image_paths:
        md5_map[_file_md5(path)].append(path)

    exact_groups: list[dict] = []
    duplicate_paths: set[Path] = set()
    group_id = 0
    for digest, paths in md5_map.items():
        if len(paths) > 1:
            exact_groups.append(
                {"group_id": group_id, "hash": digest, "count": len(paths), "paths": paths}
            )
            duplicate_paths.update(paths[1:])
            group_id += 1

    near_groups: list[dict] = []
    for class_name, paths in class_paths.items():
        print(f"  Near-duplicate scan: {class_name} ({len(paths)} images)...")
        class_near, group_id = _near_duplicates_in_paths(
            paths, duplicate_paths, threshold, group_id
        )
        near_groups.extend(class_near)

    exact_count = sum(max(0, g["count"] - 1) for g in exact_groups)
    near_count = sum(max(0, g["count"] - 1) for g in near_groups)
    return {
        "exact_groups": exact_groups,
        "near_groups": near_groups,
        "exact_duplicate_count": exact_count,
        "near_duplicate_count": near_count,
    }


def run_audit() -> dict:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    class_dirs = _discover_tomato_classes(DATASET_ROOT)
    class_audits: dict[str, ClassAudit] = {}
    all_image_paths: list[Path] = []
    class_paths: dict[str, list[Path]] = {}

    for class_dir in class_dirs:
        audit = _audit_class(class_dir)
        class_audits[class_dir.name] = audit
        paths = collect_image_paths(class_dir, tuple(SUPPORTED_EXTENSIONS))
        class_paths[class_dir.name] = paths
        all_image_paths.extend(paths)

    print("Detecting exact duplicates (MD5)...")
    duplicate_result = _detect_duplicates(class_paths)

    all_widths = [w for a in class_audits.values() for w in a.widths]
    all_heights = [h for a in class_audits.values() for h in a.heights]
    global_formats: dict[str, int] = defaultdict(int)
    global_color: dict[str, int] = defaultdict(int)
    for audit in class_audits.values():
        for ext, count in audit.formats.items():
            global_formats[ext] += count
        for mode, count in audit.color_modes.items():
            global_color[mode] += count

    images_per_class = {name: a.image_count for name, a in class_audits.items()}
    total_images = sum(images_per_class.values())

    corrupted_all = [
        {"class": name, **item}
        for name, audit in class_audits.items()
        for item in audit.corrupted
    ]
    zero_byte_all = [
        {"class": name, "file": f}
        for name, audit in class_audits.items()
        for f in audit.zero_byte_files
    ]
    hidden_all = [
        {"class": name, "file": f}
        for name, audit in class_audits.items()
        for f in audit.hidden_files
    ]
    invalid_names = [
        {"class": name, "file": f}
        for name, audit in class_audits.items()
        for f in audit.invalid_filenames
    ]
    unsupported_all = [
        {"class": name, "file": f}
        for name, audit in class_audits.items()
        for f in audit.unsupported_files
    ]

    empty_folders = [d.name for d in class_dirs if class_audits[d.name].image_count == 0]
    cross_class = _cross_class_duplicates(class_paths)
    leafmap_check = _leafmap_missing(class_dirs)
    imbalance = _class_imbalance(images_per_class)

    structure_valid = (
        len(class_dirs) >= 10
        and not empty_folders
        and cross_class["single_class_assignment_verified"]
    )

    return {
        "metadata": {
            "crop": "tomato",
            "source": "PlantVillage (Original) — raw/color",
            "dataset_root": str(DATASET_ROOT),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_mode": "read_only",
        },
        "summary": {
            "total_images": total_images,
            "number_of_classes": len(class_dirs),
            "class_names": [d.name for d in class_dirs],
            "images_per_class": images_per_class,
            "exact_duplicate_count": duplicate_result["exact_duplicate_count"],
            "exact_duplicate_groups": len(duplicate_result["exact_groups"]),
            "near_duplicate_count": duplicate_result["near_duplicate_count"],
            "near_duplicate_groups": len(duplicate_result["near_groups"]),
            "corrupted_images": len(corrupted_all),
            "zero_byte_files": len(zero_byte_all),
            "hidden_or_system_files": len(hidden_all),
            "invalid_filenames": len(invalid_names),
            "unsupported_formats": len(unsupported_all),
            "class_imbalance": imbalance,
            "structure_valid": structure_valid,
        },
        "directory_structure": {
            "root": str(DATASET_ROOT),
            "class_folders": [d.name for d in class_dirs],
            "empty_folders": empty_folders,
            "unexpected_non_tomato_folders": [],
            "validation_passed": structure_valid,
        },
        "global_image_statistics": {
            "formats": dict(global_formats),
            "color_modes": dict(global_color),
            **_resolution_stats(all_widths, all_heights),
        },
        "per_class": {
            name: {
                "folder_name": audit.folder_name,
                "image_count": audit.image_count,
                "formats": audit.formats,
                "color_modes": audit.color_modes,
                **_resolution_stats(audit.widths, audit.heights),
                "zero_byte_files": audit.zero_byte_files,
                "corrupted_count": len(audit.corrupted),
                "corrupted": audit.corrupted,
                "invalid_filenames": audit.invalid_filenames,
                "unsupported_files": audit.unsupported_files,
                "hidden_files": audit.hidden_files,
            }
            for name, audit in class_audits.items()
        },
        "duplicates": {
            "exact_groups": [
                {
                    "group_id": g["group_id"],
                    "hash": g["hash"],
                    "count": g["count"],
                    "paths": [str(p.relative_to(DATASET_ROOT)) for p in g["paths"]],
                }
                for g in duplicate_result["exact_groups"]
            ],
            "near_groups": [
                {
                    "group_id": g["group_id"],
                    "phash": g["phash"],
                    "count": g["count"],
                    "paths": [str(p.relative_to(DATASET_ROOT)) for p in g["paths"]],
                }
                for g in duplicate_result["near_groups"]
            ],
        },
        "cross_class_assignment": cross_class,
        "leafmap_validation": leafmap_check,
        "issues": {
            "corrupted": corrupted_all,
            "zero_byte": zero_byte_all,
            "hidden_files": hidden_all,
            "invalid_filenames": invalid_names,
            "unsupported_formats": unsupported_all,
        },
        "recommended_next_actions": _recommendations(
            total_images,
            imbalance,
            duplicate_result["exact_duplicate_count"],
            duplicate_result["near_duplicate_count"],
            len(corrupted_all),
            len(zero_byte_all),
            cross_class,
            leafmap_check,
        ),
    }


def _recommendations(
    total: int,
    imbalance: dict,
    exact_dups: int,
    near_dups: int,
    corrupted: int,
    zero_byte: int,
    cross_class: dict,
    leafmap: dict,
) -> list[str]:
    actions = [
        "Proceed with read-only preprocessing pipeline design for tomato using PlantVillage raw/color as source.",
        "Define canonical folder names and class mapping YAML (10 classes including Tomato mosaic virus).",
    ]
    if exact_dups:
        actions.append(f"Review {exact_dups} exact duplicate(s) before training; deduplicate during preprocessing.")
    if near_dups:
        actions.append(f"Review {near_dups} near-duplicate(s); apply perceptual-hash dedup (threshold=5) like grape.")
    if corrupted or zero_byte:
        actions.append(f"Exclude or replace {corrupted} corrupted and {zero_byte} zero-byte file(s) during preprocessing.")
    if imbalance.get("imbalance_ratio_max_to_min", 1) and imbalance["imbalance_ratio_max_to_min"] > 3:
        actions.append(
            f"Address class imbalance (ratio {imbalance['imbalance_ratio_max_to_min']}:1): "
            f"consider undersampling majority class '{imbalance.get('majority_class')}' "
            f"or augmenting minority class '{imbalance.get('minority_class')}'."
        )
    if not cross_class["single_class_assignment_verified"]:
        actions.append("Resolve cross-class duplicate images before splitting to prevent data leakage.")
    missing_total = sum(v.get("missing_from_disk_count", 0) for v in leafmap.values())
    if missing_total:
        actions.append(f"Investigate {missing_total} file(s) referenced in leaf maps but missing on disk.")
    actions.append("Copy tomato classes to Dataset/datasets/tomato/raw/ without modifying PlantVillage source.")
    actions.append("Run corruption check and duplicate detection in preprocessing before train/val/test split.")
    return actions


def generate_markdown(report: dict) -> str:
    meta = report["metadata"]
    summary = report["summary"]
    global_stats = report["global_image_statistics"]
    imbalance = summary["class_imbalance"]

    lines = [
        "# PlantDiseaseAI — Tomato Dataset Audit Report",
        "",
        f"**Crop:** {meta['crop']}",
        f"**Source:** {meta['source']}",
        f"**Dataset root:** `{meta['dataset_root']}`",
        f"**Generated:** {meta['generated_at']}",
        f"**Mode:** {meta['analysis_mode']} (no files modified)",
        "",
        "## Executive Summary",
        "",
        f"- **Total images:** {summary['total_images']:,}",
        f"- **Number of classes:** {summary['number_of_classes']}",
        f"- **Exact duplicates:** {summary['exact_duplicate_count']} ({summary['exact_duplicate_groups']} groups)",
        f"- **Near duplicates:** {summary['near_duplicate_count']} ({summary['near_duplicate_groups']} groups)",
        f"- **Corrupted images:** {summary['corrupted_images']}",
        f"- **Zero-byte files:** {summary['zero_byte_files']}",
        f"- **Hidden/system files:** {summary['hidden_or_system_files']}",
        f"- **Structure valid:** {summary['structure_valid']}",
        "",
        "## Class Folders",
        "",
        "| # | Folder Name | Images | Min Resolution | Max Resolution | Avg Resolution |",
        "|---|-------------|--------|----------------|----------------|----------------|",
    ]

    for i, (name, data) in enumerate(report["per_class"].items(), 1):
        lines.append(
            f"| {i} | `{name}` | {data['image_count']:,} | "
            f"{data['min_resolution']} | {data['max_resolution']} | {data['avg_resolution']} |"
        )

    lines.extend([
        "",
        "## Images Per Class",
        "",
        "| Class | Count | % of Total |",
        "|-------|------:|-----------:|",
    ])
    total = summary["total_images"]
    for name, count in sorted(summary["images_per_class"].items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        lines.append(f"| `{name}` | {count:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Global Image Statistics",
        "",
        f"- **Formats:** {global_stats['formats']}",
        f"- **Color modes:** {global_stats['color_modes']}",
        f"- **Min resolution:** {global_stats['min_resolution']}",
        f"- **Max resolution:** {global_stats['max_resolution']}",
        f"- **Average resolution:** {global_stats['avg_resolution']}",
        f"- **Width range:** {global_stats['min_width']}–{global_stats['max_width']} (avg {global_stats['avg_width']})",
        f"- **Height range:** {global_stats['min_height']}–{global_stats['max_height']} (avg {global_stats['avg_height']})",
        "",
        "## Class Imbalance",
        "",
        f"- **Min per class:** {imbalance.get('min_per_class')}",
        f"- **Max per class:** {imbalance.get('max_per_class')}",
        f"- **Mean per class:** {imbalance.get('mean_per_class')}",
        f"- **Imbalance ratio (max/min):** {imbalance.get('imbalance_ratio_max_to_min')}",
        f"- **Coefficient of variation:** {imbalance.get('coefficient_of_variation')}",
        f"- **Majority class:** `{imbalance.get('majority_class')}`",
        f"- **Minority class:** `{imbalance.get('minority_class')}`",
        "",
        "## Directory Structure Validation",
        "",
        f"- **Class folders found:** {summary['number_of_classes']}",
        f"- **Empty folders:** {report['directory_structure']['empty_folders'] or 'None'}",
        f"- **Validation passed:** {report['directory_structure']['validation_passed']}",
        "",
        "## Single-Class Assignment",
        "",
        f"- **Cross-class exact hash conflicts:** {report['cross_class_assignment']['cross_class_exact_hash_groups']}",
        f"- **Cross-class same-filename conflicts:** {report['cross_class_assignment']['cross_class_same_filename_groups']}",
        f"- **Every image in exactly one class:** {report['cross_class_assignment']['single_class_assignment_verified']}",
        "",
        "## Duplicate Detection",
        "",
        f"- **Exact duplicate groups:** {summary['exact_duplicate_groups']}",
        f"- **Exact duplicates (extra copies):** {summary['exact_duplicate_count']}",
        f"- **Near duplicate groups:** {summary['near_duplicate_groups']}",
        f"- **Near duplicates (extra copies):** {summary['near_duplicate_count']}",
        "",
    ])

    if report["duplicates"]["exact_groups"]:
        lines.append("### Exact Duplicate Groups (sample)")
        lines.append("")
        for group in report["duplicates"]["exact_groups"][:10]:
            lines.append(f"- Group {group['group_id']} ({group['count']} files): {group['paths'][0]} ...")
        lines.append("")

    if report["duplicates"]["near_groups"]:
        lines.append("### Near Duplicate Groups (sample)")
        lines.append("")
        for group in report["duplicates"]["near_groups"][:10]:
            lines.append(f"- Group {group['group_id']} ({group['count']} files): {', '.join(group['paths'][:3])}")
        lines.append("")

    lines.extend([
        "## Data Quality Issues",
        "",
        f"- **Corrupted images:** {summary['corrupted_images']}",
        f"- **Zero-byte files:** {summary['zero_byte_files']}",
        f"- **Invalid filenames:** {summary['invalid_filenames']}",
        f"- **Unsupported formats:** {summary['unsupported_formats']}",
        f"- **Hidden/system files:** {summary['hidden_or_system_files']}",
        "",
    ])

    if report["issues"]["corrupted"]:
        lines.append("### Corrupted Files")
        for item in report["issues"]["corrupted"][:20]:
            lines.append(f"- `{item['class']}/{item['file']}`: {item['error']}")
        lines.append("")

    if report["issues"]["zero_byte"]:
        lines.append("### Zero-Byte Files")
        for item in report["issues"]["zero_byte"]:
            lines.append(f"- `{item['class']}/{item['file']}`")
        lines.append("")

    lines.extend([
        "## Leaf Map Validation",
        "",
        "| Class | Leafmap | Expected | On Disk | Missing | Extra |",
        "|-------|---------|----------|---------|---------|-------|",
    ])
    for name, check in report["leafmap_validation"].items():
        lines.append(
            f"| `{name}` | {check.get('leafmap_exists', False)} | "
            f"{check.get('expected_from_leafmap', '—')} | {check.get('actual_on_disk', '—')} | "
            f"{check.get('missing_from_disk_count', '—')} | {check.get('extra_on_disk_count', '—')} |"
        )

    lines.extend([
        "",
        "## Recommended Next Actions",
        "",
    ])
    for i, action in enumerate(report["recommended_next_actions"], 1):
        lines.append(f"{i}. {action}")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    print("Scanning PlantVillage Tomato dataset (read-only)...")
    report = run_audit()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "tomato_dataset_audit.md"
    json_path = REPORTS_DIR / "tomato_dataset_audit.json"

    md_path.write_text(generate_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    imb = s["class_imbalance"]
    print("\n=== Tomato Dataset Audit Summary ===")
    print(f"Total images:     {s['total_images']:,}")
    print(f"Number of classes: {s['number_of_classes']}")
    print("Images per class:")
    for name, count in sorted(s["images_per_class"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {count:,}")
    print(f"Exact duplicates:  {s['exact_duplicate_count']} ({s['exact_duplicate_groups']} groups)")
    print(f"Near duplicates:   {s['near_duplicate_count']} ({s['near_duplicate_groups']} groups)")
    print(f"Corrupted images:  {s['corrupted_images']}")
    print(f"Zero-byte files:   {s['zero_byte_files']}")
    print(f"Class imbalance:   ratio {imb.get('imbalance_ratio_max_to_min')} (max/min)")
    print(f"Reports saved:")
    print(f"  {md_path}")
    print(f"  {json_path}")
    print("\nRecommended next actions:")
    for action in report["recommended_next_actions"][:5]:
        print(f"  • {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
