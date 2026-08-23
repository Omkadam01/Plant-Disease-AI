#!/usr/bin/env python3
"""Read-only audit of the PlantVillage Potato dataset for PlantDiseaseAI."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import imagehash
from PIL import Image

from utils.image_utils import collect_image_paths, validate_image

CANONICAL_ROOT = PROJECT_ROOT / "datasets" / "potato"
PLANTVILLAGE_ROOT = PROJECT_ROOT / "PlantVillage-Dataset" / "raw" / "color"
LEAFMAPS_DIR = PROJECT_ROOT / "PlantVillage-Dataset" / "leaf_grouping" / "filtered_leafmaps"
REPORTS_DIR = CANONICAL_ROOT / "reports"

EXPECTED_FOLDERS = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy",
}

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
    display_name: str
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


def _resolve_dataset_root() -> tuple[Path, str]:
    """Prefer Dataset/datasets/potato/raw, then Dataset/datasets/potato, else PlantVillage source."""
    candidates = [
        (CANONICAL_ROOT / "raw", "Dataset/datasets/potato/raw"),
        (CANONICAL_ROOT, "Dataset/datasets/potato"),
        (PLANTVILLAGE_ROOT, "PlantVillage-Dataset/raw/color"),
    ]
    for root, label in candidates:
        if not root.exists():
            continue
        potato_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Potato")]
        if potato_dirs:
            return root, label
    raise FileNotFoundError(
        "No Potato class folders found under Dataset/datasets/potato/ or PlantVillage-Dataset/raw/color"
    )


def _file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _discover_potato_classes(root: Path) -> list[Path]:
    found = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Potato"))
    return found


def _scan_hidden_and_non_images(class_dir: Path) -> tuple[list[str], list[str]]:
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
    return hidden, unsupported


def _audit_class(class_dir: Path) -> ClassAudit:
    display = EXPECTED_FOLDERS.get(class_dir.name, class_dir.name)
    audit = ClassAudit(folder_name=class_dir.name, display_name=display)
    hidden, unsupported = _scan_hidden_and_non_images(class_dir)
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
        corrupted_files = {c["file"] for c in audit.corrupted}
        if not validation.is_valid and rel not in corrupted_files:
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


def _leafmap_disk_name(filename: str) -> str:
    """Map on-disk PlantVillage filename to leafmap short name when prefixed with UUID."""
    if "___" in filename:
        return filename.split("___", 1)[1]
    return filename


def _leafmap_missing(class_dirs: list[Path]) -> dict:
    import csv

    results: dict[str, dict] = {}
    for class_dir in class_dirs:
        csv_path = LEAFMAPS_DIR / f"{class_dir.name}.csv"
        if not csv_path.exists():
            results[class_dir.name] = {
                "leafmap_exists": False,
                "expected_from_leafmap": None,
                "actual_on_disk": len([p for p in class_dir.iterdir() if p.is_file()]),
                "missing_from_disk_count": None,
                "extra_on_disk_count": None,
            }
            continue

        expected: set[str] = set()
        with csv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                expected.add(row["File Name"].strip())

        actual_files = [p.name for p in class_dir.iterdir() if p.is_file()]
        actual_short = {_leafmap_disk_name(name) for name in actual_files}

        missing = sorted(expected - actual_short)
        extra = sorted(actual_short - expected)

        results[class_dir.name] = {
            "leafmap_exists": True,
            "expected_from_leafmap": len(expected),
            "actual_on_disk": len(actual_files),
            "missing_from_disk_count": len(missing),
            "missing_from_disk_sample": missing[:10],
            "extra_on_disk_count": len(extra),
            "extra_on_disk_sample": extra[:10],
            "leafmap_match_strategy": "strip UUID prefix before ___ for comparison",
        }
    return results


def _near_duplicates_in_paths(
    paths: list[Path],
    duplicate_paths: set[Path],
    threshold: int,
    start_group_id: int,
) -> tuple[list[dict], int]:
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


def _quality_summary(
    corrupted: int,
    zero_byte: int,
    exact_dups: int,
    near_dups: int,
    cross_class: dict,
    missing_folders: list[str],
    unexpected_folders: list[str],
) -> dict:
    issues: list[str] = []
    if corrupted:
        issues.append(f"{corrupted} corrupted image(s)")
    if zero_byte:
        issues.append(f"{zero_byte} zero-byte file(s)")
    if exact_dups:
        issues.append(f"{exact_dups} exact duplicate(s)")
    if near_dups:
        issues.append(f"{near_dups} near duplicate(s)")
    if missing_folders:
        issues.append(f"missing folder(s): {', '.join(missing_folders)}")
    if unexpected_folders:
        issues.append(f"unexpected folder(s): {', '.join(unexpected_folders)}")
    if not cross_class["single_class_assignment_verified"]:
        issues.append("cross-class duplicate assignment detected")

    if not issues:
        verdict = "Dataset quality is good — suitable for preprocessing with minor class imbalance noted."
    elif corrupted or zero_byte or not cross_class["single_class_assignment_verified"]:
        verdict = "Dataset quality issues found — resolve before training."
    else:
        verdict = "Dataset quality is acceptable — review noted issues before preprocessing."

    return {
        "overall_verdict": verdict,
        "issues_found": issues or ["None"],
        "cross_class_leakage_risk": not cross_class["single_class_assignment_verified"],
    }


def _recommendations(
    imbalance: dict,
    exact_dups: int,
    near_dups: int,
    corrupted: int,
    zero_byte: int,
    cross_class: dict,
    leafmap: dict,
    audited_from: str,
) -> list[str]:
    actions = [
        "Proceed with potato raw preprocessing: copy PlantVillage classes to Dataset/datasets/potato/raw/.",
        "Define Models/configs/crops/potato.yaml with 3 classes (Early Blight, Healthy, Late Blight).",
    ]
    if exact_dups:
        actions.append(f"Review {exact_dups} exact duplicate(s) before training; deduplicate during preprocessing.")
    if near_dups:
        actions.append(f"Review {near_dups} near-duplicate(s); apply perceptual-hash dedup (threshold=5) like tomato.")
    if corrupted or zero_byte:
        actions.append(f"Exclude or replace {corrupted} corrupted and {zero_byte} zero-byte file(s) during preprocessing.")
    ratio = imbalance.get("imbalance_ratio_max_to_min")
    if ratio and ratio > 3:
        actions.append(
            f"Address class imbalance (ratio {ratio}:1): consider balancing Healthy "
            f"({imbalance.get('minority_class')}) against Early/Late Blight during preprocessing."
        )
    if not cross_class["single_class_assignment_verified"]:
        actions.append("Resolve cross-class duplicate images before splitting to prevent data leakage.")
    missing_total = sum(v.get("missing_from_disk_count") or 0 for v in leafmap.values())
    if missing_total:
        actions.append(f"Investigate {missing_total} file(s) referenced in leaf maps but missing on disk.")
    if "PlantVillage" in audited_from:
        actions.append("Copy potato classes to Dataset/datasets/potato/raw/ without modifying PlantVillage source.")
    actions.append("Run corruption check and duplicate detection in preprocessing before train/val/test split.")
    return actions


def run_audit() -> dict:
    dataset_root, audited_from = _resolve_dataset_root()
    class_dirs = _discover_potato_classes(dataset_root)
    found_names = {d.name for d in class_dirs}
    missing_folders = sorted(set(EXPECTED_FOLDERS) - found_names)
    unexpected_folders = sorted(found_names - set(EXPECTED_FOLDERS))

    class_audits: dict[str, ClassAudit] = {}
    class_paths: dict[str, list[Path]] = {}

    for class_dir in class_dirs:
        audit = _audit_class(class_dir)
        class_audits[class_dir.name] = audit
        paths = collect_image_paths(class_dir, tuple(SUPPORTED_EXTENSIONS))
        class_paths[class_dir.name] = paths

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
    zero_byte_all = [{"class": name, "file": f} for name, audit in class_audits.items() for f in audit.zero_byte_files]
    hidden_all = [{"class": name, "file": f} for name, audit in class_audits.items() for f in audit.hidden_files]
    invalid_names = [{"class": name, "file": f} for name, audit in class_audits.items() for f in audit.invalid_filenames]
    unsupported_all = [{"class": name, "file": f} for name, audit in class_audits.items() for f in audit.unsupported_files]

    empty_folders = [d.name for d in class_dirs if class_audits[d.name].image_count == 0]
    cross_class = _cross_class_duplicates(class_paths)
    leafmap_check = _leafmap_missing(class_dirs)
    imbalance = _class_imbalance(images_per_class)

    structure_valid = (
        len(missing_folders) == 0
        and len(unexpected_folders) == 0
        and not empty_folders
        and cross_class["single_class_assignment_verified"]
    )

    quality = _quality_summary(
        len(corrupted_all),
        len(zero_byte_all),
        duplicate_result["exact_duplicate_count"],
        duplicate_result["near_duplicate_count"],
        cross_class,
        missing_folders,
        unexpected_folders,
    )

    return {
        "metadata": {
            "crop": "potato",
            "source": "PlantVillage (Original)",
            "canonical_dataset_path": str(CANONICAL_ROOT),
            "audited_source_path": str(dataset_root),
            "audited_from": audited_from,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_mode": "read_only",
        },
        "summary": {
            "total_images": total_images,
            "number_of_classes": len(class_dirs),
            "expected_classes": len(EXPECTED_FOLDERS),
            "class_names": [d.name for d in class_dirs],
            "display_names": {k: EXPECTED_FOLDERS[k] for k in sorted(found_names & set(EXPECTED_FOLDERS))},
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
            "invalid_images": len(corrupted_all) + len(zero_byte_all) + len(invalid_names),
            "class_imbalance": imbalance,
            "structure_valid": structure_valid,
            "dataset_quality": quality,
        },
        "directory_structure": {
            "canonical_root": str(CANONICAL_ROOT),
            "audited_root": str(dataset_root),
            "expected_folders": list(EXPECTED_FOLDERS.keys()),
            "found_folders": [d.name for d in class_dirs],
            "missing_folders": missing_folders,
            "unexpected_folders": unexpected_folders,
            "empty_folders": empty_folders,
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
                "display_name": audit.display_name,
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
                    "paths": [str(p.relative_to(dataset_root)) for p in g["paths"]],
                }
                for g in duplicate_result["exact_groups"]
            ],
            "near_groups": [
                {
                    "group_id": g["group_id"],
                    "phash": g["phash"],
                    "count": g["count"],
                    "paths": [str(p.relative_to(dataset_root)) for p in g["paths"]],
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
            imbalance,
            duplicate_result["exact_duplicate_count"],
            duplicate_result["near_duplicate_count"],
            len(corrupted_all),
            len(zero_byte_all),
            cross_class,
            leafmap_check,
            audited_from,
        ),
    }


def generate_markdown(report: dict) -> str:
    meta = report["metadata"]
    summary = report["summary"]
    global_stats = report["global_image_statistics"]
    imbalance = summary["class_imbalance"]
    quality = summary["dataset_quality"]
    structure = report["directory_structure"]

    lines = [
        "# PlantDiseaseAI — Potato Dataset Audit Report",
        "",
        f"**Crop:** {meta['crop']}",
        f"**Source:** {meta['source']}",
        f"**Canonical path:** `{meta['canonical_dataset_path']}`",
        f"**Audited source:** `{meta['audited_source_path']}` ({meta['audited_from']})",
        f"**Generated:** {meta['generated_at']}",
        f"**Mode:** {meta['analysis_mode']} (no files modified)",
        "",
        "## Executive Summary",
        "",
        f"- **Total images:** {summary['total_images']:,}",
        f"- **Number of classes:** {summary['number_of_classes']} (expected {summary['expected_classes']})",
        f"- **Exact duplicates:** {summary['exact_duplicate_count']} ({summary['exact_duplicate_groups']} groups)",
        f"- **Near duplicates:** {summary['near_duplicate_count']} ({summary['near_duplicate_groups']} groups)",
        f"- **Corrupted images:** {summary['corrupted_images']}",
        f"- **Invalid images (total):** {summary['invalid_images']}",
        f"- **Zero-byte files:** {summary['zero_byte_files']}",
        f"- **Hidden/system files:** {summary['hidden_or_system_files']}",
        f"- **Structure valid:** {summary['structure_valid']}",
        "",
        "## Class Folders",
        "",
        "| # | Folder Name | Display Name | Images | Min Resolution | Max Resolution | Avg Resolution |",
        "|---|-------------|--------------|--------|----------------|----------------|----------------|",
    ]

    for i, (name, data) in enumerate(report["per_class"].items(), 1):
        lines.append(
            f"| {i} | `{name}` | {data['display_name']} | {data['image_count']:,} | "
            f"{data['min_resolution']} | {data['max_resolution']} | {data['avg_resolution']} |"
        )

    lines.extend([
        "",
        "## Images Per Class",
        "",
        "| Display Name | Folder | Count | % of Total |",
        "|--------------|--------|------:|-----------:|",
    ])
    total = summary["total_images"]
    for name, data in sorted(report["per_class"].items(), key=lambda x: -x[1]["image_count"]):
        count = data["image_count"]
        pct = 100 * count / total if total else 0
        lines.append(f"| {data['display_name']} | `{name}` | {count:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Folder Structure",
        "",
        f"- **Expected folders:** {structure['expected_folders']}",
        f"- **Found folders:** {structure['found_folders']}",
        f"- **Missing expected folders:** {structure['missing_folders'] or 'None'}",
        f"- **Unexpected folders:** {structure['unexpected_folders'] or 'None'}",
        f"- **Empty folders:** {structure['empty_folders'] or 'None'}",
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
        "## Class Imbalance Report",
        "",
        f"- **Min per class:** {imbalance.get('min_per_class')}",
        f"- **Max per class:** {imbalance.get('max_per_class')}",
        f"- **Mean per class:** {imbalance.get('mean_per_class')}",
        f"- **Imbalance ratio (max/min):** {imbalance.get('imbalance_ratio_max_to_min')}",
        f"- **Coefficient of variation:** {imbalance.get('coefficient_of_variation')}",
        f"- **Majority class:** `{imbalance.get('majority_class')}`",
        f"- **Minority class:** `{imbalance.get('minority_class')}`",
        "",
        "## Dataset Quality Summary",
        "",
        f"- **Overall verdict:** {quality['overall_verdict']}",
        f"- **Issues found:** {', '.join(quality['issues_found'])}",
        f"- **Cross-class leakage risk:** {'Yes' if quality['cross_class_leakage_risk'] else 'No'}",
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

    if report["duplicates"]["near_groups"]:
        lines.append("### Near Duplicate Groups")
        lines.append("")
        for group in report["duplicates"]["near_groups"]:
            paths = ", ".join(group["paths"])
            lines.append(f"- Group {group['group_id']} ({group['count']} files): {paths}")
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
    print("Scanning Potato dataset (read-only)...")
    report = run_audit()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "audit_report.md"
    json_path = REPORTS_DIR / "audit_report.json"

    md_path.write_text(generate_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    imb = s["class_imbalance"]
    print("\n=== Potato Dataset Audit Summary ===")
    print(f"Total images:      {s['total_images']:,}")
    print(f"Number of classes: {s['number_of_classes']}")
    print("Images per class:")
    for name, count in sorted(s["images_per_class"].items(), key=lambda x: -x[1]):
        display = EXPECTED_FOLDERS.get(name, name)
        print(f"  {display} ({name}): {count:,}")
    print(f"Exact duplicates:  {s['exact_duplicate_count']} ({s['exact_duplicate_groups']} groups)")
    print(f"Near duplicates:   {s['near_duplicate_count']} ({s['near_duplicate_groups']} groups)")
    print(f"Corrupted images:  {s['corrupted_images']}")
    print(f"Class imbalance:   ratio {imb.get('imbalance_ratio_max_to_min')} (max/min)")
    print(f"Quality verdict:   {s['dataset_quality']['overall_verdict']}")
    print("Reports saved:")
    print(f"  {md_path}")
    print(f"  {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
