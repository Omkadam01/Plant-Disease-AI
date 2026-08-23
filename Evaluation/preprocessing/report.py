"""Dataset audit report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from preprocessing.metadata import DatasetMetadata


def generate_audit_report_markdown(metadata: DatasetMetadata) -> str:
    """Generate human-readable Markdown audit report."""
    lines = [
        "# PlantDiseaseAI Dataset Audit Report",
        "",
        f"**Crop:** {metadata.crop}",
        f"**Source:** {metadata.dataset_source}",
        f"**Generated:** {metadata.created_at}",
        "",
        "## Summary",
        "",
        f"- Total images (after deduplication): **{metadata.total_images}**",
        f"- Number of classes: **{metadata.num_classes}**",
        f"- Train / Val / Test: **{metadata.split_counts['train']}** / "
        f"**{metadata.split_counts['val']}** / **{metadata.split_counts['test']}**",
        "",
        "## Classes",
        "",
        "| ID | Folder | Display Name |",
        "|----|--------|--------------|",
    ]

    for cls in metadata.classes:
        lines.append(f"| {cls['id']} | {cls['folder_name']} | {cls['display_name']} |")

    lines.extend([
        "",
        "## Split Distribution",
        "",
        "| Class | Train | Val | Test | Total |",
        "|-------|-------|-----|------|-------|",
    ])

    for class_name, counts in metadata.class_distribution.items():
        total = counts["train"] + counts["val"] + counts["test"]
        lines.append(
            f"| {class_name} | {counts['train']} | {counts['val']} | "
            f"{counts['test']} | {total} |"
        )

    lines.extend([
        "",
        "## Validation",
        "",
        f"- Valid structure: **{metadata.validation['is_valid']}**",
        f"- Missing folders: {metadata.validation['missing_folders'] or 'None'}",
        f"- Unexpected folders: {metadata.validation['unexpected_folders'] or 'None'}",
        f"- Empty folders: {metadata.validation['empty_folders'] or 'None'}",
        "",
        "## Corruption Check",
        "",
        f"- Total checked: {metadata.corruption['total_checked']}",
        f"- Valid: {metadata.corruption['valid_count']}",
        f"- Corrupted: {metadata.corruption['corrupted_count']}",
        "",
        "## Duplicate Detection",
        "",
        f"- Exact duplicate groups: {metadata.duplicates['exact_duplicate_groups']}",
        f"- Exact duplicates removed: {metadata.duplicates['exact_duplicate_count']}",
        f"- Near duplicate groups: {metadata.duplicates['near_duplicate_groups']}",
        f"- Near duplicates removed: {metadata.duplicates['near_duplicate_count']}",
        f"- Unique images retained: {metadata.duplicates['unique_images']}",
        "",
        "## Image Statistics",
        "",
    ])

    for class_name, stats in metadata.statistics.get("classes", {}).items():
        lines.extend([
            f"### {class_name}",
            "",
            f"- Count: {stats['count']}",
            f"- Width: min={stats['width']['min']}, max={stats['width']['max']}, "
            f"mean={stats['width']['mean']:.1f}",
            f"- Height: min={stats['height']['min']}, max={stats['height']['max']}, "
            f"mean={stats['height']['mean']:.1f}",
            f"- Mean RGB: [{stats['mean_rgb'][0]:.1f}, {stats['mean_rgb'][1]:.1f}, "
            f"{stats['mean_rgb'][2]:.1f}]",
            "",
        ])

    if metadata.validation.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in metadata.validation["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    if metadata.validation.get("errors"):
        lines.extend(["## Errors", ""])
        for error in metadata.validation["errors"]:
            lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines)


def save_audit_report(
    metadata: DatasetMetadata,
    reports_dir: Path,
) -> tuple[Path, Path]:
    """
    Save audit report as Markdown and JSON.

    Returns paths to (markdown_report, json_report).
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    md_path = reports_dir / f"audit_report_{timestamp}.md"
    json_path = reports_dir / f"audit_report_{timestamp}.json"

    md_path.write_text(generate_audit_report_markdown(metadata), encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # Also save latest copies
    latest_md = reports_dir / "audit_report_latest.md"
    latest_json = reports_dir / "audit_report_latest.json"
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    return md_path, json_path
