"""Label and folder structure validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from utils.config import AppConfig


@dataclass
class LabelValidationResult:
    """Outcome of label/folder validation."""

    is_valid: bool
    found_folders: list[str] = field(default_factory=list)
    missing_folders: list[str] = field(default_factory=list)
    unexpected_folders: list[str] = field(default_factory=list)
    empty_folders: list[str] = field(default_factory=list)
    alias_mappings: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_labels_and_folders(raw_dir: Path, config: AppConfig) -> LabelValidationResult:
    """
    Validate raw dataset folder structure against configured classes.

    Supports PlantVillage naming via ``class_aliases`` in crop config.
    """
    result = LabelValidationResult(is_valid=True)

    if not raw_dir.exists():
        result.is_valid = False
        result.errors.append(f"Raw dataset directory does not exist: {raw_dir}")
        return result

    expected_folders = {c.folder_name for c in config.class_configs}
    aliases: dict[str, str] = dict(config.get("class_aliases", {}))

    subdirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    found_names = {d.name for d in subdirs}
    result.found_folders = sorted(found_names)

    # Map aliases to canonical folder names
    for folder in subdirs:
        canonical = aliases.get(folder.name, folder.name)
        if canonical in expected_folders and folder.name != canonical:
            result.alias_mappings[folder.name] = canonical

    mapped_found = set()
    for name in found_names:
        mapped_found.add(aliases.get(name, name))

    result.missing_folders = sorted(expected_folders - mapped_found)
    result.unexpected_folders = sorted(mapped_found - expected_folders)

    extensions = tuple(config.get("preprocessing.image_extensions", [".jpg", ".jpeg", ".png"]))
    normalized_ext = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    for folder in subdirs:
        canonical = aliases.get(folder.name, folder.name)
        if canonical not in expected_folders:
            continue
        images = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in normalized_ext
        ]
        if not images:
            result.empty_folders.append(folder.name)
            result.warnings.append(f"Folder '{folder.name}' contains no images")

    if result.missing_folders:
        result.is_valid = False
        result.errors.append(f"Missing required class folders: {result.missing_folders}")

    if result.unexpected_folders:
        result.warnings.append(f"Unexpected folders found: {result.unexpected_folders}")

    return result
