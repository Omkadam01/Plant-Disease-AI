"""Stratified dataset splitting."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from utils.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitEntry:
    """Single image split assignment."""

    source_path: Path
    class_name: str
    class_id: int
    split: str  # train | val | test
    filename: str


@dataclass
class SplitResult:
    """Dataset split outcome."""

    entries: list[SplitEntry] = field(default_factory=list)

    @property
    def train_count(self) -> int:
        return sum(1 for e in self.entries if e.split == "train")

    @property
    def val_count(self) -> int:
        return sum(1 for e in self.entries if e.split == "val")

    @property
    def test_count(self) -> int:
        return sum(1 for e in self.entries if e.split == "test")

    def by_split(self, split: str) -> list[SplitEntry]:
        return [e for e in self.entries if e.split == split]

    def class_distribution(self) -> dict[str, dict[str, int]]:
        dist: dict[str, dict[str, int]] = {}
        for entry in self.entries:
            dist.setdefault(entry.class_name, {"train": 0, "val": 0, "test": 0})
            dist[entry.class_name][entry.split] += 1
        return dist


def stratified_split(
    labeled_paths: dict[str, list[Path]],
    config: AppConfig,
) -> SplitResult:
    """
    Perform stratified train/validation/test split.

    Ratios are read from ``preprocessing.train_ratio``, etc.
    """
    seed = int(config.get("project.seed", 42))
    train_ratio = float(config.get("preprocessing.train_ratio", 0.70))
    val_ratio = float(config.get("preprocessing.val_ratio", 0.15))
    test_ratio = float(config.get("preprocessing.test_ratio", 0.15))
    stratify = bool(config.get("preprocessing.stratify", True))

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    class_to_id = {c.folder_name: c.id for c in config.class_configs}
    result = SplitResult()

    for class_name, paths in labeled_paths.items():
        if not paths:
            logger.warning("No images for class %s, skipping", class_name)
            continue

        class_id = class_to_id[class_name]
        labels = [class_id] * len(paths)

        if stratify and len(paths) >= 3:
            # First split: train vs (val + test)
            val_test_ratio = val_ratio + test_ratio
            train_paths, val_test_paths = train_test_split(
                paths,
                test_size=val_test_ratio,
                random_state=seed,
                stratify=labels,
            )

            # Second split: val vs test
            relative_test = test_ratio / val_test_ratio if val_test_ratio > 0 else 0.5
            val_labels = [class_id] * len(val_test_paths)
            if len(val_test_paths) >= 2 and stratify:
                val_paths, test_paths = train_test_split(
                    val_test_paths,
                    test_size=relative_test,
                    random_state=seed,
                    stratify=val_labels,
                )
            else:
                split_idx = max(1, int(len(val_test_paths) * (1 - relative_test)))
                val_paths = val_test_paths[:split_idx]
                test_paths = val_test_paths[split_idx:]
        else:
            n = len(paths)
            n_train = max(1, int(n * train_ratio))
            n_val = max(0, int(n * val_ratio))
            n_test = n - n_train - n_val
            if n_test < 0:
                n_test = 0
                n_val = n - n_train
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n)
            train_paths = [paths[i] for i in indices[:n_train]]
            val_paths = [paths[i] for i in indices[n_train : n_train + n_val]]
            test_paths = [paths[i] for i in indices[n_train + n_val :]]

        for path in train_paths:
            result.entries.append(
                SplitEntry(
                    source_path=path,
                    class_name=class_name,
                    class_id=class_id,
                    split="train",
                    filename=path.name,
                )
            )
        for path in val_paths:
            result.entries.append(
                SplitEntry(
                    source_path=path,
                    class_name=class_name,
                    class_id=class_id,
                    split="val",
                    filename=path.name,
                )
            )
        for path in test_paths:
            result.entries.append(
                SplitEntry(
                    source_path=path,
                    class_name=class_name,
                    class_id=class_id,
                    split="test",
                    filename=path.name,
                )
            )

    return result


def materialize_split(
    split_result: SplitResult,
    output_dirs: dict[str, Path],
    copy_files: bool = True,
) -> None:
    """
    Copy or symlink split files into train/val/test directory structure.

    Creates ``<split>/<class_name>/<filename>`` layout.
    """
    for split_name, base_dir in output_dirs.items():
        base_dir.mkdir(parents=True, exist_ok=True)

    for entry in split_result.entries:
        dest_dir = output_dirs[entry.split] / entry.class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / entry.filename

        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        if copy_files:
            shutil.copy2(entry.source_path, dest_path)
        else:
            try:
                os.symlink(entry.source_path.resolve(), dest_path)
            except OSError:
                shutil.copy2(entry.source_path, dest_path)


def save_split_manifest(split_result: SplitResult, output_path: Path) -> None:
    """Save split assignments to JSON manifest."""
    data = {
        "train_count": split_result.train_count,
        "val_count": split_result.val_count,
        "test_count": split_result.test_count,
        "class_distribution": split_result.class_distribution(),
        "entries": [
            {
                "source_path": str(e.source_path),
                "class_name": e.class_name,
                "class_id": e.class_id,
                "split": e.split,
                "filename": e.filename,
            }
            for e in split_result.entries
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
