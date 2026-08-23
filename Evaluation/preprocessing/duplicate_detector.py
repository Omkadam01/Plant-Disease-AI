"""Exact and near-duplicate image detection."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image

from utils.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DuplicateGroup:
    """Group of duplicate or near-duplicate images."""

    group_id: int
    paths: list[Path]
    hash_value: str
    duplicate_type: str  # exact | near


@dataclass
class DuplicateDetectionResult:
    """Aggregate duplicate detection results."""

    exact_duplicates: list[DuplicateGroup] = field(default_factory=list)
    near_duplicates: list[DuplicateGroup] = field(default_factory=list)
    unique_paths: list[Path] = field(default_factory=list)

    @property
    def exact_duplicate_count(self) -> int:
        return sum(max(0, len(g.paths) - 1) for g in self.exact_duplicates)

    @property
    def near_duplicate_count(self) -> int:
        return sum(max(0, len(g.paths) - 1) for g in self.near_duplicates)


def _file_md5(path: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _perceptual_hash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


def detect_duplicates(
    image_paths: list[Path],
    config: AppConfig,
) -> DuplicateDetectionResult:
    """
    Detect exact (MD5) and near (perceptual hash) duplicates.

    Near duplicates use Hamming distance threshold from config.
    """
    threshold = int(config.get("preprocessing.near_duplicate_threshold", 5))
    result = DuplicateDetectionResult()

    # Exact duplicates
    md5_map: dict[str, list[Path]] = {}
    for path in image_paths:
        digest = _file_md5(path)
        md5_map.setdefault(digest, []).append(path)

    duplicate_paths: set[Path] = set()
    group_id = 0
    for digest, paths in md5_map.items():
        if len(paths) > 1:
            result.exact_duplicates.append(
                DuplicateGroup(group_id=group_id, paths=paths, hash_value=digest, duplicate_type="exact")
            )
            duplicate_paths.update(paths[1:])
            group_id += 1

    # Near duplicates on unique-by-md5 representatives
    representatives = [paths[0] for paths in md5_map.values()]
    phash_map: dict[str, list[tuple[Path, imagehash.ImageHash]]] = {}

    for path in representatives:
        try:
            phash = _perceptual_hash(path)
            phash_map.setdefault(str(phash), []).append((path, phash))
        except Exception as exc:
            logger.warning("Failed to hash %s: %s", path, exc)

    all_hashes: list[tuple[Path, imagehash.ImageHash]] = []
    for entries in phash_map.values():
        all_hashes.extend(entries)

    near_seen: set[Path] = set()
    group_id = 0
    for i, (path_a, hash_a) in enumerate(all_hashes):
        group_paths = [path_a]
        for path_b, hash_b in all_hashes[i + 1 :]:
            if path_b in near_seen or path_b in duplicate_paths:
                continue
            if hash_a - hash_b <= threshold:
                group_paths.append(path_b)
                near_seen.add(path_b)

        if len(group_paths) > 1:
            result.near_duplicates.append(
                DuplicateGroup(
                    group_id=group_id,
                    paths=group_paths,
                    hash_value=str(hash_a),
                    duplicate_type="near",
                )
            )
            group_id += 1

    near_dup_extras = set()
    for group in result.near_duplicates:
        near_dup_extras.update(group.paths[1:])

    result.unique_paths = [
        p for p in image_paths if p not in duplicate_paths and p not in near_dup_extras
    ]
    return result
