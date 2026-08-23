"""Image I/O and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageValidationResult:
    """Result of validating a single image file."""

    path: Path
    is_valid: bool
    width: int = 0
    height: int = 0
    channels: int = 0
    error: str | None = None


def collect_image_paths(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Recursively collect image paths with given extensions."""
    normalized = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in normalized:
            paths.append(path)
    return sorted(paths)


def validate_image(
    path: Path,
    min_size: int = 32,
    max_size: int = 4096,
) -> ImageValidationResult:
    """
    Validate image integrity and dimensions using OpenCV and PIL.

    Returns validation result with dimensions or error message.
    """
    try:
        # OpenCV decode check
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            return ImageValidationResult(path=path, is_valid=False, error="OpenCV failed to decode")

        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width = image.shape[:2]
            channels = image.shape[2]

        if width < min_size or height < min_size:
            return ImageValidationResult(
                path=path,
                is_valid=False,
                width=width,
                height=height,
                channels=channels,
                error=f"Image too small: {width}x{height}",
            )

        if width > max_size or height > max_size:
            return ImageValidationResult(
                path=path,
                is_valid=False,
                width=width,
                height=height,
                channels=channels,
                error=f"Image too large: {width}x{height}",
            )

        # PIL verify check
        with Image.open(path) as pil_img:
            pil_img.verify()

        return ImageValidationResult(
            path=path,
            is_valid=True,
            width=width,
            height=height,
            channels=channels,
        )
    except Exception as exc:
        return ImageValidationResult(path=path, is_valid=False, error=str(exc))


def read_image_bgr(path: Path) -> np.ndarray:
    """Read image as BGR numpy array."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def read_image_rgb(path: Path) -> np.ndarray:
    """Read image as RGB numpy array with EXIF orientation correction."""
    try:
        with Image.open(path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return np.asarray(pil_img)
    except Exception:
        bgr = read_image_bgr(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
