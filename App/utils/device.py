"""Device selection utilities."""

from __future__ import annotations

import logging

import torch

from utils.config import AppConfig
from utils.platform import is_raspberry_pi

logger = logging.getLogger(__name__)


def get_device(config: AppConfig | None = None) -> torch.device:
    """
    Select the best available compute device.

    Honors ``device.auto_select`` and ``device.preferred`` from config.
    On Raspberry Pi, defaults to CPU unless explicitly overridden.
    """
    preferred = "cuda"
    auto_select = True

    if config is not None:
        preferred = str(config.get("device.preferred", "cuda")).lower()
        auto_select = bool(config.get("device.auto_select", True))

    if is_raspberry_pi() and preferred == "cuda" and not torch.cuda.is_available():
        preferred = "cpu"

    if not auto_select:
        return torch.device(preferred)

    if preferred == "cpu":
        logger.info("Using CPU device")
        return torch.device("cpu")

    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device

    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS device")
        return torch.device("mps")

    if torch.cuda.is_available() and not is_raspberry_pi():
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    logger.info("Using CPU device")
    return torch.device("cpu")
