"""Cached multi-crop model loading for the desktop application."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from desktop_app.services.inference_service import InferenceService
from utils.config import AppConfig, apply_platform_overrides, load_config
from utils.platform import is_raspberry_pi

logger = logging.getLogger(__name__)


class ModelManager:
    """Load and cache inference services per crop without reloading duplicates."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._cache: dict[str, InferenceService] = {}
        self._single_model_mode = is_raspberry_pi()

    def is_cached(self, crop: str) -> bool:
        return crop in self._cache

    def get_cached(self, crop: str) -> InferenceService | None:
        return self._cache.get(crop)

    def load_config(self, crop: str) -> AppConfig:
        return apply_platform_overrides(load_config(crop=crop, project_root=self.project_root))

    def _unload_all(self) -> None:
        self._cache.clear()
        gc.collect()
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def get_or_load(self, crop: str) -> InferenceService:
        cached = self._cache.get(crop)
        if cached is not None:
            return cached

        # Pi has limited RAM — keep only one crop model loaded at a time.
        if self._single_model_mode and self._cache:
            logger.info("Pi low-memory mode: unloading cached models before loading %s", crop)
            self._unload_all()

        config = self.load_config(crop)
        service = InferenceService(config)
        service.load_model()
        self._cache[crop] = service
        return service
