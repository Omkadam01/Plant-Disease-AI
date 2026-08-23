"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClassConfig:
    """Single disease class configuration."""

    id: int
    folder_name: str
    display_name: str
    slug: str


@dataclass
class AppConfig:
    """Merged application configuration."""

    raw: dict[str, Any] = field(default_factory=dict)
    project_root: Path = field(default_factory=lambda: Path("."))

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation accessor, e.g. ``training.batch_size``."""
        keys = key.split(".")
        value: Any = self.raw
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    @property
    def crop_name(self) -> str:
        return str(self.get("crop.name", "grape"))

    @property
    def num_classes(self) -> int:
        classes = self.get("classes", [])
        return len(classes) if classes else 0

    @property
    def class_configs(self) -> list[ClassConfig]:
        return [
            ClassConfig(
                id=c["id"],
                folder_name=c["folder_name"],
                display_name=c["display_name"],
                slug=c["slug"],
            )
            for c in self.get("classes", [])
        ]

    @property
    def class_names(self) -> list[str]:
        return [c.display_name for c in self.class_configs]

    @property
    def class_folders(self) -> list[str]:
        return [c.folder_name for c in self.class_configs]

    @property
    def class_slug_map(self) -> dict[str, str]:
        return {c.display_name: c.slug for c in self.class_configs}

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to project root."""
        rel = self.get(key)
        if rel is None:
            raise KeyError(f"Path key not found in config: {key}")
        return (self.project_root / rel).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def load_config(
    crop: str = "grape",
    base_config_path: Path | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    """
    Load and merge base + crop-specific YAML configuration.

    Parameters
    ----------
    crop:
        Crop identifier (e.g. ``grape``).
    base_config_path:
        Optional override for base config location.
    project_root:
        Project root directory. Defaults to cwd.
    """
    root = (project_root or Path.cwd()).resolve()
    config_root = root / "Models" / "configs"
    base_path = base_config_path or config_root / "base.yaml"
    crop_path = config_root / "crops" / f"{crop}.yaml"

    base_cfg = _load_yaml(base_path)
    crop_cfg = _load_yaml(crop_path)
    merged = _deep_merge(base_cfg, crop_cfg)

    return AppConfig(raw=merged, project_root=root)


def apply_platform_overrides(config: AppConfig) -> AppConfig:
    """Merge Raspberry Pi (or other platform) overrides when detected."""
    from utils.platform import is_raspberry_pi

    if not is_raspberry_pi():
        return config

    platform_path = (
        config.project_root / "Models" / "configs" / "platform" / "raspberry_pi.yaml"
    )
    if not platform_path.exists():
        return config

    platform_cfg = _load_yaml(platform_path)
    merged = _deep_merge(config.raw, platform_cfg)
    return AppConfig(raw=merged, project_root=config.project_root)
