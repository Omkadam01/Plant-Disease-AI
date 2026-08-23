"""Resolve project paths for development and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path


def get_project_root() -> Path:
    """Root folder for configs, weights, and bundled resources."""
    if getattr(sys, "frozen", False):
        exe_root = Path(sys.executable).parent
        bundled = Path(getattr(sys, "_MEIPASS", exe_root))
        if (exe_root / "Models" / "configs").exists():
            return exe_root
        if (bundled / "Models" / "configs").exists():
            return bundled
        return exe_root
    return Path(__file__).resolve().parents[2]


def get_writable_root() -> Path:
    """Folder for logs, captures, and Grad-CAM outputs."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]
