"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.config import AppConfig


def setup_logging(
    config: AppConfig,
    log_name: str = "plantdisease",
    log_subdir: str = "general",
) -> logging.Logger:
    """
    Configure application-wide logging.

    Returns the named logger instance.
    """
    log_level = getattr(logging, str(config.get("logging.level", "INFO")).upper(), logging.INFO)
    log_format = config.get("logging.format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    date_format = config.get("logging.date_format", "%Y-%m-%d %H:%M:%S")

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    if config.get("logging.console_enabled", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    if config.get("logging.file_enabled", True):
        log_dir = config.path("paths.logs") / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / f"{log_name}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger
