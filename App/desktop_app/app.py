"""PlantDiseaseAI desktop application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from desktop_app.ui.main_window import MainWindow
from utils.config import load_config, apply_platform_overrides
from utils.logging import setup_logging
from utils.platform import is_raspberry_pi
from utils.runtime_paths import get_project_root


def run_app(crop: str = "grape") -> int:
    """Launch the PlantDiseaseAI v1.0 desktop application."""
    config = apply_platform_overrides(
        load_config(crop=crop, project_root=get_project_root())
    )
    setup_logging(config, log_name="desktop_app", log_subdir="desktop")

    if is_raspberry_pi():
        import logging

        logging.getLogger(__name__).info(
            "Raspberry Pi mode enabled | device=%s gradcam=%s camera=%s",
            config.get("device.preferred", "cpu"),
            config.get("inference.enable_gradcam", True),
            config.get("desktop_app.camera.backend", "auto"),
        )

    app = QApplication(sys.argv)
    app.setApplicationName("PlantDiseaseAI")
    app.setOrganizationName("PlantDiseaseAI")

    window = MainWindow(config)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
