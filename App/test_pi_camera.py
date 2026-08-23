#!/usr/bin/env python3
"""Quick camera diagnostic for Raspberry Pi / Linux desktop setups."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for source_root in (
    PROJECT_ROOT / "App",
    PROJECT_ROOT / "Evaluation",
    PROJECT_ROOT / "Models",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from utils.platform import append_pi_system_packages, is_raspberry_pi

append_pi_system_packages()

from desktop_app.services.camera_service import create_camera_backend
from utils.config import apply_platform_overrides, load_config


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raspberry Pi detected: {is_raspberry_pi()}")

    config = apply_platform_overrides(load_config(project_root=PROJECT_ROOT))
    print(f"Camera backend preference: {config.get('desktop_app.camera.backend', 'auto')}")
    print(f"Camera index: {config.get('desktop_app.camera.index', 0)}")
    print(f"Device path: {config.get('desktop_app.camera.device_path')}")

    camera = create_camera_backend(config)
    if camera is None:
        print("FAIL: Could not open any camera backend.")
        print("Tips:")
        print("  - Enable camera: sudo raspi-config -> Interface Options -> Camera")
        print("  - USB webcam: check ls -l /dev/video*")
        print("  - Pi Camera: sudo apt install -y python3-picamera2")
        print("  - Add user to video group: sudo usermod -aG video $USER")
        return 1

    print(f"OK: Opened camera via {camera.name}")
    frame = camera.read_rgb()
    camera.close()

    if frame is None:
        print("FAIL: Camera opened but could not read a frame.")
        return 1

    print(f"OK: Captured frame shape={frame.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
