#!/usr/bin/env python3
"""Launch PlantDiseaseAI desktop application."""

from __future__ import annotations

import argparse
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

from utils.platform import append_pi_system_packages, configure_pi_low_memory

append_pi_system_packages()
configure_pi_low_memory()

from desktop_app.app import run_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch PlantDiseaseAI desktop application")
    parser.add_argument("--crop", type=str, default="grape", help="Crop identifier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_app(crop=args.crop)


if __name__ == "__main__":
    sys.exit(main())
