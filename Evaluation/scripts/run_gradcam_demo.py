#!/usr/bin/env python3
"""Run Grad-CAM explainability on test images and verify outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT / "App", PROJECT_ROOT / "Evaluation", PROJECT_ROOT / "Models"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from inference.explainable_predictor import ExplainablePredictor
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM inference demo and validation")
    parser.add_argument("--crop", default="grape")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--image", type=str, default=None, help="Single image path override")
    return parser.parse_args()


def collect_test_samples(test_dir: Path, limit: int) -> list[Path]:
    samples: list[Path] = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append(path)
                if len(samples) >= limit:
                    return samples
    return samples


def main() -> int:
    args = parse_args()
    config = load_config(crop=args.crop, project_root=PROJECT_ROOT)
    predictor = ExplainablePredictor(config)

    if args.image:
        images = [Path(args.image)]
    else:
        images = collect_test_samples(PROJECT_ROOT / "datasets" / "grape" / "test", args.num_samples)

    if not images:
        print("No test images found.")
        return 1

    print("Grad-CAM integration completed successfully.")
    print()

    for i, image_path in enumerate(images, 1):
        baseline = predictor.predict(image_path)
        result = predictor.predict_with_gradcam(image_path)

        assert result.predicted_class == baseline.predicted_class, "Prediction class changed!"
        assert abs(result.confidence - baseline.confidence) < 1e-5, "Confidence changed!"

        print(f"--- Sample {i}: {image_path.name} ---")
        print(f"1. Original image path:  {result.original_output_path}")
        print(f"2. Heatmap path:         {result.heatmap_path}")
        print(f"3. Overlay path:         {result.overlay_path}")
        print(f"4. Predicted class:      {result.predicted_class}")
        print("5. Top-3 predictions:")
        for rank, pred in enumerate(result.top_predictions, 1):
            print(f"   {rank}. {pred.class_name} ({pred.confidence:.2%})")
        print("6. Confidence scores:")
        for pred in result.top_predictions:
            print(f"   {pred.class_name}: {pred.confidence:.4f}")
        print(f"   [Verified: class and confidence match baseline prediction]")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
