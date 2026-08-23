# Grad-CAM Explainability

PlantDiseaseAI integrates **Grad-CAM** (Gradient-weighted Class Activation Mapping) into the PyTorch inference pipeline to visualize which leaf regions influenced the model's Top-1 prediction.

## Overview

| Component | Location |
|-----------|----------|
| Grad-CAM core | `inference/gradcam.py` |
| Explainable predictor | `inference/explainable_predictor.py` |
| Demo / validation script | `scripts/run_gradcam_demo.py` |
| Unit tests | `tests/test_gradcam.py` |
| Output directory | `outputs/gradcam/<image_stem>/` |

## Requirements

- PyTorch backend only (uses `weights/grape/best_model.pth`)
- EfficientNet-B0 final conv layer: `model.features[-1][0]`
- Native Grad-CAM implementation (falls back if `pytorch-grad-cam` unavailable)

## Usage

### Python API

```python
from pathlib import Path
from utils.config import load_config
from inference.explainable_predictor import ExplainablePredictor

config = load_config(crop="grape", project_root=Path("."))
predictor = ExplainablePredictor(config)

result = predictor.predict_with_gradcam("path/to/leaf.jpg")

print(result.predicted_class)
print(result.confidence)
print(result.top_predictions)
print(result.original_output_path)
print(result.heatmap_path)
print(result.overlay_path)
```

### CLI Demo

```powershell
python scripts/run_gradcam_demo.py --crop grape --num-samples 3
```

Single image:

```powershell
python scripts/run_gradcam_demo.py --image datasets/grape/test/Black_Rot/sample.jpg
```

## Output Files

For each inference, the following files are saved:

```
outputs/gradcam/<image_stem>/
    original.jpg    # Copy of input image
    heatmap.png     # Grad-CAM colormap (JET)
    overlay.png     # Blended heatmap + original (alpha=0.45)
```

## Return Object

`ExplainablePredictionResult` extends `PredictionResult` with:

| Field | Description |
|-------|-------------|
| `predicted_class` | Top-1 class name |
| `confidence` | Top-1 confidence |
| `top_predictions` | Top-3 predictions with scores |
| `original_output_path` | Saved original image path |
| `heatmap_path` | Saved heatmap path |
| `overlay_path` | Saved overlay path |
| `heatmap` | In-memory normalized CAM array |

## How It Works

1. Run standard forward inference (no gradients) → Top-3 predictions
2. Enable gradients on the same input tensor
3. Backpropagate the score for the Top-1 class
4. Weight final conv feature maps by global-average-pooled gradients
5. Apply ReLU, normalize, resize to original image size
6. Blend with JET colormap onto the original leaf image

## Validation

The demo script verifies for each sample:

- Grad-CAM files are created
- Predicted class matches baseline `predict()` call
- Confidence scores are unchanged

Run tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/test_gradcam.py -v
```

## Example Output Structure

After running on test images:

```
outputs/gradcam/
    003d09ef-e16c-4e8a-badf-847d46cb3dc0___FAM_B.Rot 3184/
        original.jpg
        heatmap.png
        overlay.png
```

## Notes

- Grad-CAM is **not** applied to TorchScript or ONNX backends
- Use `ExplainablePredictor` (PyTorch) for explainability
- Standard `Predictor.predict()` remains unchanged for backward compatibility
