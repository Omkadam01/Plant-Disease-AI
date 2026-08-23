# Tomato Desktop Integration Report

**Step:** 7 — Desktop Application Integration  
**Date:** 2026-07-23  
**Status:** Completed

---

## Summary

The trained Tomato EfficientNet-B0 model is integrated into the existing PlantDiseaseAI desktop application alongside Grape. Users can switch crops from the dropdown without restarting the app. The unified inference, Grad-CAM, and Groq explanation pipelines are reused with no duplicated logic.

---

## Supported Crops

| Crop | Weights | Classes | Input Size | Class Mapping |
|------|---------|---------|------------|---------------|
| Grape | `weights/grape/best_model.pth` | 4 | 224 | `datasets/grape/reports/class_mapping.json` |
| Tomato | `weights/tomato/best_model.pth` | 10 | 256 | `datasets/tomato/reports/class_mapping.json` |

---

## Files Modified

| File | Change |
|------|--------|
| `desktop_app/widgets/left_panel.py` | Added Tomato to crop dropdown; translation keys for both crops |
| `desktop_app/controllers/app_controller.py` | Dynamic crop switching, model cache integration, Groq crop slug |
| `desktop_app/ui/main_window.py` | Initial crop sync, clear state on crop change |
| `desktop_app/services/workers.py` | ModelLoadWorker accepts crop + ModelManager |
| `desktop_app/services/groq_service.py` | Crop-aware prompts, system role, cache key includes crop |
| `desktop_app/widgets/status_panel.py` | Added `clear()` for crop switch reset |
| `inference/predictor.py` | Config-driven defaults (no hardcoded grape paths) |
| `inference/explainable_predictor.py` | Grad-CAM output per crop (`outputs/gradcam/<crop>/`) |
| `configs/crops/grape.yaml` | Added inference metadata (image_size, num_classes, mean, std) |
| `configs/crops/tomato.yaml` | Added inference metadata (image_size, num_classes, mean, std) |
| `resources/translations/en.json` | Added `left_panel.crop_tomato` |
| `resources/translations/hi.json` | Added `left_panel.crop_tomato` |
| `resources/translations/mr.json` | Added `left_panel.crop_tomato` |
| `scripts/test_groq_integration.py` | Updated for `crop_slug` parameter |

## Files Created

| File | Purpose |
|------|---------|
| `desktop_app/services/model_manager.py` | Per-crop model cache; load-once, reuse |
| `scripts/validate_crop_integration.py` | Headless validation for both crops |
| `reports/tomato_integration.md` | This report |

---

## Components Updated

### 1. Crop Selection
- Dropdown options: **Grape**, **Tomato**
- Selection triggers automatic model switch without app restart
- UI layout unchanged — only dropdown content updated

### 2. Dynamic Model Loading
- `ModelManager` caches loaded `InferenceService` instances per crop
- Switching back to a previously loaded crop reuses the cached model (no reload)
- First load for each crop runs in a background thread via `ModelLoadWorker`

### 3. Configuration
- Both `configs/crops/grape.yaml` and `configs/crops/tomato.yaml` define:
  - Model path, class mapping path, image size, num_classes, ImageNet mean/std
- Class names loaded from JSON at runtime — never hardcoded in the app

### 4. Inference Pipeline
- Single path: `InferenceService` → `ExplainablePredictor` → `Predictor`
- Top-3 predictions and confidence scores work for both crops
- Tomato uses 256×256 input; Grape uses 224×224 (from config/checkpoint)

### 5. Grad-CAM
- Reuses existing `generate_gradcam()` implementation
- Outputs saved under `outputs/gradcam/grape/` or `outputs/gradcam/tomato/`
- Original, heatmap, and overlay images generated for both crops

### 6. Groq AI Explanation
- Prompts adapt to crop (viticulture vs tomato horticulture)
- Cache key includes crop slug to prevent cross-crop stale responses
- Supports English, Hindi, and Marathi

---

## Validation Results

Automated validation via `python scripts/validate_crop_integration.py`:

| Check | Grape | Tomato |
|-------|-------|--------|
| Model loaded | Pass | Pass |
| Class count | 4 | 10 |
| Top-3 predictions | Pass | Pass |
| Confidence in [0, 1] | Pass | Pass |
| Grad-CAM overlay | Pass | Pass |
| Grad-CAM heatmap | Pass | Pass |
| Model cache reuse | Pass | — |

**Sample predictions:**
- Grape (Black Rot test image): `Black Rot` @ 100.00%
- Tomato (Healthy test image): `Healthy` @ 100.00%

**Backward compatibility:** Grape pipeline unchanged; grape config, weights, and inference path preserved.

---

## Warnings

1. **Groq API key required** for AI explanations — set `GROQ_API_KEY` in `.env`. Without it, predictions and Grad-CAM still work; the explanation panel shows unavailable.
2. **First crop switch** loads the model in background (~5–10 s depending on GPU). Subsequent switches to a cached crop are instant.
3. **Legacy desktop files** (`desktop_app/main_window.py`, `desktop_app/widgets.py`) remain in the repo but are not used by the active app entry point (`desktop_app/app.py`).

---

## How to Run

```bash
python scripts/run_app.py
```

Optional default crop at launch:

```bash
python scripts/run_app.py --crop grape
python scripts/run_app.py --crop tomato
```

Validate integration without GUI:

```bash
python scripts/validate_crop_integration.py
```

---

## Conclusion

The desktop application now supports Grape and Tomato through a single unified pipeline. Users can select a crop, upload or capture an image, receive Top-3 predictions with confidence, view Grad-CAM overlays, and get crop-specific AI explanations — all without retraining or modifying datasets.
