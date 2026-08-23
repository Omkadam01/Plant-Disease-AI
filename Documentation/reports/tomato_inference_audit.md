# Tomato Inference Pipeline Audit

**Date:** 2026-07-23  
**Scope:** End-to-end desktop inference vs training/evaluation pipeline  
**Status:** Aligned — no integration fixes required

---

## Executive Summary

The Tomato desktop inference pipeline is **correctly integrated** and produces **bit-identical** predictions to the training/evaluation pipeline on the same GPU device. No mismatches were found in weights, preprocessing, class mapping, softmax/argmax logic, or Top-3 output.

---

## Verified Components

| Check | Expected | Desktop App | Match |
|-------|----------|-------------|-------|
| Weights | `weights/tomato/best_model.pth` | `weights/tomato/best_model.pth` | Yes |
| Class mapping | `datasets/tomato/reports/class_mapping.json` | Same path via config | Yes |
| Image size | 256 (from checkpoint + config) | 256 | Yes |
| Resize | `A.Resize(256, 256)` | Same (`get_inference_transforms`) | Yes |
| RGB conversion | `read_image_rgb` (BGR→RGB) | Same in `_preprocess` | Yes |
| Normalization | ImageNet mean/std | `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` | Yes |
| Softmax | `torch.softmax(outputs, dim=1)` | Same | Yes |
| Top-1 | `torch.argmax(probs, dim=1)` | Same (updated explicitly) | Yes |
| Top-3 | `torch.topk(probs, k=3, dim=1)` | Same (updated explicitly) | Yes |
| Num classes | 10 | 10 from JSON `id_to_display` | Yes |

---

## Pipeline Comparison

### Training / Evaluation
```
PlantDiseaseDataset + get_val_transforms(256)
  → read_image_rgb → Resize → Normalize → ToTensorV2
  → model.forward()
  → torch.softmax → torch.argmax / torch.topk
```

### Desktop Application
```
Predictor._preprocess + get_inference_transforms(256)  [= get_val_transforms]
  → read_image_rgb → Resize → Normalize → ToTensorV2
  → model.forward()
  → torch.softmax → torch.argmax / torch.topk
  → ExplainablePredictor.predict_with_gradcam (Grad-CAM only, no prediction change)
```

Both paths use the **same transform function** (`get_inference_transforms` delegates to `get_val_transforms`).

---

## Quantitative Comparison (Healthy test sample)

| Metric | Evaluation | Desktop | Diff |
|--------|------------|---------|------|
| Predicted index | 2 | 2 | 0 |
| Predicted class | Healthy | Healthy | — |
| Top-3 indices | [2, 6, 3] | [2, 6, 3] | — |
| Max logit diff | — | — | **0.0** |
| Max prob diff | — | — | **0.0** |

---

## Debug Logging Added

Enable with environment variable:

```bash
set PLANT_DISEASE_INFERENCE_DEBUG=1
python scripts/run_app.py
```

Logs include:
- Selected crop (controller)
- Loaded model path and class mapping path
- Preprocessing parameters (resize, mean, std, tensor shape)
- Raw logits
- Softmax probabilities
- Predicted class index and mapped name
- Top-3 predictions with confidence

Debug logging is **disabled by default** and does not affect inference output.

---

## Code Changes (Integration Hardening)

| File | Change |
|------|--------|
| `inference/predictor.py` | Explicit `torch.argmax` + `torch.topk`; debug logging; class mapping path tracking; class-count mismatch warning |
| `desktop_app/services/inference_service.py` | Debug log on predict; expose `weights_path`, `class_names_path`, `crop_name` |
| `desktop_app/controllers/app_controller.py` | Debug log selected crop before inference |
| `scripts/audit_tomato_inference.py` | Automated side-by-side audit script |

---

## How to Re-run Audit

```bash
python scripts/audit_tomato_inference.py
```

JSON report: `reports/tomato_inference_audit.json`

---

## Conclusion

No inference integration bugs were found. The desktop application loads the correct Tomato checkpoint, applies identical preprocessing to training/evaluation, reads class names from the JSON mapping file (not hardcoded), and returns correct Top-3 predictions. Minor hardening was applied to use explicit PyTorch argmax/topk APIs and optional debug instrumentation.
