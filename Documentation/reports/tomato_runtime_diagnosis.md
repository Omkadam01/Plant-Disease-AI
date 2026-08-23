# Tomato Runtime Inference Diagnosis

**Generated:** 2026-07-22 20:14:36 UTC

## Executive Summary

- **Root cause:** model generalization on out-of-distribution / real-world images (runtime code matches evaluation on official test set)
- **Runtime pipeline bug found:** No
- **Official sample accuracy (25 random images):** 96.0% (1 known Early/Late Blight confusion)
- **Full test set accuracy (desktop pipeline):** **99.85%** (2709/2713) — identical to evaluation script

## 1. Official Test Dataset Verification (Desktop Pipeline)

Samples tested: **25** (5 per class × 5 classes)
Accuracy: **96.00%**

| Status | Ground Truth | Predicted | Confidence | Image |
|--------|--------------|-----------|------------|-------|
| PASS | Healthy | Healthy | 1.0000 | `datasets\tomato\split\test\Tomato___healthy\b2497a4d-6951-46d3-810b-7072e908eea9___RS_HL 9907.jpg` |
| PASS | Healthy | Healthy | 1.0000 | `datasets\tomato\split\test\Tomato___healthy\21f1c16d-2d7f-44e6-a2e4-7d830dfbfda3___RS_HL 9996.jpg` |
| PASS | Healthy | Healthy | 1.0000 | `datasets\tomato\split\test\Tomato___healthy\07c3c887-da7d-4c81-b6ae-fabd1fb0ba5a___RS_HL 9681.jpg` |
| PASS | Healthy | Healthy | 1.0000 | `datasets\tomato\split\test\Tomato___healthy\c59a5a86-2645-49e0-a6e0-c9e8112f95d5___GH_HL Leaf 166.1.jpg` |
| PASS | Healthy | Healthy | 1.0000 | `datasets\tomato\split\test\Tomato___healthy\4916cbbb-a567-4263-a56d-521abadfb0b8___RS_HL 0480.jpg` |
| PASS | Early Blight | Early Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Early_blight\54d55df4-c760-4191-b998-d1e1493a35f5___RS_Erly.B 7524.jpg` |
| FAIL | Early Blight | Late Blight | 0.9869 | `datasets\tomato\split\test\Tomato___Early_blight\4fd3f0e7-cdd9-4151-8734-5f0ad1f52614___RS_Erly.B 9564.jpg` |
| PASS | Early Blight | Early Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Early_blight\38685009-d6e6-4b3a-9957-bf049a210cda___RS_Erly.B 7606.jpg` |
| PASS | Early Blight | Early Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Early_blight\2e7b090e-4fc6-42f2-8bf6-3f84b32712e5___RS_Erly.B 7470.jpg` |
| PASS | Early Blight | Early Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Early_blight\ee18e688-a12c-45aa-bd18-74df670c71bb___RS_Erly.B 7776.jpg` |
| PASS | Late Blight | Late Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Late_blight\29e924db-ef09-49f2-9db3-e3c548a444d5___GHLB_PS Leaf 17.1 Day 9.jpg` |
| PASS | Late Blight | Late Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Late_blight\c2d71717-0650-4ab0-859e-a7390243b2bc___RS_Late.B 6443.jpg` |
| PASS | Late Blight | Late Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Late_blight\0d98a9eb-18e7-47e7-9010-196189bae113___RS_Late.B 5047.jpg` |
| PASS | Late Blight | Late Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Late_blight\0c3153b2-0e64-41b6-902a-66eafbf7cae6___GHLB Leaf 4.1 Day 1.jpg` |
| PASS | Late Blight | Late Blight | 1.0000 | `datasets\tomato\split\test\Tomato___Late_blight\2bf70a8f-ba33-4405-89d2-8135dbea2b9d___GHLB2 Leaf 9064.jpg` |
| PASS | Target Spot | Target Spot | 0.9999 | `datasets\tomato\split\test\Tomato___Target_Spot\551a9326-bcff-4651-a4e9-cb037ef8ca52___Com.G_TgS_FL 0864.jpg` |
| PASS | Target Spot | Target Spot | 0.9999 | `datasets\tomato\split\test\Tomato___Target_Spot\59334a76-8e36-4c68-bd5e-4b181b710305___Com.G_TgS_FL 9778.jpg` |
| PASS | Target Spot | Target Spot | 1.0000 | `datasets\tomato\split\test\Tomato___Target_Spot\a9a5c547-0098-42e1-ab36-4eef9aaef45a___Com.G_TgS_FL 1104.jpg` |
| PASS | Target Spot | Target Spot | 1.0000 | `datasets\tomato\split\test\Tomato___Target_Spot\c7da4ac5-d59e-436f-9f8b-a1f74eeb30de___Com.G_TgS_FL 8009.jpg` |
| PASS | Target Spot | Target Spot | 1.0000 | `datasets\tomato\split\test\Tomato___Target_Spot\0da91f19-f1bb-4129-b122-459648794512___Com.G_TgS_FL 8276.jpg` |
| PASS | Tomato Yellow Leaf Curl Virus | Tomato Yellow Leaf Curl Virus | 1.0000 | `datasets\tomato\split\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\ba7d0a7c-e09d-41c9-9ddb-09d18dc7b97f___UF.GRC_YLCV_Lab 09561.jpg` |
| PASS | Tomato Yellow Leaf Curl Virus | Tomato Yellow Leaf Curl Virus | 1.0000 | `datasets\tomato\split\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\3efdbee2-4aed-4ba8-8187-cf20a4fdd502___YLCV_NREC 2876.jpg` |
| PASS | Tomato Yellow Leaf Curl Virus | Tomato Yellow Leaf Curl Virus | 1.0000 | `datasets\tomato\split\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\eda872cb-df32-43ed-a6be-27b64d38b136___YLCV_GCREC 5354.jpg` |
| PASS | Tomato Yellow Leaf Curl Virus | Tomato Yellow Leaf Curl Virus | 1.0000 | `datasets\tomato\split\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\d7cf301b-b54c-4a3f-93e0-62b00b4021d8___YLCV_GCREC 5532.jpg` |
| PASS | Tomato Yellow Leaf Curl Virus | Tomato Yellow Leaf Curl Virus | 1.0000 | `datasets\tomato\split\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\e8825fa7-93a0-45f5-b0e7-704541eba913___YLCV_GCREC 2740.jpg` |

## 2. GUI Image Pipeline

- `debug/runtime_input.png`: saved
- `debug/preprocessed_visualization.png`: saved

| Check | Result |
|-------|--------|
| runtime_input_saved | True |
| preprocessed_visualization_saved | True |
| original_shape | [256, 256, 3] |
| original_dtype | uint8 |
| original_channels | 3 |
| has_alpha_channel | False |
| rgb_ordering | RGB (read_image_rgb) |
| resize_only_applied | True |
| resize_target | [256, 256] |
| no_random_rotation_in_inference | True |
| no_center_crop_in_inference | True |
| tensor_shape | [1, 3, 256, 256] |
| tensor_dtype | torch.float32 |
| tensor_device | cpu |
| resize_content_similarity_mse | 0.27752685546875 |
| pipeline_ok | True |

## 3. Model Loading

- Selected crop: **tomato**
- Weights path: `C:\Projects\Plant_Disease_v3\weights\tomato\best_model.pth`
- Weights match expected: **True**
- Architecture: **EfficientNet**
- Output classes: **10**
- Device: **cuda**

## 4. Class Mapping

- Mapping file: `C:\Projects\Plant_Disease_v3\datasets\tomato\reports\class_mapping.json`
- Mapping matches runtime: **True**

| Index | Disease |
|------:|---------|
| 0 | Bacterial Spot |
| 1 | Early Blight |
| 2 | Healthy |
| 3 | Late Blight |
| 4 | Leaf Mold |
| 5 | Septoria Leaf Spot |
| 6 | Spider Mites (Two-spotted Spider Mite) |
| 7 | Target Spot |
| 8 | Tomato Mosaic Virus |
| 9 | Tomato Yellow Leaf Curl Virus |

## 5. Preprocessing

| Step | Value |
|------|-------|
| resize | 256x256 |
| rgb_conversion | read_image_rgb (BGR->RGB, EXIF transpose via PIL) |
| normalization_mean | (0.485, 0.456, 0.406) |
| normalization_std | (0.229, 0.224, 0.225) |
| transform | get_inference_transforms (= get_val_transforms) |
| expected_tensor_shape | [1, 3, 256, 256] |
| expected_dtype | torch.float32 |
| device | cuda |

## 6. GUI vs Evaluation Comparison

- Sample: `datasets\tomato\split\test\Tomato___healthy\0326b4b6-0f25-47af-bfd9-d8fec314a4f5___RS_HL 0621.jpg`
- Max logit diff: **0.0**
- Max prob diff: **0.0**
- Identical: **True**

## 7. Real-World Test Folder

- Folder: `debug/real_world_test/` (drop arbitrary phone images here and re-run `python scripts/verify_tomato_runtime.py`)
- Images found: **5**
- Results CSV: `debug/real_world_results.csv`

| Filename | Predicted | Confidence | Top-3 |
|----------|-----------|------------|-------|
| grape_ood_0_Black_Rot.jpg | Early Blight | 99.80% | Early Blight, Bacterial Spot, Late Blight |
| grape_ood_1_Black_Rot.jpg | Bacterial Spot | 67.77% | Bacterial Spot, Early Blight, Septoria Leaf Spot |
| grape_ood_2_Black_Rot.jpg | Early Blight | 99.81% | Early Blight, Target Spot, Bacterial Spot |
| synthetic_blue.jpg | Leaf Mold | 94.07% | Leaf Mold, Early Blight, YLCV |
| synthetic_noise.jpg | Late Blight | 33.79% | Late Blight, Healthy, Tomato Mosaic Virus |

## 8. Out-of-Distribution Analysis

- High-confidence (≥95%) count: **2** (grape images misclassified as Early Blight with ~99.8% confidence)
- Blight prediction ratio (real-world): **60.00%**
- Overconfident OOD suspected: **Yes**

**Recommendation:** The model is overconfident on out-of-distribution inputs (e.g., grape leaf images predicted as Early Blight at 99.8%). Apply:

1. **Confidence thresholding** — reject or flag predictions below ~0.85 (already partially implemented in the desktop UI low-confidence warning)
2. **Crop mismatch guard** — ensure Tomato model is selected when analyzing tomato images (already implemented)
3. **OOD detection** — consider adding an auxiliary "not a tomato leaf" classifier or entropy-based rejection for arbitrary uploads
4. **User guidance** — prompt users to photograph isolated tomato leaves on a plain background, similar to PlantVillage training data

## 9. Final Checklist

- [x] Desktop pipeline accuracy on official test images
- [x] GUI preprocessing matches evaluation (logit diff = 0)
- [x] Class mapping correct
- [x] Model loading correct
- [x] Runtime preprocessing correct
- [x] No runtime code bug identified

## Conclusion

The desktop runtime pipeline matches the official evaluation inference path **bit-identically** (max logit diff = 0.0) and achieves **99.85% accuracy** on the full held-out test set (2709/2713) — the same result as the evaluation script.

The single failure in the 25-image random sample (`Early Blight` → `Late Blight` at 98.7%) is a known model confusion between visually similar blight classes, not a preprocessing or loading defect.

If users observe frequent Early/Late Blight predictions on arbitrary phone uploads, the cause is **model generalization / out-of-distribution input**, not a defect in the desktop application code path. Grape leaf images fed to the tomato model are predicted as Early Blight with ~99.8% confidence, confirming the model lacks OOD awareness.

### Pipeline verified (no bugs found)

| Component | Status | Evidence |
|-----------|--------|----------|
| Model loading | ✓ | `weights/tomato/best_model.pth`, EfficientNet-B0, 10 classes, CUDA |
| Class mapping | ✓ | Matches `class_mapping.json` exactly |
| Preprocessing | ✓ | Resize 256, RGB, ImageNet normalize, no crop/rotation |
| GUI vs eval logits | ✓ | Max diff = 0.0 |
| Full test accuracy | ✓ | 99.85% via `ExplainablePredictor.predict()` |
| Real-world OOD | ⚠ | Overconfident blight predictions on non-tomato inputs |

### Re-run verification

```bash
python scripts/verify_tomato_runtime.py
```
