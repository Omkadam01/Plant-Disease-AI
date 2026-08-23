# PlantDiseaseAI — Tomato EfficientNet-B0 Model Report

**Crop:** Tomato  
**Model:** EfficientNet-B0  
**Report Generated:** 2026-07-23  
**Project:** PlantDiseaseAI v3

---

## 1. Training Summary

| Item | Value |
|------|-------|
| **Architecture** | EfficientNet-B0 (ImageNet pretrained) |
| **Planned Epochs** | 50 |
| **Actual Epochs Completed** | **47** (stopped early) |
| **Stop Reason** | Early stopping (patience = 8, monitor = val_loss) |
| **Best Epoch (lowest val loss)** | Epoch 39 (val loss = 0.000670) |
| **Best Validation Accuracy** | **100.00%** (achieved at epochs 38–39, 41–42) |
| **Total Training Time** | ~71.8 minutes (4,305 s) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Input Size | 256 × 256 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Loss | Weighted Cross Entropy |
| Scheduler | Cosine Annealing |
| Mixed Precision (AMP) | Enabled |
| Gradient Clipping | 1.0 |
| Early Stopping Patience | 8 |
| Class Weights Source | `datasets/tomato/split/train` (original, unbalanced) |

### Dataset Used for Training

| Split | Images | Notes |
|-------|--------|-------|
| Train | 17,230 | Balanced via oversampling (`datasets/tomato/balanced_train/`) |
| Validation | 2,716 | Original split only (`datasets/tomato/split/val/`) |
| Test | 2,713 | Held out — **not used during training** |

### Disease Classes (10)

| ID | Display Name |
|----|--------------|
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

---

## 2. Per-Epoch Training Report

Training ran from **Epoch 1 through Epoch 47** (early stopped before reaching the planned 50 epochs).

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | F1 Macro | Learning Rate | Time (s) |
|------:|-----------:|---------:|----------:|--------:|---------:|--------------:|---------:|
| 1 | 0.483327 | 0.154087 | 81.72% | 95.47% | 0.9472 | 0.000999 | 177.7 |
| 2 | 0.213545 | 0.140240 | 91.82% | 96.02% | 0.9567 | 0.000996 | 121.6 |
| 3 | 0.168628 | 0.084868 | 93.70% | 97.13% | 0.9704 | 0.000991 | 109.6 |
| 4 | 0.136446 | 0.060246 | 94.50% | 97.83% | 0.9774 | 0.000984 | 102.2 |
| 5 | 0.118685 | 0.055505 | 95.48% | 98.12% | 0.9800 | 0.000976 | 102.4 |
| 6 | 0.103382 | 0.053948 | 96.02% | 98.01% | 0.9810 | 0.000965 | 99.3 |
| 7 | 0.101896 | 0.049537 | 96.24% | 98.60% | 0.9821 | 0.000953 | 84.0 |
| 8 | 0.085862 | 0.032834 | 96.68% | 98.82% | 0.9870 | 0.000939 | 114.1 |
| 9 | 0.089744 | 0.029877 | 96.67% | 99.01% | 0.9874 | 0.000923 | 86.9 |
| 10 | 0.071350 | 0.056715 | 97.01% | 97.90% | 0.9764 | 0.000905 | 83.2 |
| 11 | 0.073429 | 0.024929 | 97.25% | 99.12% | 0.9890 | 0.000886 | 83.5 |
| 12 | 0.070371 | 0.030034 | 97.48% | 98.86% | 0.9885 | 0.000866 | 84.0 |
| 13 | 0.061699 | 0.019265 | 97.60% | 99.41% | 0.9918 | 0.000844 | 83.6 |
| 14 | 0.055422 | 0.017412 | 97.98% | 99.45% | 0.9937 | 0.000821 | 84.2 |
| 15 | 0.058235 | 0.013909 | 97.94% | 99.45% | 0.9942 | 0.000796 | 83.4 |
| 16 | 0.047060 | 0.027789 | 98.10% | 99.15% | 0.9899 | 0.000770 | 83.3 |
| 17 | 0.053480 | 0.018081 | 98.08% | 99.41% | 0.9933 | 0.000743 | 83.5 |
| 18 | 0.039381 | 0.024576 | 98.52% | 99.26% | 0.9925 | 0.000716 | 83.5 |
| 19 | 0.045419 | 0.011950 | 98.41% | 99.56% | 0.9952 | 0.000687 | 83.4 |
| 20 | 0.037085 | 0.017113 | 98.63% | 99.45% | 0.9926 | 0.000658 | 83.7 |
| 21 | 0.033807 | 0.018236 | 98.65% | 99.37% | 0.9916 | 0.000628 | 83.8 |
| 22 | 0.034785 | 0.025296 | 98.76% | 99.12% | 0.9907 | 0.000598 | 84.4 |
| 23 | 0.031533 | 0.016613 | 98.83% | 99.37% | 0.9926 | 0.000567 | 83.6 |
| 24 | 0.030019 | 0.010353 | 98.82% | 99.56% | 0.9950 | 0.000536 | 84.1 |
| 25 | 0.025470 | 0.008936 | 99.06% | 99.67% | 0.9966 | 0.000505 | 85.1 |
| 26 | 0.022396 | 0.010375 | 99.13% | 99.63% | 0.9954 | 0.000474 | 84.3 |
| 27 | 0.024399 | 0.008747 | 99.05% | 99.71% | 0.9970 | 0.000443 | 85.1 |
| 28 | 0.020047 | 0.005868 | 99.23% | 99.82% | 0.9978 | 0.000412 | 85.1 |
| 29 | 0.020182 | 0.004036 | 99.26% | 99.85% | 0.9984 | 0.000382 | 85.5 |
| 30 | 0.016888 | 0.018722 | 99.31% | 99.41% | 0.9948 | 0.000352 | 85.4 |
| 31 | 0.015631 | 0.006415 | 99.41% | 99.82% | 0.9979 | 0.000323 | 85.6 |
| 32 | 0.015081 | 0.005122 | 99.41% | 99.82% | 0.9984 | 0.000294 | 85.1 |
| 33 | 0.015302 | 0.008120 | 99.45% | 99.78% | 0.9978 | 0.000267 | 101.1 |
| 34 | 0.013529 | 0.001923 | 99.40% | 99.89% | 0.9983 | 0.000240 | 92.0 |
| 35 | 0.012239 | 0.005957 | 99.54% | 99.74% | 0.9964 | 0.000214 | 104.7 |
| 36 | 0.014941 | 0.002891 | 99.54% | 99.89% | 0.9989 | 0.000189 | 85.3 |
| 37 | 0.010368 | 0.002171 | 99.57% | 99.93% | 0.9992 | 0.000166 | 85.2 |
| 38 | 0.007950 | 0.001127 | 99.62% | **100.00%** | 1.0000 | 0.000144 | 86.4 |
| 39 | 0.007786 | **0.000670** | 99.66% | **100.00%** | 1.0000 | 0.000124 | 87.6 |
| 40 | 0.007497 | 0.001282 | 99.69% | 99.96% | 0.9996 | 0.000105 | 88.2 |
| 41 | 0.006020 | 0.001035 | 99.73% | **100.00%** | 1.0000 | 0.000087 | 88.1 |
| 42 | 0.006974 | 0.001042 | 99.76% | **100.00%** | 1.0000 | 0.000071 | 88.3 |
| 43 | 0.008191 | 0.001387 | 99.74% | 99.96% | 0.9998 | 0.000057 | 88.4 |
| 44 | 0.005956 | 0.001890 | 99.79% | 99.93% | 0.9987 | 0.000045 | 89.8 |
| 45 | 0.004927 | 0.001392 | 99.83% | 99.93% | 0.9985 | 0.000034 | 89.4 |
| 46 | 0.004389 | 0.001208 | 99.82% | 99.96% | 0.9996 | 0.000026 | 101.2 |
| 47 | 0.006127 | 0.000893 | 99.81% | 99.96% | 0.9996 | 0.000019 | 95.4 |

**Best checkpoint saved at:** Epoch 39 (lowest validation loss = 0.000670)

### Training Progress Notes

- Steady improvement through epochs 1–28; validation accuracy exceeded 99% from epoch 9 onward.
- Validation accuracy reached **100%** for the first time at **epoch 38**.
- Best validation loss (**0.000670**) achieved at **epoch 39** — this checkpoint was saved as `best_model.pth`.
- Early stopping triggered after epoch 47 (no val_loss improvement for 8 consecutive epochs).
- Final epoch train accuracy: **99.81%** | final epoch val accuracy: **99.96%**.

---

## 3. Saved Model Artifacts

| Artifact | Path |
|----------|------|
| Best Model | `weights/tomato/best_model.pth` |
| Last Model | `weights/tomato/last_model.pth` |
| Training History | `logs/tomato_training_history.json` |
| Training Log (CSV) | `logs/tomato_training_log.csv` |
| Training Curves | `weights/tomato/training_curves.png` |
| Validation Confusion Matrix | `weights/tomato/confusion_matrix.png` |
| Class Mapping | `datasets/tomato/reports/class_mapping.json` |

---

## 4. Testing Status

**Testing: COMPLETED**

Final evaluation was performed on the held-out test set (Step 6 — no retraining):

- **Test Directory:** `datasets/tomato/split/test/`
- **Test Images:** 2,713 (original images only — untouched held-out split)
- **Model Used:** `weights/tomato/best_model.pth`
- **Evaluation Date:** 2026-07-23
- **Evaluation Time:** 31.2 seconds

---

## 5. Test Set Evaluation Report

### Overall Metrics

| Metric | Score |
|--------|------:|
| **Test Accuracy (Top-1)** | **99.85%** |
| Test Loss | 0.008305 |
| Top-3 Accuracy | 100.00% |
| Macro Precision | 0.9984 |
| Macro Recall | 0.9980 |
| Macro F1 | 0.9982 |
| Weighted Precision | 0.9985 |
| Weighted Recall | 0.9985 |
| Weighted F1 | 0.9985 |
| Macro AUC | 0.99999 |
| Weighted AUC | 0.99999 |
| Average Confidence | 99.96% |
| Misclassifications | 4 / 2,713 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| Bacterial Spot | 1.00 | 1.00 | 1.00 | 318 |
| Early Blight | 1.00 | 0.99 | 1.00 | 150 |
| Healthy | 1.00 | 1.00 | 1.00 | 236 |
| Late Blight | 1.00 | 1.00 | 1.00 | 285 |
| Leaf Mold | 1.00 | 1.00 | 1.00 | 142 |
| Septoria Leaf Spot | 1.00 | 1.00 | 1.00 | 266 |
| Spider Mites (Two-spotted Spider Mite) | 0.99 | 1.00 | 1.00 | 250 |
| Target Spot | 1.00 | 0.99 | 1.00 | 210 |
| Tomato Mosaic Virus | 1.00 | 1.00 | 1.00 | 56 |
| Tomato Yellow Leaf Curl Virus | 1.00 | 1.00 | 1.00 | 800 |
| **Total** | | | | **2,713** |

- **Best-performing class:** Tomato Yellow Leaf Curl Virus (F1 = 1.00)
- **Worst-performing class:** Target Spot (F1 = 0.995)

### Classification Report

```
                                        precision    recall  f1-score   support

                        Bacterial Spot       1.00      1.00      1.00       318
                          Early Blight       1.00      0.99      1.00       150
                               Healthy       1.00      1.00      1.00       236
                           Late Blight       1.00      1.00      1.00       285
                             Leaf Mold       1.00      1.00      1.00       142
                    Septoria Leaf Spot       1.00      1.00      1.00       266
Spider Mites (Two-spotted Spider Mite)       0.99      1.00      1.00       250
                           Target Spot       1.00      0.99      1.00       210
                   Tomato Mosaic Virus       1.00      1.00      1.00        56
         Tomato Yellow Leaf Curl Virus       1.00      1.00      1.00       800

                              accuracy                           1.00      2713
                             macro avg       1.00      1.00      1.00      2713
                          weighted avg       1.00      1.00      1.00      2713
```

### Misclassifications (4 images)

| # | Ground Truth | Predicted | Confidence |
|--:|--------------|-----------|----------:|
| 1 | Early Blight | Late Blight | 98.69% |
| 2 | Late Blight | Healthy | 99.98% |
| 3 | Target Spot | Spider Mites | 99.90% |
| 4 | Target Spot | Spider Mites | 84.72% |

Misclassified images saved to: `reports/tomato_misclassified/`

### Validation Checks

| Check | Result |
|-------|--------|
| Every test image evaluated exactly once | Pass (2,713 / 2,713) |
| No missing predictions | Pass |
| No duplicate predictions | Pass |
| Test dataset unchanged | Pass |

### Evaluation Artifacts

| File | Location |
|------|----------|
| Summary (Markdown) | `reports/tomato_test_summary.md` |
| Summary (JSON) | `reports/tomato_test_summary.json` |
| Classification Report | `reports/tomato_test_classification_report.txt` |
| Predictions CSV | `reports/tomato_test_predictions.csv` |
| Misclassified Images | `reports/tomato_misclassified/` |
| Confusion Matrix Plot | `weights/tomato/test_confusion_matrix.png` |
| ROC Curves Plot | `weights/tomato/test_roc_curves.png` |
| Precision-Recall Curves | `weights/tomato/test_precision_recall_curves.png` |
| Confidence Distribution | `weights/tomato/confidence_distribution.png` |

---

## 6. Conclusion

| Stage | Status |
|-------|--------|
| Training | Completed (47/50 epochs, early stopped) |
| Validation | Best accuracy **100.00%** (epoch 39 checkpoint) |
| Testing | **Completed** — accuracy **99.85%** (4 misclassifications) |
| Production Ready | Yes (inference via desktop app) |

The Tomato EfficientNet-B0 model achieved near-perfect performance on the held-out test set. Validation reached 100% accuracy; test accuracy is 99.85% with all errors confined to visually similar disease pairs (Early/Late Blight, Target Spot/Spider Mites, Late Blight/Healthy).
