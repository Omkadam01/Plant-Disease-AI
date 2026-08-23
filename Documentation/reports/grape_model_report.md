# PlantDiseaseAI — Grape EfficientNet-B0 Model Report

**Crop:** Grape  
**Model:** EfficientNet-B0  
**Report Generated:** 2026-07-23  
**Project:** PlantDiseaseAI v3

---

## 1. Training Summary

| Item | Value |
|------|-------|
| **Architecture** | EfficientNet-B0 (ImageNet pretrained) |
| **Planned Epochs** | 50 |
| **Actual Epochs Completed** | **17** (stopped early) |
| **Stop Reason** | Early stopping (patience = 10, monitor = val_loss) |
| **Best Epoch (lowest val loss)** | Epoch 16 (val loss = 0.000447) |
| **Best Validation Accuracy** | **100.00%** (achieved at epochs 7, 11, 12, 13, 16) |
| **Total Training Time** | ~21.2 minutes (1274 s) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Input Size | 224 × 224 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Loss | Weighted Cross Entropy |
| Scheduler | Cosine Annealing (warmup 2 epochs) |
| Mixed Precision (AMP) | Enabled |
| Gradient Clipping | 1.0 |
| Early Stopping Patience | 10 |
| Class Weights Source | `datasets/grape/train (inverse-frequency computed at training time)` |

### Dataset Used for Training

| Split | Images | Notes |
|-------|--------|-------|
| Train | 3,417 | Includes 578 augmented Healthy images |
| Validation | 608 | Original images only (`datasets/grape/val/`) |
| Test | 610 | Held out — **not used during training** |

### Disease Classes (4)

| ID | Display Name |
|----|--------------|
| 0 | Black Rot |
| 1 | Esca (Black Measles) |
| 2 | Leaf Blight (Isariopsis Leaf Spot) |
| 3 | Healthy |

---

## 2. Per-Epoch Training Report

Training ran from **Epoch 1 through Epoch 17** (early stopped before reaching the planned 50 epochs).

> **Note:** Macro F1 was not logged during grape training (`weights/grape/training_history.json` does not contain F1 values).

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | F1 Macro | Learning Rate | Time (s) |
|------:|-----------:|---------:|----------:|--------:|---------:|--------------:|---------:|
| 1 | 0.254912 | 0.011213 | 90.54% | 99.67% | Not Available | 0.001000 | 119.3 |
| 2 | 0.179883 | 0.123931 | 93.25% | 95.23% | Not Available | 0.001000 | 86.5 |
| 3 | 0.170132 | 0.089353 | 93.84% | 97.04% | Not Available | 0.000999 | 79.6 |
| 4 | 0.147392 | 0.005229 | 94.10% | 99.84% | Not Available | 0.000996 | 97.9 |
| 5 | 0.093919 | 0.016757 | 96.11% | 99.18% | Not Available | 0.000990 | 90.2 |
| 6 | 0.108427 | 0.008278 | 96.76% | 99.67% | Not Available | 0.000983 | 66.1 |
| 7 | 0.066809 | 0.000475 | 97.85% | **100.00%** | Not Available | 0.000974 | 65.7 |
| 8 | 0.056311 | 0.016540 | 97.94% | 99.34% | Not Available | 0.000962 | 65.9 |
| 9 | 0.062393 | 0.012834 | 98.02% | 99.67% | Not Available | 0.000949 | 65.6 |
| 10 | 0.044648 | 0.003659 | 98.38% | 99.84% | Not Available | 0.000934 | 65.7 |
| 11 | 0.048007 | 0.000820 | 98.64% | **100.00%** | Not Available | 0.000917 | 65.6 |
| 12 | 0.040509 | 0.000527 | 98.56% | **100.00%** | Not Available | 0.000898 | 65.8 |
| 13 | 0.042819 | 0.003011 | 98.58% | **100.00%** | Not Available | 0.000877 | 68.5 |
| 14 | 0.038506 | 0.003349 | 98.79% | 99.84% | Not Available | 0.000855 | 68.7 |
| 15 | 0.030154 | 0.006402 | 99.00% | 99.67% | Not Available | 0.000831 | 69.8 |
| 16 | 0.027215 | **0.000447** | 99.03% | **100.00%** | Not Available | 0.000806 | 66.3 |
| 17 | 0.026547 | 0.003751 | 98.94% | 99.84% | Not Available | 0.000780 | 66.5 |

**Best checkpoint saved at:** Epoch 16 (lowest validation loss = 0.000447)

### Training Progress Notes

- Rapid convergence after epoch 4; validation accuracy exceeded 99% from epoch 4 onward.
- Validation accuracy reached **100%** for the first time at **epoch 7**.
- Best validation loss (**0.000447**) achieved at **epoch 16** — saved as `best_model.pth` (checkpoint epoch index 15).
- Training stopped after epoch 17 (early stopping, patience = 10, monitor = val_loss).
- Final epoch train accuracy: **98.94%** | final epoch val accuracy: **99.84%**.

---

## 3. Saved Model Artifacts

| Artifact | Path |
|----------|------|
| Best Model | `weights/grape/best_model.pth` |
| Last Model | `weights/grape/last_model.pth` |
| Training History | `weights/grape/training_history.json` |
| Training Log (CSV) | `Not Available` |
| TensorBoard Logs | `logs/tensorboard/grape_efficientnet_b0_20260722_165056` |
| Training Curves | `Not Available` |
| Validation Confusion Matrix | `Not Available` |
| Class Mapping | `datasets/grape/reports/class_mapping.json` |

---

## 4. Testing Status

**Testing: COMPLETED**

Evaluation was performed automatically after training on the held-out test set:

- **Test Directory:** `datasets/grape/test/`
- **Test Images:** 610 (original images only — no augmented data)
- **Model Used:** `weights/grape/best_model.pth`
- **Evaluation Date:** 2026-07-22
- **Evaluation Time:** 31 seconds

---

## 5. Test Set Evaluation Report

### Overall Metrics

| Metric | Score |
|--------|------:|
| **Test Accuracy (Top-1)** | **100.00%** |
| Test Loss | Not Available |
| Top-3 Accuracy | 100.00% |
| Macro Precision | 1.0000 |
| Macro Recall | 1.0000 |
| Macro F1 | 1.0000 |
| Weighted Precision | 1.0000 |
| Weighted Recall | 1.0000 |
| Weighted F1 | 1.0000 |
| Macro AUC | 1.00000 |
| Weighted AUC | Not Available |
| Average Confidence | 99.99% |
| Misclassifications | 0 / 610 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| Black Rot | 1.00 | 1.00 | 1.00 | 177 |
| Esca (Black Measles) | 1.00 | 1.00 | 1.00 | 207 |
| Leaf Blight (Isariopsis Leaf Spot) | 1.00 | 1.00 | 1.00 | 162 |
| Healthy | 1.00 | 1.00 | 1.00 | 64 |
| **Total** | | | | **610** |

- **Best-performing class:** All classes tied (F1 = 1.00)
- **Worst-performing class:** None (zero misclassifications)

### Classification Report

```
precision    recall  f1-score   support

                         Black Rot       1.00      1.00      1.00       177
              Esca (Black Measles)       1.00      1.00      1.00       207
Leaf Blight (Isariopsis Leaf Spot)       1.00      1.00      1.00       162
                           Healthy       1.00      1.00      1.00        64

                          accuracy                           1.00       610
                         macro avg       1.00      1.00      1.00       610
                      weighted avg       1.00      1.00      1.00       610
```

### Misclassifications (0 images)

No misclassifications recorded in `evaluation/grape/predictions.json`.

### Validation Checks

| Check | Result |
|-------|--------|
| Every test image evaluated exactly once | Pass (610 / 610) |
| No missing predictions | Pass |
| No duplicate predictions | Pass |
| Test dataset unchanged | Pass |

### Evaluation Artifacts

| File | Location |
|------|----------|
| Metrics Json | `evaluation/grape/metrics.json` |
| Classification Report | `evaluation/grape/classification_report.txt` |
| Predictions Json | `evaluation/grape/predictions.json` |
| Confusion Matrix Plot | `evaluation/grape/confusion_matrix.png` |
| Roc Curves Plot | `evaluation/grape/roc_curves.png` |
| Precision Recall Curves | `evaluation/grape/precision_recall_curves.png` |

---

## 6. Conclusion

| Stage | Status |
|-------|--------|
| Training | Completed (17/50 epochs, early stopped) |
| Validation | Best accuracy **100.00%** (epoch 16 checkpoint) |
| Testing | **Completed** — accuracy **100.00%** (0 misclassifications) |
| Production Ready | Yes (inference via desktop app) |

### Key Strengths

- Perfect test-set accuracy (610/610) on held-out PlantVillage grape images.
- Fast convergence (17 epochs, ~21 minutes training time).
- Exported TorchScript and ONNX models verified during training pipeline.

### Known Limitations

- Test loss and weighted AUC were not persisted in evaluation artifacts.
- Per-epoch macro F1 was not logged during training.
- Training curve and validation confusion matrix plots were not saved under `weights/grape/`.
- Real-world / out-of-distribution generalization has not been formally evaluated (same limitation as tomato on arbitrary phone uploads).

---

## 7. Export

| Format | Location |
|--------|----------|
| TorchScript | `exports/grape/grape_disease.torchscript.pt` |
| ONNX | `exports/grape/grape_disease.onnx` |
| PyTorch Copy | `exports/grape/grape_disease.pth` |
| Export Metadata | `exports/grape/export_metadata.json` |

