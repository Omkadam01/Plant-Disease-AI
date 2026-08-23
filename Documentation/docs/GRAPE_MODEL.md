# PlantDiseaseAI — EfficientNet-B0 Model Report

**Crop:** Grape  
**Model:** EfficientNet-B0  
**Report Generated:** 2026-07-22  
**Project:** PlantDiseaseAI v1.0.0

---

## 1. Training Summary

| Item | Value |
|------|-------|
| **Architecture** | EfficientNet-B0 (ImageNet pretrained) |
| **Planned Epochs** | 50 |
| **Actual Epochs Completed** | **17** (stopped early) |
| **Stop Reason** | Early stopping (patience = 10, monitor = val_loss) |
| **Best Epoch (lowest val loss)** | Epoch 7 (val loss = 0.000475) |
| **Best Validation Accuracy** | **100.00%** (achieved at epochs 7, 11, 12, 13, 16) |
| **Device** | NVIDIA GeForce RTX 4050 Laptop GPU (CUDA) |
| **Total Training Time** | ~22 minutes |

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
| Random Seed | 42 |

### Dataset Used for Training

| Split | Images | Notes |
|-------|--------|-------|
| Train | 3,417 | Includes 578 augmented Healthy images |
| Validation | 608 | Original images only (no augmentation) |
| Test | 610 | Original images only (no augmentation) |

---

## 2. Per-Epoch Training Report

Training ran from **Epoch 1 through Epoch 17** (early stopped before reaching the planned 50 epochs).

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Learning Rate | Time (s) |
|------:|-----------:|---------:|----------:|--------:|--------------:|---------:|
| 1 | 0.2549 | 0.0112 | 90.54% | 99.67% | 0.001000 | 119.3 |
| 2 | 0.1799 | 0.1239 | 93.25% | 95.23% | 0.001000 | 86.5 |
| 3 | 0.1701 | 0.0894 | 93.84% | 97.04% | 0.000999 | 79.6 |
| 4 | 0.1474 | 0.0052 | 94.10% | 99.84% | 0.000996 | 97.9 |
| 5 | 0.0939 | 0.0168 | 96.11% | 99.18% | 0.000990 | 90.2 |
| 6 | 0.1084 | 0.0083 | 96.76% | 99.67% | 0.000983 | 66.1 |
| 7 | 0.0668 | **0.0005** | 97.85% | **100.00%** | 0.000974 | 65.7 |
| 8 | 0.0563 | 0.0165 | 97.94% | 99.34% | 0.000962 | 65.9 |
| 9 | 0.0624 | 0.0128 | 98.02% | 99.67% | 0.000949 | 65.6 |
| 10 | 0.0446 | 0.0037 | 98.38% | 99.84% | 0.000934 | 65.7 |
| 11 | 0.0480 | 0.0008 | 98.64% | **100.00%** | 0.000917 | 65.6 |
| 12 | 0.0405 | 0.0005 | 98.56% | **100.00%** | 0.000898 | 65.8 |
| 13 | 0.0428 | 0.0030 | 98.58% | **100.00%** | 0.000877 | 68.5 |
| 14 | 0.0385 | 0.0033 | 98.79% | 99.84% | 0.000855 | 68.7 |
| 15 | 0.0302 | 0.0064 | 99.00% | 99.67% | 0.000831 | 69.8 |
| 16 | 0.0272 | 0.0004 | 99.03% | **100.00%** | 0.000806 | 66.3 |
| 17 | 0.0265 | 0.0038 | 98.94% | 99.84% | 0.000780 | 66.5 |

**Best checkpoint saved at:** Epoch 7 (lowest validation loss = 0.000475)

### Training Progress Notes

- Rapid convergence after epoch 4; validation accuracy reached 99%+ consistently from epoch 4 onward.
- Validation accuracy hit **100%** for the first time at **epoch 7**.
- No improvement in validation loss for 10 consecutive epochs after epoch 7, triggering early stopping at epoch 17.
- Final epoch train accuracy: **98.94%** | final epoch val accuracy: **99.84%**.

---

## 3. Saved Model Artifacts

| Artifact | Path |
|----------|------|
| Best Model | `weights/grape/best_model.pth` |
| Last Model | `weights/grape/last_model.pth` |
| Training History | `weights/grape/training_history.json` |
| TensorBoard Logs | `logs/tensorboard/grape_efficientnet_b0_20260722_165056/` |
| TorchScript Export | `exports/grape/grape_disease.torchscript.pt` |
| ONNX Export | `exports/grape/grape_disease.onnx` |
| PyTorch Export Copy | `exports/grape/grape_disease.pth` |

---

## 4. Testing Status

**Testing: COMPLETED**

Evaluation was performed automatically after training on the held-out test set:

- **Test Directory:** `datasets/grape/test/`
- **Test Images:** 610 (original images only — no augmented data)
- **Model Used:** `weights/grape/best_model.pth`
- **Evaluation Date:** 2026-07-22

---

## 5. Test Set Evaluation Report

### Overall Metrics

| Metric | Score |
|--------|------:|
| **Accuracy** | **100.00%** |
| Macro Precision | 1.0000 |
| Macro Recall | 1.0000 |
| Macro F1 | 1.0000 |
| Weighted Precision | 1.0000 |
| Weighted Recall | 1.0000 |
| Weighted F1 | 1.0000 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| Black Rot | 1.00 | 1.00 | 1.00 | 177 |
| Esca (Black Measles) | 1.00 | 1.00 | 1.00 | 207 |
| Leaf Blight (Isariopsis Leaf Spot) | 1.00 | 1.00 | 1.00 | 162 |
| Healthy | 1.00 | 1.00 | 1.00 | 64 |
| **Total** | | | | **610** |

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

### Confusion Matrix

| True \ Predicted | Black Rot | Esca | Leaf Blight | Healthy |
|------------------|----------:|-----:|------------:|--------:|
| **Black Rot** | 177 | 0 | 0 | 0 |
| **Esca** | 0 | 207 | 0 | 0 |
| **Leaf Blight** | 0 | 0 | 162 | 0 |
| **Healthy** | 0 | 0 | 0 | 64 |

Perfect classification — zero misclassifications across all 610 test images.

### ROC AUC (One-vs-Rest)

| Class | AUC |
|-------|----:|
| Black Rot | 1.0000 |
| Esca (Black Measles) | 1.0000 |
| Leaf Blight (Isariopsis Leaf Spot) | 1.0000 |
| Healthy | 1.0000 |

### Precision-Recall AUC (One-vs-Rest)

| Class | Average Precision |
|-------|------------------:|
| Black Rot | 1.0000 |
| Esca (Black Measles) | 1.0000 |
| Leaf Blight (Isariopsis Leaf Spot) | 1.0000 |
| Healthy | 1.0000 |

### Evaluation Artifacts

| File | Location |
|------|----------|
| Metrics JSON | `evaluation/grape/metrics.json` |
| Classification Report | `evaluation/grape/classification_report.txt` |
| Confusion Matrix Plot | `evaluation/grape/confusion_matrix.png` |
| ROC Curves Plot | `evaluation/grape/roc_curves.png` |
| Precision-Recall Curves | `evaluation/grape/precision_recall_curves.png` |
| Raw Predictions | `evaluation/grape/predictions.json` |

---

## 6. Export Verification

Both exported models were verified with a sample test image (`Black Rot` leaf):

| Backend | Prediction | Confidence |
|---------|------------|------------|
| PyTorch | Black Rot | 100.00% |
| TorchScript | Black Rot | 100.00% |
| ONNX | Black Rot | 100.00% |

---

## 7. Conclusion

| Stage | Status |
|-------|--------|
| Training | Completed (17/50 epochs, early stopped) |
| Validation | Best accuracy **100.00%** |
| Testing | **Completed** — accuracy **100.00%** |
| Export | TorchScript + ONNX verified |
| Production Ready | Yes |

The EfficientNet-B0 model is trained, evaluated, and exported for deployment via the PlantDiseaseAI desktop application.
