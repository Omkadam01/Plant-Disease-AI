# Tomato Final Test Evaluation Summary

## Dataset
- Test path: `datasets/tomato/split/test`
- Test images: **2713**
- Classes: **10**
- Weights: `weights/tomato/best_model.pth`

## Overall Metrics
- Test loss: **0.008305**
- Test accuracy (Top-1): **0.9985**
- Top-3 accuracy: **1.0000**
- Precision (macro): **0.9984**
- Recall (macro): **0.9980**
- F1-score (macro): **0.9982**
- Precision (weighted): **0.9985**
- Recall (weighted): **0.9985**
- F1-score (weighted): **0.9985**
- Macro AUC: **1.0000**
- Weighted AUC: **1.0000**
- Average confidence: **0.9996**
- Misclassifications: **4**
- Evaluation time: **31.20s**

## Class Highlights
- Best-performing class: **Tomato Yellow Leaf Curl Virus**
- Worst-performing class: **Target Spot**

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Bacterial Spot | 1.0000 | 1.0000 | 1.0000 | 318 |
| Early Blight | 1.0000 | 0.9933 | 0.9967 | 150 |
| Healthy | 0.9958 | 1.0000 | 0.9979 | 236 |
| Late Blight | 0.9965 | 0.9965 | 0.9965 | 285 |
| Leaf Mold | 1.0000 | 1.0000 | 1.0000 | 142 |
| Septoria Leaf Spot | 1.0000 | 1.0000 | 1.0000 | 266 |
| Spider Mites (Two-spotted Spider Mite) | 0.9921 | 1.0000 | 0.9960 | 250 |
| Target Spot | 1.0000 | 0.9905 | 0.9952 | 210 |
| Tomato Mosaic Virus | 1.0000 | 1.0000 | 1.0000 | 56 |
| Tomato Yellow Leaf Curl Virus | 1.0000 | 1.0000 | 1.0000 | 800 |

## Validation
- Every test image evaluated exactly once.
- No missing or duplicate predictions.
- Test dataset remained unchanged.
