# PlantDiseaseAI — Tomato Training Balancing Report

**Generated:** 2026-07-22T18:34:14.522755+00:00
**Source:** `C:\Projects\Plant_Disease_v3\datasets\tomato\split\train`
**Destination:** `C:\Projects\Plant_Disease_v3\datasets\tomato\balanced_train`
**Target per class:** 1500
**Random seed:** 42

## Summary

- **Original training images:** 12,669
- **Generated (augmented):** 4,561
- **Final balanced size:** 17,230

## Per-Class Statistics

| Class | Original | Augmented | Final |
|-------|----------:|----------:|------:|
| `Tomato___Bacterial_spot` | 1,485 | 15 | 1,500 |
| `Tomato___Early_blight` | 700 | 800 | 1,500 |
| `Tomato___healthy` | 1,105 | 395 | 1,500 |
| `Tomato___Late_blight` | 1,330 | 170 | 1,500 |
| `Tomato___Leaf_Mold` | 666 | 834 | 1,500 |
| `Tomato___Septoria_leaf_spot` | 1,240 | 260 | 1,500 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | 1,170 | 330 | 1,500 |
| `Tomato___Target_Spot` | 983 | 517 | 1,500 |
| `Tomato___Tomato_mosaic_virus` | 260 | 1,240 | 1,500 |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 3,730 | 0 | 3,730 |

## Validation

- **every_class_exists:** True
- **all_images_open:** True
- **all_rgb:** True
- **all_256x256:** True
- **no_duplicate_filenames:** True
- **no_augmented_outside_balanced_train:** True
- **val_test_untouched:** True
- **passed:** True
