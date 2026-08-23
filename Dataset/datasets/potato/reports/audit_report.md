# PlantDiseaseAI — Potato Dataset Audit Report

**Crop:** potato
**Source:** PlantVillage (Original)
**Canonical path:** `C:\Projects\Plant_Disease_v3\datasets\potato`
**Audited source:** `C:\Projects\Plant_Disease_v3\PlantVillage-Dataset\raw\color` (PlantVillage-Dataset/raw/color)
**Generated:** 2026-07-23T05:41:10.019913+00:00
**Mode:** read_only (no files modified)

## Executive Summary

- **Total images:** 2,152
- **Number of classes:** 3 (expected 3)
- **Exact duplicates:** 0 (0 groups)
- **Near duplicates:** 3 (3 groups)
- **Corrupted images:** 0
- **Invalid images (total):** 0
- **Zero-byte files:** 0
- **Hidden/system files:** 0
- **Structure valid:** True

## Class Folders

| # | Folder Name | Display Name | Images | Min Resolution | Max Resolution | Avg Resolution |
|---|-------------|--------------|--------|----------------|----------------|----------------|
| 1 | `Potato___Early_blight` | Early Blight | 1,000 | 256x256 | 256x256 | 256x256 |
| 2 | `Potato___healthy` | Healthy | 152 | 256x256 | 256x256 | 256x256 |
| 3 | `Potato___Late_blight` | Late Blight | 1,000 | 256x256 | 256x256 | 256x256 |

## Images Per Class

| Display Name | Folder | Count | % of Total |
|--------------|--------|------:|-----------:|
| Early Blight | `Potato___Early_blight` | 1,000 | 46.5% |
| Late Blight | `Potato___Late_blight` | 1,000 | 46.5% |
| Healthy | `Potato___healthy` | 152 | 7.1% |

## Folder Structure

- **Expected folders:** ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
- **Found folders:** ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
- **Missing expected folders:** None
- **Unexpected folders:** None
- **Empty folders:** None

## Global Image Statistics

- **Formats:** {'.jpg': 2152}
- **Color modes:** {'RGB': 2152}
- **Min resolution:** 256x256
- **Max resolution:** 256x256
- **Average resolution:** 256x256
- **Width range:** 256–256 (avg 256)
- **Height range:** 256–256 (avg 256)

## Class Imbalance Report

- **Min per class:** 152
- **Max per class:** 1000
- **Mean per class:** 717.33
- **Imbalance ratio (max/min):** 6.579
- **Coefficient of variation:** 0.557
- **Majority class:** `Potato___Early_blight`
- **Minority class:** `Potato___healthy`

## Dataset Quality Summary

- **Overall verdict:** Dataset quality is acceptable — review noted issues before preprocessing.
- **Issues found:** 3 near duplicate(s)
- **Cross-class leakage risk:** No

## Single-Class Assignment

- **Cross-class exact hash conflicts:** 0
- **Cross-class same-filename conflicts:** 0
- **Every image in exactly one class:** True

## Duplicate Detection

- **Exact duplicate groups:** 0
- **Exact duplicates (extra copies):** 0
- **Near duplicate groups:** 3
- **Near duplicates (extra copies):** 3

### Near Duplicate Groups

- Group 0 (2 files): Potato___Early_blight\1c4e0445-375a-4c5d-b6fd-6123d22f009f___RS_Early.B 7972.JPG, Potato___Early_blight\49b24f9e-12d4-4f32-8bc8-482d66514a17___RS_Early.B 8037.JPG
- Group 1 (2 files): Potato___Early_blight\5aee3e9f-0469-42e7-8ae7-56f7a1900fd5___RS_Early.B 9091.JPG, Potato___Early_blight\bc378ba0-533d-42db-a8b2-82decc73f4d5___RS_Early.B 9092.JPG
- Group 2 (2 files): Potato___Late_blight\c89d5a2a-2c1e-496e-84b8-b7a844a97ac9___RS_LB 4920.JPG, Potato___Late_blight\f528b00d-99a6-41db-8d96-9eed58b519c1___RS_LB 4544.JPG

## Data Quality Issues

- **Corrupted images:** 0
- **Zero-byte files:** 0
- **Invalid filenames:** 0
- **Unsupported formats:** 0
- **Hidden/system files:** 0

## Leaf Map Validation

| Class | Leafmap | Expected | On Disk | Missing | Extra |
|-------|---------|----------|---------|---------|-------|
| `Potato___Early_blight` | True | 1000 | 1000 | 0 | 0 |
| `Potato___healthy` | True | 152 | 152 | 0 | 0 |
| `Potato___Late_blight` | True | 1000 | 1000 | 0 | 0 |

## Recommended Next Actions

1. Proceed with potato raw preprocessing: copy PlantVillage classes to datasets/potato/raw/.
2. Define configs/crops/potato.yaml with 3 classes (Early Blight, Healthy, Late Blight).
3. Review 3 near-duplicate(s); apply perceptual-hash dedup (threshold=5) like tomato.
4. Address class imbalance (ratio 6.579:1): consider balancing Healthy (Potato___healthy) against Early/Late Blight during preprocessing.
5. Copy potato classes to datasets/potato/raw/ without modifying PlantVillage source.
6. Run corruption check and duplicate detection in preprocessing before train/val/test split.
