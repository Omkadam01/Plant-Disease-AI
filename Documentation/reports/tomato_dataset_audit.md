# PlantDiseaseAI — Tomato Dataset Audit Report

**Crop:** tomato
**Source:** PlantVillage (Original) — raw/color
**Dataset root:** `C:\Projects\Plant_Disease_v3\PlantVillage-Dataset\raw\color`
**Generated:** 2026-07-22T18:13:53.121906+00:00
**Mode:** read_only (no files modified)

## Executive Summary

- **Total images:** 18,160
- **Number of classes:** 10
- **Exact duplicates:** 14 (14 groups)
- **Near duplicates:** 47 (47 groups)
- **Corrupted images:** 0
- **Zero-byte files:** 0
- **Hidden/system files:** 0
- **Structure valid:** True

## Class Folders

| # | Folder Name | Images | Min Resolution | Max Resolution | Avg Resolution |
|---|-------------|--------|----------------|----------------|----------------|
| 1 | `Tomato___Bacterial_spot` | 2,127 | 256x256 | 256x256 | 256x256 |
| 2 | `Tomato___Early_blight` | 1,000 | 256x256 | 256x256 | 256x256 |
| 3 | `Tomato___healthy` | 1,591 | 256x256 | 256x256 | 256x256 |
| 4 | `Tomato___Late_blight` | 1,909 | 256x256 | 256x256 | 256x256 |
| 5 | `Tomato___Leaf_Mold` | 952 | 256x256 | 256x256 | 256x256 |
| 6 | `Tomato___Septoria_leaf_spot` | 1,771 | 256x256 | 256x256 | 256x256 |
| 7 | `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | 256x256 | 256x256 | 256x256 |
| 8 | `Tomato___Target_Spot` | 1,404 | 256x256 | 256x256 | 256x256 |
| 9 | `Tomato___Tomato_mosaic_virus` | 373 | 256x256 | 256x256 | 256x256 |
| 10 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | 256x256 | 256x256 | 256x256 |

## Images Per Class

| Class | Count | % of Total |
|-------|------:|-----------:|
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | 29.5% |
| `Tomato___Bacterial_spot` | 2,127 | 11.7% |
| `Tomato___Late_blight` | 1,909 | 10.5% |
| `Tomato___Septoria_leaf_spot` | 1,771 | 9.8% |
| `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | 9.2% |
| `Tomato___healthy` | 1,591 | 8.8% |
| `Tomato___Target_Spot` | 1,404 | 7.7% |
| `Tomato___Early_blight` | 1,000 | 5.5% |
| `Tomato___Leaf_Mold` | 952 | 5.2% |
| `Tomato___Tomato_mosaic_virus` | 373 | 2.1% |

## Global Image Statistics

- **Formats:** {'.jpg': 18159, '.jpeg': 1}
- **Color modes:** {'RGB': 18160}
- **Min resolution:** 256x256
- **Max resolution:** 256x256
- **Average resolution:** 256x256
- **Width range:** 256–256 (avg 256)
- **Height range:** 256–256 (avg 256)

## Class Imbalance

- **Min per class:** 373
- **Max per class:** 5357
- **Mean per class:** 1816.0
- **Imbalance ratio (max/min):** 14.362
- **Coefficient of variation:** 0.705
- **Majority class:** `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
- **Minority class:** `Tomato___Tomato_mosaic_virus`

## Directory Structure Validation

- **Class folders found:** 10
- **Empty folders:** None
- **Validation passed:** True

## Single-Class Assignment

- **Cross-class exact hash conflicts:** 0
- **Cross-class same-filename conflicts:** 0
- **Every image in exactly one class:** True

## Duplicate Detection

- **Exact duplicate groups:** 14
- **Exact duplicates (extra copies):** 14
- **Near duplicate groups:** 47
- **Near duplicates (extra copies):** 47

### Exact Duplicate Groups (sample)

- Group 0 (2 files): Tomato___healthy\068e324c-faf6-40d6-8f83-578907f1cac5___GH_HL Leaf 466.1.JPG ...
- Group 1 (2 files): Tomato___healthy\1af0bfe1-4bcf-4b8b-be66-5d0953eb647e___GH_HL Leaf 482.2.JPG ...
- Group 2 (2 files): Tomato___healthy\37203047-d8ba-43f7-b31e-d496c41c569c___GH_HL Leaf 389.JPG ...
- Group 3 (2 files): Tomato___healthy\37aad83b-7ff8-4b35-b3ed-fb8e0f54910b___GH_HL Leaf 342.1.JPG ...
- Group 4 (2 files): Tomato___healthy\488feb1c-4b9f-44e7-8aa6-4103a9601f5f___GH_HL Leaf 434.JPG ...
- Group 5 (2 files): Tomato___healthy\9662364c-aaba-45e3-b907-10792d60578c___GH_HL Leaf 220.JPG ...
- Group 6 (2 files): Tomato___Late_blight\1a69b38b-c4eb-42c4-9584-bcb14fb8db0c___GHLB2 Leaf 9011.JPG ...
- Group 7 (2 files): Tomato___Late_blight\3fae9c64-18f0-4a67-9f97-554248bb1bed___GHLB_PS Leaf 24 Day 16.jpg ...
- Group 8 (2 files): Tomato___Late_blight\48c55974-9fe9-4f4b-94f7-c8cd127d1e05___GHLB_PS Leaf 23.7 Day 13.jpg ...
- Group 9 (2 files): Tomato___Late_blight\5688ea99-c949-41d0-bbab-9cbf0ffb8bcd___GHLB2 Leaf 8677.JPG ...

### Near Duplicate Groups (sample)

- Group 14 (2 files): Tomato___Bacterial_spot\196be0af-51c3-4587-bd91-9a606631d4f3___GCREC_Bact.Sp 3453.JPG, Tomato___Bacterial_spot\6d746b6c-52f8-4831-b541-c74d0984c130___GCREC_Bact.Sp 3169.JPG
- Group 15 (2 files): Tomato___Bacterial_spot\29184f19-db21-475d-8032-a8b7c3a9c7f4___GCREC_Bact.Sp 5784.JPG, Tomato___Bacterial_spot\5c5dbca4-4322-4227-a543-6ff6f876a531___GCREC_Bact.Sp 5785.JPG
- Group 16 (2 files): Tomato___Bacterial_spot\2b1247ef-594a-4ada-9377-1f46af51859c___GCREC_Bact.Sp 3095.JPG, Tomato___Bacterial_spot\dfca025b-a6d9-431a-977e-4bd174daa3a1___GCREC_Bact.Sp 5948.JPG
- Group 17 (2 files): Tomato___Bacterial_spot\611c6fa8-d805-4956-a786-2a4522c59f71___GCREC_Bact.Sp 5616.JPG, Tomato___Bacterial_spot\efd6193d-7d5b-4207-999d-2aeb08cd64e2___GCREC_Bact.Sp 5739.JPG
- Group 18 (2 files): Tomato___Bacterial_spot\741e2f09-1b62-42c2-b218-6504d57c0f27___GCREC_Bact.Sp 3302.JPG, Tomato___Bacterial_spot\e2286f6d-1e30-4528-ab51-84c18a16b7e1___GCREC_Bact.Sp 3503.JPG
- Group 19 (2 files): Tomato___healthy\0b330273-890c-4995-af72-cba070fc0061___GH_HL Leaf 312.JPG, Tomato___healthy\240a9583-9e6a-446d-8850-62bc23312ce9___GH_HL Leaf 312.4.JPG
- Group 20 (2 files): Tomato___healthy\0c4b06d5-4053-44fc-99b6-504934fdd3a9___GH_HL Leaf 199.1.JPG, Tomato___healthy\84991306-21f6-41fb-a723-0c7ba1dac9d3___GH_HL Leaf 199.JPG
- Group 21 (2 files): Tomato___healthy\2bd08574-9555-42a5-93b1-7fab8f9f8786___GH_HL Leaf 475.2.JPG, Tomato___healthy\5300bf32-485d-4978-9099-3175fb666da8___GH_HL Leaf 475.JPG
- Group 22 (2 files): Tomato___healthy\5015e0d2-a6e3-4859-aceb-78089d1b60f7___GH_HL Leaf 186.2.JPG, Tomato___healthy\a7c3088b-6aeb-4e8c-97dd-4661a4cacc38___GH_HL Leaf 186.JPG
- Group 23 (2 files): Tomato___healthy\71967e69-6356-49c1-81ce-624792b4d964___GH_HL Leaf 315.1.JPG, Tomato___healthy\9b5626b0-ea1d-45d0-bc41-c898f7253a75___GH_HL Leaf 315.2.JPG

## Data Quality Issues

- **Corrupted images:** 0
- **Zero-byte files:** 0
- **Invalid filenames:** 0
- **Unsupported formats:** 0
- **Hidden/system files:** 0

## Leaf Map Validation

| Class | Leafmap | Expected | On Disk | Missing | Extra |
|-------|---------|----------|---------|---------|-------|
| `Tomato___Bacterial_spot` | True | 2128 | 2127 | 2128 | 2127 |
| `Tomato___Early_blight` | True | 1000 | 1000 | 1000 | 1000 |
| `Tomato___healthy` | True | 1000 | 1591 | 1000 | 1591 |
| `Tomato___Late_blight` | True | 920 | 1909 | 920 | 1909 |
| `Tomato___Leaf_Mold` | True | 952 | 952 | 952 | 952 |
| `Tomato___Septoria_leaf_spot` | True | 1000 | 1771 | 1000 | 1771 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | True | 1676 | 1676 | 1676 | 1676 |
| `Tomato___Target_Spot` | True | 1404 | 1404 | 1404 | 1404 |
| `Tomato___Tomato_mosaic_virus` | False | None | 373 | — | — |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | True | 2740 | 5357 | 2740 | 5357 |

## Recommended Next Actions

1. Proceed with read-only preprocessing pipeline design for tomato using PlantVillage raw/color as source.
2. Define canonical folder names and class mapping YAML (10 classes including Tomato mosaic virus).
3. Review 14 exact duplicate(s) before training; deduplicate during preprocessing.
4. Review 47 near-duplicate(s); apply perceptual-hash dedup (threshold=5) like grape.
5. Address class imbalance (ratio 14.362:1): consider undersampling majority class 'Tomato___Tomato_Yellow_Leaf_Curl_Virus' or augmenting minority class 'Tomato___Tomato_mosaic_virus'.
6. Investigate 12820 file(s) referenced in leaf maps but missing on disk.
7. Copy tomato classes to datasets/tomato/raw/ without modifying PlantVillage source.
8. Run corruption check and duplicate detection in preprocessing before train/val/test split.
