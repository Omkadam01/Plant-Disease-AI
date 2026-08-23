# Plant Disease AI

Plant Disease AI is a PySide6 desktop application for classifying grape and tomato leaf diseases. It uses two crop-specific EfficientNet-B0 checkpoints, displays top predictions, produces Grad-CAM explanations, supports image upload and camera capture, and can optionally request farmer-oriented guidance from Groq.

## Final models

| Crop | Classes | Input | Checkpoint | Retained test result |
|---|---:|---:|---|---:|
| Grape | 4 | 224 × 224 | `Models/weights/grape/best_model.pth` | 100% on 610 PlantVillage test images |
| Tomato | 10 | 256 × 256 | `Models/weights/tomato/best_model.pth` | 99.85% on 2,713 PlantVillage test images |

The recorded results are limited to held-out PlantVillage images and do not establish field performance. The retained tomato analysis documents overconfident predictions on out-of-distribution inputs.

## Dataset

Raw PlantVillage images and prepared train/validation/test splits are not included. `Dataset/` contains the grape and tomato class mappings required at runtime and a retained potato dataset audit. Potato is audit-only; there is no potato model.

## Run the application

Python 3.11 was the documented development target.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python App\run_app.py
```

On Linux or macOS, activate with `source .venv/bin/activate` and run `python App/run_app.py`. Use `--crop tomato` to open with the tomato model selected.

Groq guidance is optional. Copy `App/.env.example` to a local root-level `.env` file and replace the placeholder. Do not publish `.env`.

## Repository structure

```text
Plant-Disease-AI/
├── README.md
├── requirements.txt
├── Dataset/
├── Models/
├── Evaluation/
├── Outputs/
├── Documentation/
└── App/
```

- `Dataset/` — runtime class mappings and dataset audit metadata.
- `Models/` — final checkpoints, crop configuration, and model factory.
- `Evaluation/` — inference, preprocessing, Grad-CAM, training/evaluation modules, and retained reusable scripts.
- `Outputs/` — final grape and tomato evaluation artifacts.
- `Documentation/` — architecture, model notes, dataset/model reports, and Raspberry Pi requirements.
- `App/` — PySide6 UI, shared runtime utilities, translations, camera support, launchers, and setup helpers.

## Limitations

- Raw datasets and prepared splits are not distributed.
- The classifiers do not provide crop verification or out-of-distribution rejection.
- Groq guidance requires network access and a user-supplied API key.
- No prebuilt EXE or installer is included.
- No license file was present, so no license is asserted.

