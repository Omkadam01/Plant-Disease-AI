# Models

This folder contains the two final EfficientNet-B0 checkpoints used by the application and their crop/runtime YAML configuration.

- `Final_Model/grape/best_model.pth` — four grape classes.
- `Final_Model/tomato/best_model.pth` — ten tomato classes.
- `Final_Model/configs/` — shared, crop-specific, and Raspberry Pi settings.

No intermediate checkpoints, optimizer states, duplicate weights, or exported model copies are included. Reusable model construction code is under `Evaluation/models/`.
