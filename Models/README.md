# Models

This folder contains the two final EfficientNet-B0 checkpoints used by the application, their crop/runtime YAML configuration, and the model factory source.

- `weights/grape/best_model.pth` — four grape classes.
- `weights/tomato/best_model.pth` — ten tomato classes.
- `configs/` — shared, crop-specific, and Raspberry Pi settings.
- `models/` — model construction code.

No intermediate checkpoints, optimizer states, duplicate weights, or exported model copies are included.

