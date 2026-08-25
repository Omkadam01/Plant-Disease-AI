"""PyTorch dataset for folder-based image classification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.image_utils import read_image_rgb

if TYPE_CHECKING:
    from albumentations import Compose


class PlantDiseaseDataset(Dataset):
    """
    Image folder dataset for plant disease classification.

    Expects directory structure: ``<root>/<class_folder>/<images>``.
    """

    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform: Compose | Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for class_name, class_id in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in extensions:
                    self.samples.append((path, class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = read_image_rgb(path)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    @property
    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted CE loss."""
        counts = self.class_counts
        if not counts:
            raise ValueError("Dataset has no samples")

        num_classes = max(counts.keys()) + 1
        total = sum(counts.values())
        weights = []
        for i in range(num_classes):
            count = counts.get(i, 0)
            if count == 0:
                weights.append(0.0)
            else:
                weights.append(total / (num_classes * count))
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    class_to_idx: dict[str, int],
    batch_size: int,
    num_workers: int,
    image_size: int,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, PlantDiseaseDataset]:
    """Create train and validation dataloaders."""
    from preprocessing.transforms import get_train_transforms, get_val_transforms

    train_dataset = PlantDiseaseDataset(
        root=train_dir,
        class_to_idx=class_to_idx,
        transform=get_train_transforms(image_size),
    )
    val_dataset = PlantDiseaseDataset(
        root=val_dir,
        class_to_idx=class_to_idx,
        transform=get_val_transforms(image_size),
    )

    train_loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": len(train_dataset) >= batch_size,
    }
    if num_workers > 0 and persistent_workers:
        train_loader_kwargs["persistent_workers"] = True

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset
