"""
AgriMind AI — PyTorch Dataset for Tomato Leaf Disease Classification.

Loads split manifests (train.csv, val.csv, test.csv) and returns
(image_tensor, label, image_path) tuples.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Deterministic label mapping (clean_class -> integer index)
CLASS_TO_IDX: dict[str, int] = {
    "Healthy": 0,
    "Early Blight": 1,
    "Late Blight": 2,
}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

# ImageNet normalisation (used by torchvision pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------


def get_train_transforms(image_size: int = 224) -> Callable:
    """Return training transforms with modest augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> Callable:
    """Return deterministic validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TomatoLeafDataset(Dataset):
    """PyTorch Dataset for tomato leaf disease classification.

    Loads a split manifest CSV with columns ``image_path`` and ``clean_class``.

    Parameters
    ----------
    manifest_path : str or Path
        Path to the split CSV file.
    transform : callable, optional
        Image transform pipeline. If ``None``, uses eval transforms.
    image_size : int
        Target image size (default 224).
    class_to_idx : dict of str -> int, optional
        Crop-specific label mapping (clean class name -> integer index).
        Defaults to the module-level ``CLASS_TO_IDX`` (tomato/potato), so
        existing behaviour is unchanged for crops that share those names.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
        image_size: int = 224,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.class_to_idx: dict[str, int] = (
            dict(class_to_idx) if class_to_idx is not None else CLASS_TO_IDX
        )

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        import pandas as pd

        self.df = pd.read_csv(self.manifest_path)
        required = {"image_path", "clean_class"}
        if not required.issubset(self.df.columns):
            raise ValueError(
                f"Manifest missing required columns: {required - set(self.df.columns)}"
            )

        self.transform = transform or get_eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_path = str(row["image_path"])
        label_str = str(row["clean_class"])
        label = self.class_to_idx.get(label_str)

        if label is None:
            raise ValueError(
                f"Unknown class '{label_str}' at index {idx}. "
                f"Expected one of: {list(self.class_to_idx.keys())}"
            )

        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img)

        return img_tensor, label, image_path

    @property
    def class_names(self) -> list[str]:
        """Return ordered class names (index -> name)."""
        return [
            name
            for name, idx in sorted(
                self.class_to_idx.items(), key=lambda kv: kv[1]
            )
        ]


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------


def create_dataloaders(
    splits_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    class_to_idx: dict[str, int] | None = None,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create train/val/test DataLoaders from split manifests.

    Parameters
    ----------
    splits_dir : str or Path
        Directory containing train.csv, val.csv, test.csv.
    batch_size : int
        Batch size for all loaders.
    image_size : int
        Target image size.
    num_workers : int
        Number of worker processes (default 0 for main process).
    class_to_idx : dict of str -> int, optional
        Crop-specific label mapping. Defaults to ``CLASS_TO_IDX``.

    Returns
    -------
    dict with keys "train", "val", "test".
    """
    splits_dir = Path(splits_dir)

    train_ds = TomatoLeafDataset(
        splits_dir / "train.csv",
        transform=get_train_transforms(image_size),
        image_size=image_size,
        class_to_idx=class_to_idx,
    )
    val_ds = TomatoLeafDataset(
        splits_dir / "val.csv",
        transform=get_eval_transforms(image_size),
        image_size=image_size,
        class_to_idx=class_to_idx,
    )
    test_ds = TomatoLeafDataset(
        splits_dir / "test.csv",
        transform=get_eval_transforms(image_size),
        image_size=image_size,
        class_to_idx=class_to_idx,
    )

    return {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
