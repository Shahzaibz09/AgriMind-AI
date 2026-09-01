"""
Tests for src.dataset — PyTorch Dataset for Tomato Leaf Disease Classification.

Covers manifest loading, class mapping, path resolution, image loading,
deterministic transforms, dataset length, and invalid path handling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from src.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    TomatoLeafDataset,
    create_dataloaders,
    get_eval_transforms,
    get_train_transforms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(path: Path, size=(64, 64)) -> None:
    """Create a simple test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(str(path))


def _create_manifest(tmp_path: Path, name: str, n: int = 6) -> Path:
    """Create a test split manifest with images."""
    classes = ["Healthy", "Early Blight", "Late Blight"]
    rows = []
    for i in range(n):
        cls = classes[i % len(classes)]
        img_path = tmp_path / "images" / f"img_{i}.jpg"
        _create_test_image(img_path, size=(256, 256))
        rows.append({"image_path": str(img_path), "clean_class": cls})
    df = pd.DataFrame(rows)
    manifest_path = tmp_path / "splits" / f"{name}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return manifest_path


# ---------------------------------------------------------------------------
# Class mapping tests
# ---------------------------------------------------------------------------


class TestClassMapping:
    def test_class_to_idx_has_3_classes(self):
        assert len(CLASS_TO_IDX) == 3
        assert NUM_CLASSES == 3

    def test_deterministic_mapping(self):
        assert CLASS_TO_IDX["Healthy"] == 0
        assert CLASS_TO_IDX["Early Blight"] == 1
        assert CLASS_TO_IDX["Late Blight"] == 2

    def test_idx_to_class_inverse(self):
        for cls, idx in CLASS_TO_IDX.items():
            assert IDX_TO_CLASS[idx] == cls


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_train_transforms_output_shape(self):
        transform = get_train_transforms(image_size=224)
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_eval_transforms_deterministic(self):
        transform = get_eval_transforms(image_size=224)
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        t1 = transform(img)
        t2 = transform(img)
        assert torch.equal(t1, t2)

    def test_eval_transforms_no_augmentation(self):
        """Eval transforms should produce identical outputs for same input."""
        transform = get_eval_transforms(image_size=224)
        img = Image.new("RGB", (256, 256), color=(50, 100, 150))
        results = [transform(img) for _ in range(5)]
        for r in results[1:]:
            assert torch.equal(results[0], r)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestTomatoLeafDataset:
    def test_manifest_loading(self, tmp_path):
        manifest = _create_manifest(tmp_path, "train", n=6)
        ds = TomatoLeafDataset(manifest)
        assert len(ds) == 6

    def test_class_names(self, tmp_path):
        manifest = _create_manifest(tmp_path, "train", n=3)
        ds = TomatoLeafDataset(manifest)
        assert ds.class_names == ["Healthy", "Early Blight", "Late Blight"]

    def test_getitem_returns_tuple(self, tmp_path):
        manifest = _create_manifest(tmp_path, "train", n=3)
        ds = TomatoLeafDataset(manifest, image_size=64)
        img, label, path = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert 0 <= label <= 2
        assert isinstance(path, str)

    def test_dataset_length(self, tmp_path):
        manifest = _create_manifest(tmp_path, "val", n=9)
        ds = TomatoLeafDataset(manifest)
        assert len(ds) == 9

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TomatoLeafDataset(tmp_path / "nonexistent.csv")

    def test_missing_columns_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"wrong": [1, 2, 3]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            TomatoLeafDataset(bad_csv)

    def test_unknown_class_raises(self, tmp_path):
        manifest = tmp_path / "bad_class.csv"
        pd.DataFrame({
            "image_path": [str(tmp_path / "img.jpg")],
            "clean_class": ["UnknownClass"],
        }).to_csv(manifest, index=False)
        img_path = tmp_path / "img.jpg"
        _create_test_image(img_path)
        ds = TomatoLeafDataset(manifest)
        with pytest.raises(ValueError, match="Unknown class"):
            ds[0]

    def test_invalid_path_raises(self, tmp_path):
        manifest = tmp_path / "bad_path.csv"
        pd.DataFrame({
            "image_path": [str(tmp_path / "nonexistent.jpg")],
            "clean_class": ["Healthy"],
        }).to_csv(manifest, index=False)
        ds = TomatoLeafDataset(manifest)
        with pytest.raises(Exception):  # FileNotFoundError from PIL
            ds[0]


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------


class TestCreateDataLoaders:
    def test_creates_all_loaders(self, tmp_path):
        for name in ["train", "val", "test"]:
            _create_manifest(tmp_path, name, n=6)
        loaders = create_dataloaders(
            tmp_path / "splits", batch_size=2, image_size=64
        )
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_batch_size(self, tmp_path):
        for name in ["train", "val", "test"]:
            _create_manifest(tmp_path, name, n=4)
        loaders = create_dataloaders(
            tmp_path / "splits", batch_size=2, image_size=64
        )
        batch = next(iter(loaders["val"]))
        images, labels, paths = batch
        assert images.shape[0] == 2
        assert labels.shape[0] == 2
