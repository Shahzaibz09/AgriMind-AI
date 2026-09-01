"""
Tests for src.train — Training Pipeline.

Covers configuration, seed handling, checkpoint metadata,
one-epoch training with synthetic data, and CPU compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.train import (
    TrainingConfig,
    compute_class_weights,
    load_config,
    resolve_device,
    set_seed,
    train,
    train_one_epoch,
    validate,
)
from src.dataset import get_eval_transforms, get_train_transforms


class _MockDataset(Dataset):
    """Tiny dataset that returns (image_tensor, label, path)."""
    def __init__(self, images, labels, paths):
        self.images = images
        self.labels = labels
        self.paths = paths
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.paths[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(path: Path, size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(str(path))


def _create_mini_manifests(tmp_path: Path, n_per_split: int = 6) -> Path:
    """Create mini train/val/test manifests with real images."""
    classes = ["Healthy", "Early Blight", "Late Blight"]
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    uid = 0
    for split in ["train", "val", "test"]:
        rows = []
        for i in range(n_per_split):
            cls = classes[i % len(classes)]
            img_path = tmp_path / "images" / f"img_{uid}.jpg"
            _create_test_image(img_path, size=(64, 64))
            rows.append({"image_path": str(img_path), "clean_class": cls})
            uid += 1
        df = pd.DataFrame(rows)
        df.to_csv(splits_dir / f"{split}.csv", index=False)
    return splits_dir


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.seed == 42
        assert config.batch_size == 32
        assert config.epochs == 10
        assert config.optimizer == "Adam"
        assert config.num_classes == 3
        assert config.class_weighted_loss is False

    def test_custom_values(self):
        config = TrainingConfig(batch_size=16, epochs=5, learning_rate=0.01)
        assert config.batch_size == 16
        assert config.epochs == 5
        assert config.learning_rate == 0.01


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "seed: 99\ntraining:\n  batch_size: 64\n  epochs: 20\n"
            "num_classes: 3\nimage:\n  height: 128\n",
            encoding="utf-8",
        )
        config = load_config(str(config_file))
        assert config.seed == 99
        assert config.batch_size == 64
        assert config.epochs == 20
        assert config.image_size == 128

    def test_missing_config_returns_defaults(self):
        config = load_config("nonexistent_config.yaml")
        assert config.seed == 42
        assert config.batch_size == 32


# ---------------------------------------------------------------------------
# Device and seed tests
# ---------------------------------------------------------------------------


class TestDeviceAndSeed:
    def test_resolve_device_cpu(self):
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_resolve_device_auto(self):
        device = resolve_device("auto")
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device == expected

    def test_set_seed_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# One-epoch training tests
# ---------------------------------------------------------------------------


class TestTrainOneEpoch:
    def test_train_one_epoch(self):
        """Test one epoch with tiny synthetic data."""
        from src.model import build_model

        model = build_model(num_classes=3, pretrained=False)
        model.to("cpu")

        # Create tiny synthetic data
        images = torch.randn(8, 3, 64, 64)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        paths = [f"img_{i}.jpg" for i in range(8)]
        dataset = _MockDataset(images, labels, paths)
        loader = DataLoader(dataset, batch_size=4)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss, acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1


class TestValidate:
    def test_validate(self):
        """Test validation with tiny synthetic data."""
        from src.model import build_model

        model = build_model(num_classes=3, pretrained=False)
        model.to("cpu")

        images = torch.randn(6, 3, 64, 64)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        paths = [f"img_{i}.jpg" for i in range(6)]
        dataset = _MockDataset(images, labels, paths)
        loader = DataLoader(dataset, batch_size=3)

        criterion = torch.nn.CrossEntropyLoss()
        loss, acc = validate(model, loader, criterion, torch.device("cpu"))
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0


# ---------------------------------------------------------------------------
# Full training pipeline test (one epoch)
# ---------------------------------------------------------------------------


class TestTrainPipeline:
    def test_one_epoch_training(self, tmp_path):
        """Run one epoch of training with mini synthetic data."""
        splits_dir = _create_mini_manifests(tmp_path, n_per_split=6)

        config = TrainingConfig(
            seed=42,
            batch_size=3,
            epochs=1,
            learning_rate=0.001,
            pretrained=False,
            freeze_backbone=False,
            image_size=64,
            num_classes=3,
            device="cpu",
            splits_dir=str(splits_dir),
            checkpoint_path=str(tmp_path / "models" / "best.pth"),
            metadata_path=str(tmp_path / "models" / "metadata.json"),
            history_path=str(tmp_path / "reports" / "history.json"),
            loss_curve_path=str(tmp_path / "reports" / "loss.png"),
            accuracy_curve_path=str(tmp_path / "reports" / "acc.png"),
            early_stopping_patience=5,
        )

        result = train(config)

        assert "history" in result
        assert len(result["history"]["train_loss"]) == 1
        assert len(result["history"]["val_loss"]) == 1
        assert result["best_val_acc"] >= 0
        assert result["best_epoch"] >= 0

        # Check checkpoint was saved
        assert Path(config.checkpoint_path).exists()

        # Check metadata was saved
        assert Path(config.metadata_path).exists()
        meta = json.loads(Path(config.metadata_path).read_text())
        assert meta["architecture"] == "mobilenet_v3_small"
        assert meta["seed"] == 42

        # Check history was saved
        assert Path(config.history_path).exists()

        # Check plots were saved
        assert Path(config.loss_curve_path).exists()
        assert Path(config.accuracy_curve_path).exists()


# ---------------------------------------------------------------------------
# Checkpoint metadata test
# ---------------------------------------------------------------------------


class TestCheckpointMetadata:
    def test_metadata_contents(self, tmp_path):
        splits_dir = _create_mini_manifests(tmp_path, n_per_split=6)
        ckpt_path = str(tmp_path / "models" / "best.pth")

        config = TrainingConfig(
            seed=42,
            batch_size=3,
            epochs=1,
            pretrained=False,
            freeze_backbone=False,
            image_size=64,
            num_classes=3,
            device="cpu",
            splits_dir=str(splits_dir),
            checkpoint_path=ckpt_path,
            metadata_path=str(tmp_path / "meta.json"),
            history_path=str(tmp_path / "hist.json"),
            loss_curve_path=str(tmp_path / "loss.png"),
            accuracy_curve_path=str(tmp_path / "acc.png"),
        )

        train(config)

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "model_state_dict" in ckpt
        assert "metadata" in ckpt
        assert ckpt["metadata"]["seed"] == 42
        assert ckpt["metadata"]["num_classes"] == 3
        assert ckpt["architecture"] == "mobilenet_v3_small"


# ---------------------------------------------------------------------------
# Class-weight tests
# ---------------------------------------------------------------------------


def _make_train_csv(tmp_path, class_counts: dict[str, int]) -> Path:
    """Create a training CSV with specified class counts."""
    rows = []
    for cls, count in class_counts.items():
        for i in range(count):
            img_path = tmp_path / "images" / f"{cls.replace(' ', '_')}_{i}.jpg"
            _create_test_image(img_path, size=(64, 64))
            rows.append({"image_path": str(img_path), "clean_class": cls})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestComputeClassWeights:
    def test_balanced_classes(self, tmp_path):
        """Equal counts → all weights = 1.0."""
        csv = _make_train_csv(tmp_path, {"Healthy": 100, "Early Blight": 100, "Late Blight": 100})
        weights = compute_class_weights(csv)
        assert weights.shape == (3,)
        for w in weights:
            assert w.item() == pytest.approx(1.0, abs=1e-5)

    def test_imbalanced_weights(self, tmp_path):
        """Inverse-frequency formula with normalisation."""
        csv = _make_train_csv(tmp_path, {"Healthy": 200, "Early Blight": 100, "Late Blight": 300})
        weights = compute_class_weights(csv)

        # Raw: total=600, num_classes=3
        # Healthy: 600/(3*200) = 1.0
        # Early Blight: 600/(3*100) = 2.0
        # Late Blight: 600/(3*300) = 0.6667
        # mean = (1.0 + 2.0 + 0.6667)/3 ≈ 1.2222
        # Normalised: Healthy=0.818, EB=1.636, LB=0.545
        assert weights[0].item() == pytest.approx(0.8182, abs=0.01)
        assert weights[1].item() == pytest.approx(1.6364, abs=0.01)
        assert weights[2].item() == pytest.approx(0.5455, abs=0.01)

    def test_weights_mean_is_one(self, tmp_path):
        """Normalised weights should have mean ≈ 1.0."""
        csv = _make_train_csv(tmp_path, {"Healthy": 1114, "Early Blight": 704, "Late Blight": 1330})
        weights = compute_class_weights(csv)
        assert weights.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_weights_order_matches_class_index(self, tmp_path):
        """Weights ordered by CLASS_TO_IDX: Healthy=0, Early Blight=1, Late Blight=2."""
        csv = _make_train_csv(tmp_path, {"Healthy": 500, "Early Blight": 100, "Late Blight": 400})
        weights = compute_class_weights(csv)
        # Early Blight has fewest samples → should have highest weight
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]

    def test_missing_class_raises(self, tmp_path):
        """Should raise ValueError if a class is missing."""
        csv = _make_train_csv(tmp_path, {"Healthy": 100, "Late Blight": 100})
        with pytest.raises(ValueError, match="Missing classes"):
            compute_class_weights(csv)

    def test_deterministic(self, tmp_path):
        """Same input → same weights."""
        csv = _make_train_csv(tmp_path, {"Healthy": 200, "Early Blight": 100, "Late Blight": 300})
        w1 = compute_class_weights(csv)
        w2 = compute_class_weights(csv)
        assert torch.equal(w1, w2)

    def test_returns_float32_tensor(self, tmp_path):
        csv = _make_train_csv(tmp_path, {"Healthy": 100, "Early Blight": 100, "Late Blight": 100})
        weights = compute_class_weights(csv)
        assert weights.dtype == torch.float32
        assert isinstance(weights, torch.Tensor)


class TestClassWeightedLossConfig:
    def test_config_default_false(self):
        config = TrainingConfig()
        assert config.class_weighted_loss is False

    def test_config_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "seed: 42\ntraining:\n  class_weighted_loss: true\n  batch_size: 32\n"
            "  epochs: 10\nnum_classes: 3\nimage:\n  height: 224\n",
            encoding="utf-8",
        )
        config = load_config(str(config_file))
        assert config.class_weighted_loss is True

    def test_config_missing_key_defaults_false(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "seed: 42\ntraining:\n  batch_size: 32\n  epochs: 10\n"
            "num_classes: 3\nimage:\n  height: 224\n",
            encoding="utf-8",
        )
        config = load_config(str(config_file))
        assert config.class_weighted_loss is False


class TestWeightedTrainingPipeline:
    def test_weighted_training_one_epoch(self, tmp_path):
        """Training with class_weighted_loss=True should work end-to-end."""
        splits_dir = _create_mini_manifests(tmp_path, n_per_split=6)

        config = TrainingConfig(
            seed=42,
            batch_size=3,
            epochs=1,
            learning_rate=0.001,
            pretrained=False,
            freeze_backbone=False,
            image_size=64,
            num_classes=3,
            device="cpu",
            class_weighted_loss=True,
            splits_dir=str(splits_dir),
            checkpoint_path=str(tmp_path / "models" / "best.pth"),
            metadata_path=str(tmp_path / "models" / "metadata.json"),
            history_path=str(tmp_path / "reports" / "history.json"),
            loss_curve_path=str(tmp_path / "reports" / "loss.png"),
            accuracy_curve_path=str(tmp_path / "reports" / "acc.png"),
            early_stopping_patience=5,
        )

        result = train(config)
        assert result["best_val_acc"] >= 0

        # Metadata should record class_weighted_loss
        meta = json.loads(Path(config.metadata_path).read_text())
        assert meta["class_weighted_loss"] is True
