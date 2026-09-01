"""
Tests for src.evaluate — Model Evaluation.

Covers metrics correctness, per-class metrics, confusion matrix,
probability output, and report generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.evaluate import (
    compute_confusion_matrix,
    compute_metrics,
    evaluate,
    plot_confusion_matrix,
    predict,
    save_evaluation_report,
)
from src.dataset import IDX_TO_CLASS, NUM_CLASSES
from src.model import build_model


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


def _create_mini_manifest(tmp_path: Path, name: str, n: int = 6) -> Path:
    classes = ["Healthy", "Early Blight", "Late Blight"]
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        cls = classes[i % len(classes)]
        img_path = tmp_path / "images" / f"{name}_{i}.jpg"
        _create_test_image(img_path, size=(64, 64))
        rows.append({"image_path": str(img_path), "clean_class": cls})
    df = pd.DataFrame(rows)
    manifest_path = splits_dir / f"{name}.csv"
    df.to_csv(manifest_path, index=False)
    return manifest_path


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_accuracy(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["overall"]["accuracy"] == 1.0
        assert metrics["overall"]["f1_macro"] == 1.0

    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["overall"]["accuracy"] == 0.0

    def test_per_class_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 1, 2, 2, 0])
        metrics = compute_metrics(y_true, y_pred)
        # Check per-class structure
        for cls in ["Healthy", "Early Blight", "Late Blight"]:
            assert cls in metrics["per_class"]
            assert "precision" in metrics["per_class"][cls]
            assert "recall" in metrics["per_class"][cls]
            assert "f1" in metrics["per_class"][cls]
            assert "support" in metrics["per_class"][cls]

    def test_classification_report(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(y_true, y_pred)
        assert "classification_report" in metrics
        assert "Healthy" in metrics["classification_report"]


class TestConfusionMatrix:
    def test_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_perfect_diagonal(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        cm = compute_confusion_matrix(y_true, y_pred)
        # Diagonal should be counts, off-diagonal should be 0
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2
        assert cm.sum() - cm.trace() == 0


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_output_format(self):
        model = build_model(num_classes=3, pretrained=False)
        images = torch.randn(6, 3, 64, 64)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        paths = [f"img_{i}.jpg" for i in range(6)]
        dataset = _MockDataset(images, labels, paths)
        loader = DataLoader(dataset, batch_size=3)

        result = predict(model, loader, device="cpu")
        assert "y_true" in result
        assert "y_pred" in result
        assert "probabilities" in result
        assert "image_paths" in result
        assert len(result["y_true"]) == 6
        assert len(result["y_pred"]) == 6
        assert result["probabilities"].shape == (6, 3)

    def test_probabilities_sum_to_one(self):
        model = build_model(num_classes=3, pretrained=False)
        images = torch.randn(4, 3, 64, 64)
        labels = torch.tensor([0, 1, 0, 2])
        paths = [f"img_{i}.jpg" for i in range(4)]
        dataset = _MockDataset(images, labels, paths)
        loader = DataLoader(dataset, batch_size=2)

        result = predict(model, loader, device="cpu")
        prob_sums = result["probabilities"].sum(axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------


class TestPlotConfusionMatrix:
    def test_plot_created(self, tmp_path):
        cm = np.array([[10, 1, 0], [2, 8, 1], [0, 2, 12]])
        path = tmp_path / "cm.png"
        plot_confusion_matrix(cm, ["Healthy", "Early Blight", "Late Blight"], path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Report saving tests
# ---------------------------------------------------------------------------


class TestSaveEvaluationReport:
    def test_saves_files(self, tmp_path):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(y_true, y_pred)
        cm = compute_confusion_matrix(y_true, y_pred)
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

        output_dir = tmp_path / "reports"
        save_evaluation_report(metrics, cm, class_names, output_dir)

        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "classification_report.txt").exists()
        assert (output_dir / "confusion_matrix.png").exists()

        # Verify JSON structure
        data = json.loads((output_dir / "metrics.json").read_text())
        assert "overall" in data
        assert "per_class" in data


# ---------------------------------------------------------------------------
# Full evaluate pipeline test
# ---------------------------------------------------------------------------


class TestEvaluatePipeline:
    def test_evaluate(self):
        model = build_model(num_classes=3, pretrained=False)
        images = torch.randn(6, 3, 64, 64)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        paths = [f"img_{i}.jpg" for i in range(6)]
        dataset = _MockDataset(images, labels, paths)
        loader = DataLoader(dataset, batch_size=3)

        result = evaluate(model, loader, device="cpu")
        assert "metrics" in result
        assert "confusion_matrix" in result
        assert result["confusion_matrix"].shape == (3, 3)
