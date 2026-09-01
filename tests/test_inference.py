"""
Tests for src.inference — Inference Module.

Covers deterministic preprocessing, prediction format,
confidence range, probability sum, and class mapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from src.inference import predict_image
from src.model import build_model, save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(path: Path, size=(256, 256), color=(100, 150, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(str(path))


# ---------------------------------------------------------------------------
# Predict image tests
# ---------------------------------------------------------------------------


class TestPredictImage:
    @pytest.fixture
    def model(self):
        m = build_model(num_classes=3, pretrained=False)
        m.eval()
        return m

    @pytest.fixture
    def image_path(self, tmp_path):
        path = tmp_path / "test_leaf.jpg"
        _create_test_image(path)
        return path

    def test_prediction_format(self, model, image_path):
        result = predict_image(model, image_path, device="cpu")
        assert "predicted_class" in result
        assert "predicted_index" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "image_path" in result

    def test_predicted_class_is_valid(self, model, image_path):
        result = predict_image(model, image_path, device="cpu")
        assert result["predicted_class"] in CLASS_TO_IDX

    def test_predicted_index_range(self, model, image_path):
        result = predict_image(model, image_path, device="cpu")
        assert 0 <= result["predicted_index"] < NUM_CLASSES

    def test_confidence_range(self, model, image_path):
        result = predict_image(model, image_path, device="cpu")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, model, image_path):
        result = predict_image(model, image_path, device="cpu")
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_contain_all_classes(self, model, image_path):
        result = predict_image(model, image_path, device="cpu")
        assert set(result["probabilities"].keys()) == set(CLASS_TO_IDX.keys())

    def test_deterministic_prediction(self, model, image_path):
        r1 = predict_image(model, image_path, device="cpu")
        r2 = predict_image(model, image_path, device="cpu")
        assert r1["predicted_class"] == r2["predicted_class"]
        np.testing.assert_allclose(
            list(r1["probabilities"].values()),
            list(r2["probabilities"].values()),
            atol=1e-6,
        )

    def test_image_path_in_result(self, model, image_path):
        result = predict_image(model, image_path, device="cpu")
        assert str(image_path) in result["image_path"]

    def test_with_checkpoint(self, tmp_path):
        """Test inference with a saved checkpoint."""
        model = build_model(num_classes=3, pretrained=False)
        ckpt_path = str(tmp_path / "ckpt.pth")
        save_checkpoint(model, ckpt_path)

        loaded = load_checkpoint(ckpt_path, pretrained=False)
        img_path = tmp_path / "leaf.jpg"
        _create_test_image(img_path)

        result = predict_image(loaded, img_path, device="cpu")
        assert result["predicted_class"] in CLASS_TO_IDX
        assert 0.0 <= result["confidence"] <= 1.0

    def test_custom_image_size(self, model, tmp_path):
        """Test inference with a different image size."""
        img_path = tmp_path / "leaf.jpg"
        _create_test_image(img_path, size=(128, 128))
        result = predict_image(model, img_path, device="cpu", image_size=128)
        assert result["predicted_class"] in CLASS_TO_IDX
