"""
Tests for src.gradcam — Grad-CAM Explainability Module.

Covers target layer selection, activation/gradient capture, CAM shape,
CAM normalisation, predicted class, hook cleanup, inference with and
without Grad-CAM, backward compatibility, and invalid image handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from src.gradcam import (
    GradCAM,
    denormalize,
    generate_overlay,
    get_gradcam_target_layer,
    predict_image_with_gradcam,
    save_gradcam_outputs,
)
from src.inference import predict_image
from src.model import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(path: Path, size=(256, 256), color=(100, 150, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(str(path))


@pytest.fixture
def model():
    m = build_model(num_classes=3, pretrained=False)
    m.eval()
    return m


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "test_leaf.jpg"
    _create_test_image(p)
    return p


# ---------------------------------------------------------------------------
# Target layer selection
# ---------------------------------------------------------------------------


class TestTargetLayerSelection:
    def test_default_target_is_features_last(self, model):
        layer = get_gradcam_target_layer(model)
        assert layer is model.features[-1]

    def test_target_layer_output_channels(self, model):
        layer = get_gradcam_target_layer(model)
        # Last 1x1 conv of MobileNetV3-Small outputs 576 channels
        # Find the Conv2d inside the Conv2dNormActivation
        conv = layer[0] if hasattr(layer, "__getitem__") else layer
        # Walk to find the conv submodule
        for mod in layer.modules():
            if isinstance(mod, torch.nn.Conv2d):
                conv = mod
                break
        assert conv.out_channels == 576


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


class TestActivationCapture:
    def test_activations_captured(self, model):
        from src.dataset import get_eval_transforms
        transform = get_eval_transforms(224)
        img_tensor = torch.randn(1, 3, 224, 224)

        cam = GradCAM(model)
        cam(img_tensor)
        assert cam._activations is not None
        assert cam._activations.shape[0] == 1
        assert cam._activations.shape[1] == 576  # channels
        assert cam._activations.shape[2] == 7   # height
        assert cam._activations.shape[3] == 7   # width
        cam.remove_hooks()


# ---------------------------------------------------------------------------
# Gradient capture
# ---------------------------------------------------------------------------


class TestGradientCapture:
    def test_gradients_captured(self, model):
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        cam(img_tensor)
        assert cam._gradients is not None
        assert cam._gradients.shape == cam._activations.shape
        cam.remove_hooks()

    def test_gradients_non_zero(self, model):
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        cam(img_tensor)
        assert cam._gradients.abs().sum() > 0
        cam.remove_hooks()


# ---------------------------------------------------------------------------
# CAM shape
# ---------------------------------------------------------------------------


class TestCAMShape:
    def test_cam_shape_matches_input(self, model):
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        heatmap, _, _ = cam(img_tensor)
        assert heatmap.shape == (224, 224)
        cam.remove_hooks()

    def test_cam_shape_custom_size(self, model):
        img_tensor = torch.randn(1, 3, 128, 128)
        cam = GradCAM(model)
        heatmap, _, _ = cam(img_tensor)
        assert heatmap.shape == (128, 128)
        cam.remove_hooks()


# ---------------------------------------------------------------------------
# CAM normalisation
# ---------------------------------------------------------------------------


class TestCAMNormalisation:
    def test_cam_range(self, model):
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        heatmap, _, _ = cam(img_tensor)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        cam.remove_hooks()

    def test_cam_max_is_one(self, model):
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        heatmap, _, _ = cam(img_tensor)
        assert abs(heatmap.max() - 1.0) < 1e-6
        cam.remove_hooks()


# ---------------------------------------------------------------------------
# Predicted class
# ---------------------------------------------------------------------------


class TestPredictedClass:
    def test_predicted_class_is_valid(self, model):
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        _, class_idx, _ = cam(img_tensor)
        assert 0 <= class_idx < NUM_CLASSES
        cam.remove_hooks()

    def test_explicit_class_idx(self, model):
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        _, class_idx, _ = cam(img_tensor, class_idx=2)
        assert class_idx == 2
        cam.remove_hooks()

    def test_logits_returned(self, model):
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        _, _, logits = cam(img_tensor)
        assert logits.shape == (1, 3)
        cam.remove_hooks()


# ---------------------------------------------------------------------------
# Hook cleanup
# ---------------------------------------------------------------------------


class TestHookCleanup:
    def test_hooks_removed_after_context_manager(self, model):
        target = get_gradcam_target_layer(model)
        with GradCAM(model) as cam:
            pass  # hooks registered
        # After exiting, hooks should be gone — verify by checking no
        # forward hooks remain on the target layer
        assert len(target._forward_hooks) == 0
        assert len(target._backward_hooks) == 0

    def test_manual_remove_hooks(self, model):
        target = get_gradcam_target_layer(model)
        cam = GradCAM(model)
        cam.remove_hooks()
        assert len(target._forward_hooks) == 0
        assert len(target._backward_hooks) == 0

    def test_multiple_calls_same_engine(self, model):
        """Engine can be called multiple times before removal."""
        img_tensor = torch.randn(1, 3, 224, 224)
        cam = GradCAM(model)
        h1, _, _ = cam(img_tensor)
        h2, _, _ = cam(img_tensor)
        assert h1.shape == h2.shape
        cam.remove_hooks()


# ---------------------------------------------------------------------------
# predict_image_with_gradcam
# ---------------------------------------------------------------------------


class TestPredictImageWithGradcam:
    def test_result_keys(self, model, image_path):
        result = predict_image_with_gradcam(model, image_path, device="cpu")
        assert "predicted_class" in result
        assert "predicted_index" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "gradcam" in result
        assert "gradcam_target" in result

    def test_gradcam_shape(self, model, image_path):
        result = predict_image_with_gradcam(model, image_path, device="cpu")
        assert result["gradcam"].shape == (224, 224)

    def test_probabilities_sum(self, model, image_path):
        result = predict_image_with_gradcam(model, image_path, device="cpu")
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5

    def test_save_outputs(self, model, image_path, tmp_path):
        output_dir = tmp_path / "gradcam_out"
        result = predict_image_with_gradcam(
            model, image_path, device="cpu",
            save_outputs=True, output_dir=output_dir,
        )
        assert "gradcam_paths" in result
        assert Path(result["gradcam_paths"]["heatmap"]).exists()
        assert Path(result["gradcam_paths"]["overlay"]).exists()

    def test_no_save_by_default(self, model, image_path, tmp_path):
        output_dir = tmp_path / "gradcam_out"
        result = predict_image_with_gradcam(
            model, image_path, device="cpu",
            output_dir=output_dir,
        )
        assert "gradcam_paths" not in result


# ---------------------------------------------------------------------------
# Inference backward compatibility
# ---------------------------------------------------------------------------


class TestInferenceBackwardCompat:
    def test_default_inference_unchanged(self, model, image_path):
        """predict_image() with gradcam=False must return the original keys."""
        result = predict_image(model, image_path, device="cpu")
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "gradcam" not in result
        assert "gradcam_target" not in result

    def test_gradcam_flag_adds_keys(self, model, image_path):
        """predict_image() with gradcam=True adds gradcam keys."""
        result = predict_image(model, image_path, device="cpu", gradcam=True)
        assert "predicted_class" in result
        assert "gradcam" in result
        assert "gradcam_target" in result

    def test_deterministic_default_path(self, model, image_path):
        """Same prediction for same image via default path."""
        r1 = predict_image(model, image_path, device="cpu")
        r2 = predict_image(model, image_path, device="cpu")
        assert r1["predicted_class"] == r2["predicted_class"]


# ---------------------------------------------------------------------------
# Invalid image handling
# ---------------------------------------------------------------------------


class TestInvalidImageHandling:
    def test_missing_image_raises(self, model, tmp_path):
        with pytest.raises(Exception):
            predict_image_with_gradcam(model, tmp_path / "no_such.jpg", device="cpu")

    def test_corrupt_image_raises(self, model, tmp_path):
        bad_path = tmp_path / "bad.jpg"
        bad_path.write_bytes(b"not a real image")
        with pytest.raises(Exception):
            predict_image_with_gradcam(model, bad_path, device="cpu")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


class TestVisualisationHelpers:
    def test_denormalize_shape(self):
        # Create a normalised tensor (1, 3, 224, 224)
        tensor = torch.randn(3, 224, 224)
        result = denormalize(tensor)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_generate_overlay_shape(self):
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cam = np.random.rand(224, 224).astype(np.float32)
        overlay = generate_overlay(original, cam, alpha=0.4)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

    def test_save_gradcam_outputs(self, tmp_path):
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cam = np.random.rand(224, 224).astype(np.float32)
        cam = cam / cam.max()
        overlay = generate_overlay(original, cam)
        img_tensor = torch.randn(1, 3, 224, 224)

        paths = save_gradcam_outputs(
            "test_image.jpg", img_tensor, cam, overlay,
            output_dir=tmp_path / "out", stem="test_image",
        )
        assert paths["heatmap"].exists()
        assert paths["overlay"].exists()
        assert paths["heatmap"].name == "test_image_heatmap.png"
        assert paths["overlay"].name == "test_image_overlay.png"
