"""
Tests for src.model — MobileNetV3 Transfer Learning Model.

Covers architecture creation, output shape, forward pass,
pretrained behavior, backbone freeze/unfreeze, and checkpoint I/O.
"""

from __future__ import annotations

import pytest
import torch

from src.model import (
    ARCHITECTURE_NAME,
    build_model,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    unfreeze_backbone,
)


# ---------------------------------------------------------------------------
# Model builder tests
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_output_shape(self):
        model = build_model(num_classes=3, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 3)

    def test_custom_num_classes(self):
        model = build_model(num_classes=5, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 5)

    def test_architecture_name(self):
        model = build_model(num_classes=3, pretrained=False)
        assert model.architecture == ARCHITECTURE_NAME

    def test_num_classes_attr(self):
        model = build_model(num_classes=3, pretrained=False)
        assert model.num_classes == 3

    def test_forward_pass(self):
        model = build_model(num_classes=3, pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Backbone freeze/unfreeze tests
# ---------------------------------------------------------------------------


class TestBackboneFreeze:
    def test_freeze_backbone(self):
        model = build_model(num_classes=3, pretrained=False, freeze_backbone=True)
        # Backbone parameters should be frozen
        frozen = sum(
            1 for p in model.features.parameters() if not p.requires_grad
        )
        total = sum(1 for _ in model.features.parameters())
        assert frozen == total

        # Classifier parameters should be trainable
        trainable = sum(
            1 for p in model.classifier.parameters() if p.requires_grad
        )
        total_cls = sum(1 for _ in model.classifier.parameters())
        assert trainable == total_cls

    def test_unfreeze_backbone(self):
        model = build_model(num_classes=3, pretrained=False, freeze_backbone=True)
        unfreeze_backbone(model)
        # All parameters should now be trainable
        trainable = sum(
            1 for p in model.features.parameters() if p.requires_grad
        )
        total = sum(1 for _ in model.features.parameters())
        assert trainable == total


# ---------------------------------------------------------------------------
# Parameter counting tests
# ---------------------------------------------------------------------------


class TestCountParameters:
    def test_count_with_frozen(self):
        model = build_model(num_classes=3, pretrained=False, freeze_backbone=True)
        counts = count_parameters(model)
        assert counts["total"] > 0
        assert counts["trainable"] > 0
        assert counts["frozen"] > 0
        assert counts["total"] == counts["trainable"] + counts["frozen"]

    def test_count_all_trainable(self):
        model = build_model(num_classes=3, pretrained=False, freeze_backbone=False)
        counts = count_parameters(model)
        assert counts["frozen"] == 0
        assert counts["trainable"] == counts["total"]


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        model = build_model(num_classes=3, pretrained=False)
        path = str(tmp_path / "test_ckpt.pth")
        metadata = {"seed": 42, "best_val_accuracy": 0.95}
        save_checkpoint(model, path, metadata=metadata)

        loaded = load_checkpoint(path, num_classes=3, pretrained=False)
        assert loaded.num_classes == 3

        # Verify weights match
        x = torch.randn(1, 3, 224, 224)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            orig_out = model(x)
            loaded_out = loaded(x)
        assert torch.allclose(orig_out, loaded_out, atol=1e-6)

    def test_metadata_in_checkpoint(self, tmp_path):
        model = build_model(num_classes=3, pretrained=False)
        path = str(tmp_path / "meta_ckpt.pth")
        metadata = {"architecture": "mobilenet_v3_small", "seed": 42}
        save_checkpoint(model, path, metadata=metadata)

        ckpt = torch.load(path, weights_only=False)
        assert "metadata" in ckpt
        assert ckpt["metadata"]["seed"] == 42
        assert ckpt["architecture"] == ARCHITECTURE_NAME
        assert ckpt["num_classes"] == 3
