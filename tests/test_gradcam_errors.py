"""
Tests for src.gradcam_errors — Grad-CAM Error Investigation.

Uses a tiny scripted model with a features attribute so the existing
GradCAM engine can hook ``model.features[-1]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.gradcam_errors import (
    _pair_dir_name,
    _save_image_outputs,
    analyze_error_gradcam,
    generate_error_grid,
    generate_error_report,
    run_gradcam_on_errors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model with .features[-1] for Grad-CAM hook compatibility."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def _create_test_image(path: Path, size=(32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 150, 200)).save(str(path))


def _make_errors_csv(tmp_path, n=3, pairs=None):
    """Create a minimal misclassified CSV."""
    if pairs is None:
        pairs = [("Early Blight", "Late Blight", 1, 2)] * n
    rows = []
    for i, (true_cls, pred_cls, true_idx, pred_idx) in enumerate(pairs):
        img_path = tmp_path / "images" / f"img_{i}.jpg"
        _create_test_image(img_path)
        rows.append(
            {
                "image_path": str(img_path),
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "true_index": true_idx,
                "predicted_index": pred_idx,
                "confidence": 0.9 - i * 0.05,
                "true_class_probability": 0.1,
                "margin": 0.8 - i * 0.05,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "errors.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# _pair_dir_name
# ---------------------------------------------------------------------------


class TestPairDirName:
    def test_basic(self):
        assert _pair_dir_name("Early Blight", "Late Blight") == (
            "Early_Blight_to_Late_Blight"
        )

    def test_single_word(self):
        assert _pair_dir_name("Healthy", "Late Blight") == (
            "Healthy_to_Late_Blight"
        )


# ---------------------------------------------------------------------------
# analyze_error_gradcam
# ---------------------------------------------------------------------------


class TestAnalyzeErrorGradcam:
    def test_returns_dual_cams(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path)
        model = _TinyModel()
        model.eval()

        result = analyze_error_gradcam(
            model, img_path, true_index=1, predicted_index=2,
            device="cpu", image_size=32,
        )

        assert result["true_index"] == 1
        assert result["predicted_index"] == 2
        assert result["true_class"] == "Early Blight"
        assert result["predicted_class"] == "Late Blight"
        assert isinstance(result["pred_cam"], np.ndarray)
        assert isinstance(result["true_cam"], np.ndarray)
        assert result["pred_cam"].shape == (32, 32)
        assert result["true_cam"].shape == (32, 32)

    def test_overlays_are_uint8(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path)
        model = _TinyModel()
        model.eval()

        result = analyze_error_gradcam(
            model, img_path, true_index=0, predicted_index=1,
            device="cpu", image_size=32,
        )

        assert result["pred_overlay"].dtype == np.uint8
        assert result["true_overlay"].dtype == np.uint8
        assert result["pred_overlay"].shape[2] == 3
        assert result["original"].shape[2] == 3

    def test_probabilities_sum_to_one(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path)
        model = _TinyModel()
        model.eval()

        result = analyze_error_gradcam(
            model, img_path, true_index=1, predicted_index=2,
            device="cpu", image_size=32,
        )

        probs = result["probabilities"]
        assert set(probs.keys()) == {"Healthy", "Early Blight", "Late Blight"}
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)

    def test_hooks_cleaned_up(self, tmp_path):
        """Verify hooks are removed after analysis."""
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path)
        model = _TinyModel()
        model.eval()

        analyze_error_gradcam(
            model, img_path, true_index=1, predicted_index=2,
            device="cpu", image_size=32,
        )

        target = model.features[-1]
        assert len(target._forward_hooks) == 0
        assert len(target._backward_hooks) == 0


# ---------------------------------------------------------------------------
# _save_image_outputs
# ---------------------------------------------------------------------------


class TestSaveImageOutputs:
    def _make_result(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path)
        model = _TinyModel()
        model.eval()
        return analyze_error_gradcam(
            model, img_path, true_index=1, predicted_index=2,
            device="cpu", image_size=32,
        )

    def test_creates_four_files(self, tmp_path):
        result = self._make_result(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        paths = _save_image_outputs(result, out_dir, "01_test")

        assert len(paths) == 4
        for key in ["pred_heatmap", "pred_overlay", "true_heatmap", "true_overlay"]:
            assert key in paths
            assert Path(paths[key]).exists()

    def test_overlay_files_are_valid_images(self, tmp_path):
        result = self._make_result(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        paths = _save_image_outputs(result, out_dir, "01_test")

        img = Image.open(paths["pred_overlay"])
        assert img.mode == "RGB"
        img = Image.open(paths["true_overlay"])
        assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# generate_error_grid
# ---------------------------------------------------------------------------


class TestGenerateErrorGrid:
    def _make_results(self, tmp_path, n=3):
        results = []
        for i in range(n):
            img_path = tmp_path / f"img_{i}.jpg"
            _create_test_image(img_path)
            model = _TinyModel()
            model.eval()
            r = analyze_error_gradcam(
                model, img_path, true_index=1, predicted_index=2,
                device="cpu", image_size=32,
            )
            r["pair_key"] = "Early_Blight_to_Late_Blight"
            r["rank_in_pair"] = i + 1
            results.append(r)
        return results

    def test_grid_created(self, tmp_path):
        results = self._make_results(tmp_path)
        by_pair = {"Early Blight_to_Late Blight": results}
        path = generate_error_grid(by_pair, tmp_path / "grid.png")
        assert path.exists()

    def test_grid_multiple_pairs(self, tmp_path):
        r1 = self._make_results(tmp_path, n=2)
        r2 = self._make_results(tmp_path, n=1)
        by_pair = {
            "Early Blight_to_Late Blight": r1,
            "Late Blight_to_Early Blight": r2,
        }
        path = generate_error_grid(by_pair, tmp_path / "grid2.png")
        assert path.exists()

    def test_grid_empty_returns_file(self, tmp_path):
        path = generate_error_grid({}, tmp_path / "empty_grid.png")
        assert path.exists()


# ---------------------------------------------------------------------------
# generate_error_report
# ---------------------------------------------------------------------------


class TestGenerateErrorReport:
    def _make_results(self, tmp_path, n=2):
        results = []
        for i in range(n):
            img_path = tmp_path / f"img_{i}.jpg"
            _create_test_image(img_path)
            model = _TinyModel()
            model.eval()
            r = analyze_error_gradcam(
                model, img_path, true_index=1, predicted_index=2,
                device="cpu", image_size=32,
            )
            r["pair_key"] = "Early_Blight_to_Late_Blight"
            r["rank_in_pair"] = i + 1
            r["output_files"] = {"pred_overlay": "some/path.png"}
            results.append(r)
        return results

    def test_json_created_with_keys(self, tmp_path):
        results = self._make_results(tmp_path)
        paths = generate_error_report(results, tmp_path)

        assert "summary_json" in paths
        data = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
        assert data["total_errors_analyzed"] == 2
        assert len(data["errors"]) == 2
        assert "true_class" in data["errors"][0]
        assert "probabilities" in data["errors"][0]

    def test_txt_report_sections(self, tmp_path):
        results = self._make_results(tmp_path)
        paths = generate_error_report(results, tmp_path)

        text = paths["report"].read_text(encoding="utf-8")
        assert "Grad-CAM Error Investigation Report" in text
        assert "Early Blight -> Late Blight" in text
        assert "Output Files" in text


# ---------------------------------------------------------------------------
# run_gradcam_on_errors (full pipeline)
# ---------------------------------------------------------------------------


class TestRunGradcamOnErrors:
    def test_full_pipeline(self, tmp_path):
        pairs = [
            ("Early Blight", "Late Blight", 1, 2),
            ("Early Blight", "Late Blight", 1, 2),
            ("Late Blight", "Early Blight", 2, 1),
        ]
        csv_path = _make_errors_csv(tmp_path, pairs=pairs)
        model = _TinyModel()

        analysis = run_gradcam_on_errors(
            model, errors_csv=csv_path,
            output_dir=tmp_path / "gradcam_out",
            device="cpu", image_size=32,
        )

        assert analysis["total_analyzed"] == 3
        assert "Early_Blight_to_Late_Blight" in analysis["results_by_pair"]
        assert "Late_Blight_to_Early_Blight" in analysis["results_by_pair"]
        assert len(analysis["results_by_pair"]["Early_Blight_to_Late_Blight"]) == 2
        assert len(analysis["results_by_pair"]["Late_Blight_to_Early_Blight"]) == 1

        # Check output files exist
        assert Path(analysis["report_paths"]["summary_json"]).exists()
        assert Path(analysis["report_paths"]["report"]).exists()
        assert analysis["grid_path"] is not None
        assert Path(analysis["grid_path"]).exists()

        # Check per-image outputs
        for r in analysis["results"]:
            for key in ["pred_heatmap", "pred_overlay", "true_heatmap", "true_overlay"]:
                assert key in r["output_files"]
                assert Path(r["output_files"][key]).exists()

    def test_top_k_limit(self, tmp_path):
        pairs = [("Early Blight", "Late Blight", 1, 2)] * 5
        csv_path = _make_errors_csv(tmp_path, pairs=pairs)
        model = _TinyModel()

        analysis = run_gradcam_on_errors(
            model, errors_csv=csv_path,
            output_dir=tmp_path / "gradcam_out",
            device="cpu", image_size=32, top_k=2,
        )

        assert analysis["total_analyzed"] == 2

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        pd.DataFrame(
            columns=[
                "image_path", "true_class", "predicted_class",
                "true_index", "predicted_index", "confidence",
                "true_class_probability", "margin",
            ]
        ).to_csv(csv_path, index=False)
        model = _TinyModel()

        analysis = run_gradcam_on_errors(
            model, errors_csv=csv_path,
            output_dir=tmp_path / "gradcam_out",
            device="cpu", image_size=32,
        )

        assert analysis["total_analyzed"] == 0
        assert analysis["grid_path"] is None

    def test_missing_image_skipped(self, tmp_path):
        """Missing images should be skipped, not crash the pipeline."""
        csv_path = tmp_path / "errors.csv"
        pd.DataFrame(
            [
                {
                    "image_path": str(tmp_path / "nonexistent.jpg"),
                    "true_class": "Early Blight",
                    "predicted_class": "Late Blight",
                    "true_index": 1,
                    "predicted_index": 2,
                    "confidence": 0.9,
                    "true_class_probability": 0.1,
                    "margin": 0.8,
                }
            ]
        ).to_csv(csv_path, index=False)
        model = _TinyModel()

        analysis = run_gradcam_on_errors(
            model, errors_csv=csv_path,
            output_dir=tmp_path / "gradcam_out",
            device="cpu", image_size=32,
        )

        assert analysis["total_analyzed"] == 0

    def test_pair_subdirectories_created(self, tmp_path):
        pairs = [
            ("Early Blight", "Late Blight", 1, 2),
            ("Late Blight", "Early Blight", 2, 1),
            ("Early Blight", "Healthy", 1, 0),
        ]
        csv_path = _make_errors_csv(tmp_path, pairs=pairs)
        model = _TinyModel()
        out_dir = tmp_path / "gradcam_out"

        run_gradcam_on_errors(
            model, errors_csv=csv_path,
            output_dir=out_dir, device="cpu", image_size=32,
        )

        assert (out_dir / "Early_Blight_to_Late_Blight").is_dir()
        assert (out_dir / "Late_Blight_to_Early_Blight").is_dir()
        assert (out_dir / "Early_Blight_to_Healthy").is_dir()
