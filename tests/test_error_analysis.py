"""
Tests for src.error_analysis - Phase 3B Error Analysis.

Uses deterministic scripted models and synthetic data.
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

from src.dataset import IDX_TO_CLASS, NUM_CLASSES
from src.error_analysis import (
    ERROR_COLUMNS,
    compute_confidence_statistics,
    compute_confusion_statistics,
    generate_report,
    generate_summary_json,
    identify_errors,
    plot_confidence_histogram,
    plot_confusion_pairs,
    plot_error_grid,
    rank_high_confidence_errors,
    run_error_analysis,
    write_misclassified_csv,
)
from src.model import build_model

CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]


def _create_test_image(path, size=(64, 64), color=(100, 150, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(str(path))


class _MockDataset(Dataset):
    def __init__(self, images, labels, paths):
        self.images = images
        self.labels = labels
        self.paths = paths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.paths[idx]


class _ScriptedModel(torch.nn.Module):
    def __init__(self, scripted_preds, num_classes=3, logit=5.0):
        super().__init__()
        self.scripted = list(scripted_preds)
        self.num_classes = num_classes
        self.logit = logit
        self._idx = 0

    def forward(self, x):
        logits = torch.full(
            (x.size(0), self.num_classes), -self.logit / 2, device=x.device
        )
        for i in range(x.size(0)):
            p = self.scripted[self._idx % len(self.scripted)]
            logits[i, p] = self.logit
            self._idx += 1
        return logits


def _make_loader(labels, scripted, paths, image_size=64, batch_size=8):
    images = torch.randn(len(labels), 3, image_size, image_size)
    ds = _MockDataset(images, torch.tensor(labels), paths)
    return DataLoader(ds, batch_size=batch_size, shuffle=False), _ScriptedModel(scripted)


def _real_paths(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / "images" / f"img_{i}.jpg"
        _create_test_image(p)
        paths.append(str(p))
    return paths


# --- identify_errors ---

Y_TRUE = np.array([0, 1, 2])
Y_PRED = np.array([0, 2, 1])
PROBS = np.array([[0.90, 0.05, 0.05], [0.10, 0.20, 0.70], [0.05, 0.60, 0.35]])
PATHS = ["a.jpg", "b.jpg", "c.jpg"]


class TestIdentifyErrors:
    def test_no_errors(self):
        assert identify_errors(Y_TRUE, Y_TRUE, PROBS, PATHS) == []

    def test_all_mismatches_captured(self):
        assert len(identify_errors(Y_TRUE, Y_PRED, PROBS, PATHS)) == 2

    def test_error_fields(self):
        e = identify_errors(Y_TRUE, Y_PRED, PROBS, PATHS)[0]
        assert set(ERROR_COLUMNS).issubset(e.keys())
        assert e["true_class"] == "Early Blight"
        assert e["predicted_class"] == "Late Blight"

    def test_confidence_values(self):
        errors = identify_errors(Y_TRUE, Y_PRED, PROBS, PATHS)
        assert errors[0]["confidence"] == pytest.approx(0.70)
        assert errors[0]["true_class_probability"] == pytest.approx(0.20)
        assert errors[0]["margin"] == pytest.approx(0.50)
        assert errors[1]["confidence"] == pytest.approx(0.60)
        assert errors[1]["margin"] == pytest.approx(0.25)

    def test_image_path_preserved(self):
        errors = identify_errors(Y_TRUE, Y_PRED, PROBS, PATHS)
        assert errors[0]["image_path"] == "b.jpg"
        assert errors[1]["image_path"] == "c.jpg"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            identify_errors(Y_TRUE, Y_PRED[:1], PROBS, PATHS)


# --- compute_confusion_statistics ---

class TestConfusionStatistics:
    @staticmethod
    def _data():
        y_true = np.array([0, 1, 2, 2, 1])
        y_pred = np.array([0, 2, 1, 1, 1])
        probs = np.full((5, 3), 1 / 3)
        paths = [f"img_{i}.jpg" for i in range(5)]
        return identify_errors(y_true, y_pred, probs, paths), y_true

    def test_total_errors(self):
        errors, y_true = self._data()
        assert compute_confusion_statistics(errors, y_true)["total_errors"] == 3

    def test_pair_counts_sorted_desc(self):
        errors, y_true = self._data()
        counts = [p["count"] for p in compute_confusion_statistics(errors, y_true)["pair_counts"]]
        assert counts == sorted(counts, reverse=True)

    def test_most_confused_pair(self):
        errors, y_true = self._data()
        mc = compute_confusion_statistics(errors, y_true)["most_confused"]
        assert mc["true_class"] == "Late Blight"
        assert mc["predicted_class"] == "Early Blight"
        assert mc["count"] == 2

    def test_per_class_error_rates(self):
        errors, y_true = self._data()
        pc = compute_confusion_statistics(errors, y_true)["per_class"]
        assert pc["Healthy"]["support"] == 1
        assert pc["Healthy"]["errors"] == 0
        assert pc["Early Blight"]["support"] == 2
        assert pc["Early Blight"]["errors"] == 1
        assert pc["Early Blight"]["error_rate"] == pytest.approx(0.5)
        assert pc["Late Blight"]["errors"] == 2

    def test_no_errors_edge(self):
        stats = compute_confusion_statistics([], np.array([0, 1, 2]))
        assert stats["total_errors"] == 0
        assert stats["pair_counts"] == []
        assert stats["most_confused"] is None


# --- compute_confidence_statistics ---

class TestConfidenceStatistics:
    Y_TRUE = np.array([0, 1, 2, 0])
    Y_PRED = np.array([0, 1, 1, 0])
    PROBS = np.array([[0.90, 0.05, 0.05], [0.10, 0.80, 0.10], [0.05, 0.70, 0.25], [0.85, 0.10, 0.05]])

    def test_keys_present(self):
        stats = compute_confidence_statistics(self.Y_TRUE, self.Y_PRED, self.PROBS)
        for key in ["mean_confidence_correct", "mean_confidence_incorrect",
                     "median_confidence_correct", "median_confidence_incorrect",
                     "confidence_gap", "mean_true_class_probability_on_errors", "bins"]:
            assert key in stats

    def test_mean_confidences(self):
        stats = compute_confidence_statistics(self.Y_TRUE, self.Y_PRED, self.PROBS)
        assert stats["mean_confidence_correct"] == pytest.approx((0.90 + 0.80 + 0.85) / 3)
        assert stats["mean_confidence_incorrect"] == pytest.approx(0.70)

    def test_confidence_gap(self):
        stats = compute_confidence_statistics(self.Y_TRUE, self.Y_PRED, self.PROBS)
        assert stats["confidence_gap"] == pytest.approx(0.85 - 0.70)

    def test_true_class_prob_on_errors(self):
        stats = compute_confidence_statistics(self.Y_TRUE, self.Y_PRED, self.PROBS)
        assert stats["mean_true_class_probability_on_errors"] == pytest.approx(0.25)

    def test_bins_structure(self):
        stats = compute_confidence_statistics(self.Y_TRUE, self.Y_PRED, self.PROBS, n_bins=10)
        assert len(stats["bins"]) == 10
        assert sum(b["count"] for b in stats["bins"]) == 4

    def test_bins_accuracy(self):
        stats = compute_confidence_statistics(self.Y_TRUE, self.Y_PRED, self.PROBS, n_bins=10)
        by_range = {(round(b["lower"], 1), round(b["upper"], 1)): b for b in stats["bins"]}
        assert by_range[(0.6, 0.7)]["count"] == 1
        assert by_range[(0.6, 0.7)]["accuracy"] == pytest.approx(0.0)
        assert by_range[(0.8, 0.9)]["count"] == 2
        assert by_range[(0.8, 0.9)]["accuracy"] == pytest.approx(1.0)

    def test_no_errors_edge(self):
        stats = compute_confidence_statistics(self.Y_TRUE, self.Y_TRUE, self.PROBS)
        assert stats["mean_confidence_incorrect"] is None
        assert stats["confidence_gap"] is None
        assert stats["mean_true_class_probability_on_errors"] is None

    def test_empty_input(self):
        stats = compute_confidence_statistics(np.array([]), np.array([]), np.empty((0, 3)))
        assert stats["mean_confidence_correct"] is None
        assert stats["bins"] == []


# --- rank_high_confidence_errors ---

class TestRankHighConfidenceErrors:
    def test_sorted_descending(self):
        errors = [{"confidence": 0.5}, {"confidence": 0.9}, {"confidence": 0.7}]
        ranked = rank_high_confidence_errors(errors)
        assert [e["confidence"] for e in ranked] == [0.9, 0.7, 0.5]

    def test_top_k_limit(self):
        errors = [{"confidence": 0.5}, {"confidence": 0.9}, {"confidence": 0.7}]
        ranked = rank_high_confidence_errors(errors, top_k=2)
        assert len(ranked) == 2
        assert ranked[0]["confidence"] == 0.9

    def test_top_k_exceeds_length(self):
        ranked = rank_high_confidence_errors([{"confidence": 0.5}], top_k=10)
        assert len(ranked) == 1

    def test_empty(self):
        assert rank_high_confidence_errors([], top_k=5) == []


# --- Visualisations ---

class TestVisualisations:
    def test_confidence_histogram_created(self, tmp_path):
        p = plot_confidence_histogram(np.array([0.9, 0.8]), np.array([0.6, 0.7]), tmp_path / "hist.png")
        assert p.exists()

    def test_confusion_pairs_created(self, tmp_path):
        pairs = [{"true_class": "A", "predicted_class": "B", "count": 3},
                 {"true_class": "B", "predicted_class": "A", "count": 1}]
        assert plot_confusion_pairs(pairs, tmp_path / "pairs.png").exists()

    def test_confusion_pairs_empty(self, tmp_path):
        assert plot_confusion_pairs([], tmp_path / "pairs_empty.png").exists()

    def test_error_grid_created(self, tmp_path):
        paths = _real_paths(tmp_path, 3)
        errors = [{"image_path": p, "true_class": "Healthy",
                   "predicted_class": "Late Blight", "confidence": 0.9 - i * 0.1}
                  for i, p in enumerate(paths)]
        p = plot_error_grid(errors, tmp_path / "grid.png")
        assert p is not None and p.exists()

    def test_error_grid_tolerates_missing_images(self, tmp_path):
        errors = [{"image_path": str(tmp_path / "missing.jpg"), "true_class": "Healthy",
                   "predicted_class": "Late Blight", "confidence": 0.8}]
        p = plot_error_grid(errors, tmp_path / "grid_missing.png")
        assert p is not None and p.exists()

    def test_error_grid_empty_returns_none(self, tmp_path):
        assert plot_error_grid([], tmp_path / "grid_empty.png") is None


# --- misclassified CSV ---

class TestMisclassifiedCsv:
    def test_csv_written_sorted(self, tmp_path):
        errors = [
            {"image_path": "a.jpg", "true_class": "Healthy", "predicted_class": "Late Blight",
             "true_index": 0, "predicted_index": 2, "confidence": 0.6,
             "true_class_probability": 0.3, "margin": 0.3},
            {"image_path": "b.jpg", "true_class": "Early Blight", "predicted_class": "Late Blight",
             "true_index": 1, "predicted_index": 2, "confidence": 0.95,
             "true_class_probability": 0.03, "margin": 0.92},
        ]
        p = write_misclassified_csv(errors, tmp_path / "mis.csv")
        df = pd.read_csv(p)
        assert list(df.columns) == ERROR_COLUMNS
        assert len(df) == 2
        assert df["confidence"].iloc[0] == pytest.approx(0.95)

    def test_empty_csv_has_header(self, tmp_path):
        p = write_misclassified_csv([], tmp_path / "mis_empty.csv")
        df = pd.read_csv(p)
        assert list(df.columns) == ERROR_COLUMNS
        assert len(df) == 0


# --- Reports ---

def _make_analysis():
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 2, 1, 1, 1])
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.2, 0.7], [0.05, 0.6, 0.35],
                      [0.1, 0.55, 0.35], [0.2, 0.5, 0.3]])
    paths = [f"img_{i}.jpg" for i in range(5)]
    errors = identify_errors(y_true, y_pred, probs, paths)
    return {
        "generated_at": "2026-09-01 00:00:00", "checkpoint": "models/best_model.pth",
        "split": "test", "class_names": CLASS_NAMES, "total_samples": 5,
        "accuracy": 0.4, "total_errors": 3, "error_rate": 0.6, "errors": errors,
        "confusion_statistics": compute_confusion_statistics(errors, y_true),
        "confidence_statistics": compute_confidence_statistics(y_true, y_pred, probs),
        "high_confidence_errors": rank_high_confidence_errors(errors, top_k=2),
        "output_files": {"report": "reports/error_analysis/error_analysis_report.txt"},
    }


class TestReports:
    def test_report_contains_sections(self, tmp_path):
        p = generate_report(_make_analysis(), tmp_path / "report.txt")
        text = p.read_text(encoding="utf-8")
        assert "Error Analysis Report" in text
        assert "Total samples: 5" in text
        assert "Misclassified: 3" in text
        assert "Confusion Statistics" in text
        assert "Confidence Statistics" in text
        assert "Top High-Confidence Errors" in text
        assert "Late Blight -> Early Blight" in text

    def test_report_no_errors(self, tmp_path):
        analysis = _make_analysis()
        analysis["total_errors"] = 0
        analysis["errors"] = []
        analysis["high_confidence_errors"] = []
        analysis["confusion_statistics"] = compute_confusion_statistics([], np.array([0, 1, 2]))
        p = generate_report(analysis, tmp_path / "report_none.txt")
        assert "none (no misclassifications)" in p.read_text(encoding="utf-8")

    def test_summary_json_keys(self, tmp_path):
        p = generate_summary_json(_make_analysis(), tmp_path / "summary.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["total_samples"] == 5
        assert data["total_errors"] == 3
        assert "errors" not in data
        assert len(data["high_confidence_errors"]) == 2
        assert "confusion_statistics" in data
        assert "confidence_statistics" in data


# --- Full pipeline ---

class TestRunErrorAnalysis:
    def test_pipeline_with_errors(self, tmp_path):
        labels = [0, 1, 2, 0, 1, 2]
        scripted = [0, 2, 1, 0, 2, 2]  # errors at idx 1, 2, 4
        paths = _real_paths(tmp_path, 6)
        loader, model = _make_loader(labels, scripted, paths)
        analysis = run_error_analysis(model, loader, device="cpu",
                                      output_dir=tmp_path / "out", top_k=2, max_grid_images=4)
        assert analysis["total_samples"] == 6
        assert analysis["total_errors"] == 3
        assert analysis["error_rate"] == pytest.approx(0.5)
        files = analysis["output_files"]
        for key in ["misclassified_csv", "confidence_histogram", "confusion_pairs",
                     "error_grid", "report", "summary_json"]:
            assert key in files
            assert Path(files[key]).exists()
        df = pd.read_csv(files["misclassified_csv"])
        assert len(df) == 3
        assert len(analysis["high_confidence_errors"]) == 2

    def test_pipeline_perfect_predictions(self, tmp_path):
        labels = [0, 1, 2, 0, 1, 2]
        paths = _real_paths(tmp_path, 6)
        loader, model = _make_loader(labels, list(labels), paths)
        analysis = run_error_analysis(model, loader, device="cpu", output_dir=tmp_path / "out")
        assert analysis["total_errors"] == 0
        assert analysis["accuracy"] == pytest.approx(1.0)
        assert "error_grid" not in analysis["output_files"]
        assert Path(analysis["output_files"]["report"]).exists()
        assert Path(analysis["output_files"]["summary_json"]).exists()

    def test_pipeline_all_wrong(self, tmp_path):
        labels = [0, 0, 0]
        scripted = [1, 1, 2]
        paths = _real_paths(tmp_path, 3)
        loader, model = _make_loader(labels, scripted, paths)
        analysis = run_error_analysis(model, loader, device="cpu", output_dir=tmp_path / "out")
        assert analysis["total_errors"] == 3
        assert analysis["accuracy"] == pytest.approx(0.0)
        mc = analysis["confusion_statistics"]["most_confused"]
        assert mc["true_class"] == "Healthy"
        assert mc["predicted_class"] == "Early Blight"
        assert mc["count"] == 2

    def test_pipeline_with_real_model(self, tmp_path):
        torch.manual_seed(42)
        labels = [0, 1, 2, 0, 1, 2]
        paths = _real_paths(tmp_path, 6)
        images = torch.randn(6, 3, 64, 64)
        ds = _MockDataset(images, torch.tensor(labels), paths)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = build_model(num_classes=3, pretrained=False)
        model.eval()
        analysis = run_error_analysis(model, loader, device="cpu", output_dir=tmp_path / "out")
        assert analysis["total_samples"] == 6
        assert 0 <= analysis["total_errors"] <= 6
        assert Path(analysis["output_files"]["summary_json"]).exists()
