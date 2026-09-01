"""
Tests for src.eda — Exploratory Data Analysis Module.

Covers manifest loading, statistics computation, image property analysis,
invalid image detection, pixel statistics, visualization generation,
and report output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.eda import (
    CLASS_NAMES,
    SPLIT_NAMES,
    analyze_image_properties,
    compute_dataset_stats,
    compute_pixel_statistics,
    compute_split_class_distribution,
    detect_invalid_images,
    generate_report,
    generate_summary_json,
    generate_visualizations,
    load_manifests,
    run_eda,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_image(
    path: Path,
    size: tuple[int, int] = (64, 64),
    color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Create a simple test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(str(path))


def _create_split_manifests(
    tmp_path: Path,
    train_count: int = 10,
    val_count: int = 5,
    test_count: int = 5,
    classes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Create test split manifests with real images."""
    if classes is None:
        classes = ["Healthy", "Early Blight", "Late Blight"]

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, pd.DataFrame] = {}
    uid = 0
    for split, count in [("train", train_count), ("val", val_count), ("test", test_count)]:
        rows = []
        for i in range(count):
            cls = classes[i % len(classes)]
            img_path = tmp_path / "images" / f"img_{uid}.jpg"
            # Vary size and color to make images distinct
            w = 64 + (uid % 5) * 16
            h = 64 + (uid % 3) * 16
            r = (uid * 37) % 256
            g = (uid * 53) % 256
            b = (uid * 71) % 256
            _create_test_image(img_path, size=(w, h), color=(r, g, b))
            rows.append({"image_path": str(img_path), "clean_class": cls})
            uid += 1
        df = pd.DataFrame(rows)
        df.to_csv(splits_dir / f"{split}.csv", index=False)
        manifests[split] = df

    return manifests


# ---------------------------------------------------------------------------
# Manifest loading tests
# ---------------------------------------------------------------------------


class TestLoadManifests:
    def test_load_valid_manifests(self, tmp_path):
        manifests = _create_split_manifests(tmp_path)
        loaded = load_manifests(tmp_path / "splits")
        assert set(loaded.keys()) == set(SPLIT_NAMES)
        for name in SPLIT_NAMES:
            assert "image_path" in loaded[name].columns
            assert "clean_class" in loaded[name].columns

    def test_missing_manifest_raises(self, tmp_path):
        (tmp_path / "splits").mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            load_manifests(tmp_path / "splits")

    def test_missing_columns_raises(self, tmp_path):
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        # Create CSV with wrong columns
        pd.DataFrame({"wrong": [1, 2, 3]}).to_csv(splits_dir / "train.csv", index=False)
        pd.DataFrame({"image_path": ["a"], "clean_class": ["X"]}).to_csv(
            splits_dir / "val.csv", index=False
        )
        pd.DataFrame({"image_path": ["b"], "clean_class": ["Y"]}).to_csv(
            splits_dir / "test.csv", index=False
        )
        with pytest.raises(ValueError, match="missing required columns"):
            load_manifests(splits_dir)


# ---------------------------------------------------------------------------
# Dataset statistics tests
# ---------------------------------------------------------------------------


class TestComputeDatasetStats:
    def test_total_images(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=10, val_count=5, test_count=5)
        stats = compute_dataset_stats(manifests)
        assert stats["total_images"] == 20

    def test_class_counts(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=9, val_count=3, test_count=3)
        stats = compute_dataset_stats(manifests)
        # Each class gets 5 images (15 total / 3 classes)
        for cls in ["Healthy", "Early Blight", "Late Blight"]:
            assert stats["class_counts"].get(cls, 0) == 5

    def test_split_counts(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=10, val_count=5, test_count=3)
        stats = compute_dataset_stats(manifests)
        assert stats["split_counts"]["train"] == 10
        assert stats["split_counts"]["val"] == 5
        assert stats["split_counts"]["test"] == 3

    def test_class_percentages(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=6, val_count=3, test_count=3)
        stats = compute_dataset_stats(manifests)
        total = 12
        for cls, count in stats["class_counts"].items():
            expected_pct = count / total * 100
            assert abs(stats["class_percentages"][cls] - expected_pct) < 0.1


# ---------------------------------------------------------------------------
# Split x class distribution tests
# ---------------------------------------------------------------------------


class TestComputeSplitClassDistribution:
    def test_distribution_shape(self, tmp_path):
        manifests = _create_split_manifests(tmp_path)
        dist = compute_split_class_distribution(manifests)
        assert set(dist.index) == set(SPLIT_NAMES)
        assert len(dist.index) == 3

    def test_distribution_values(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=9, val_count=3, test_count=3)
        dist = compute_split_class_distribution(manifests)
        # Train has 3 of each class
        assert dist.loc["train", "Healthy"] == 3
        assert dist.loc["val", "Early Blight"] == 1


# ---------------------------------------------------------------------------
# Image property analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeImageProperties:
    def test_basic_properties(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=5, val_count=2, test_count=2)
        props = analyze_image_properties(manifests)
        assert props["analyzed_count"] == 9
        assert "widths" in props
        assert "heights" in props
        assert "aspect_ratios" in props
        assert props["min_width"] > 0
        assert props["max_width"] >= props["min_width"]

    def test_sample_size(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=10, val_count=5, test_count=5)
        props = analyze_image_properties(manifests, sample_size=5, seed=42)
        assert props["analyzed_count"] == 5

    def test_deterministic_sampling(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=10, val_count=5, test_count=5)
        props1 = analyze_image_properties(manifests, sample_size=5, seed=42)
        props2 = analyze_image_properties(manifests, sample_size=5, seed=42)
        assert props1["widths"] == props2["widths"]
        assert props1["heights"] == props2["heights"]

    def test_invalid_image_handling(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=3, val_count=1, test_count=1)
        # Add an invalid image to the first manifest
        bad_path = tmp_path / "images" / "bad.jpg"
        bad_path.write_bytes(b"not a real image")
        # Modify first manifest to include the bad image
        split_path = tmp_path / "splits" / "train.csv"
        df = pd.read_csv(split_path)
        df.loc[0, "image_path"] = str(bad_path)
        df.to_csv(split_path, index=False)
        manifests["train"] = df

        props = analyze_image_properties(manifests)
        assert len(props["invalid_images"]) == 1
        assert props["analyzed_count"] == 4  # 5 - 1 invalid

    def test_common_dimensions(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=5, val_count=2, test_count=2)
        props = analyze_image_properties(manifests)
        assert "common_dimensions" in props
        assert len(props["common_dimensions"]) > 0


# ---------------------------------------------------------------------------
# Invalid image detection tests
# ---------------------------------------------------------------------------


class TestDetectInvalidImages:
    def test_all_valid(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=3, val_count=2, test_count=2)
        invalid = detect_invalid_images(manifests)
        assert len(invalid) == 0

    def test_detects_corrupted(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=3, val_count=1, test_count=1)
        # Corrupt one image
        bad_path = tmp_path / "images" / "corrupt.jpg"
        bad_path.write_bytes(b"corrupted data")
        split_path = tmp_path / "splits" / "train.csv"
        df = pd.read_csv(split_path)
        df.loc[0, "image_path"] = str(bad_path)
        df.to_csv(split_path, index=False)
        manifests["train"] = df

        invalid = detect_invalid_images(manifests)
        assert len(invalid) == 1
        assert "corrupt" in invalid[0]["path"].lower() or "corrupt" in str(bad_path).lower()


# ---------------------------------------------------------------------------
# Pixel statistics tests
# ---------------------------------------------------------------------------


class TestComputePixelStatistics:
    def test_basic_stats(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=10, val_count=5, test_count=5)
        stats = compute_pixel_statistics(manifests, sample_size=5, seed=42)
        assert stats["sampled_count"] == 5
        assert "channel_mean" in stats
        assert "channel_std" in stats
        for ch in ["R", "G", "B"]:
            assert 0 <= stats["channel_mean"][ch] <= 255

    def test_deterministic(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=10, val_count=5, test_count=5)
        stats1 = compute_pixel_statistics(manifests, sample_size=5, seed=42)
        stats2 = compute_pixel_statistics(manifests, sample_size=5, seed=42)
        assert stats1["channel_mean"] == stats2["channel_mean"]


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestGenerateVisualizations:
    def test_generates_files(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=6, val_count=3, test_count=3)
        stats = compute_dataset_stats(manifests)
        dist = compute_split_class_distribution(manifests)
        props = analyze_image_properties(manifests)
        out_dir = tmp_path / "reports"

        viz_paths = generate_visualizations(stats, dist, props, manifests, out_dir, seed=42)
        assert len(viz_paths) >= 4  # at least 4 visualizations
        for p in viz_paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_deterministic_filenames(self, tmp_path):
        manifests = _create_split_manifests(tmp_path)
        stats = compute_dataset_stats(manifests)
        dist = compute_split_class_distribution(manifests)
        props = analyze_image_properties(manifests)
        out_dir = tmp_path / "reports"

        paths1 = generate_visualizations(stats, dist, props, manifests, out_dir, seed=42)
        paths2 = generate_visualizations(stats, dist, props, manifests, out_dir, seed=42)
        assert [p.name for p in paths1] == [p.name for p in paths2]


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_created(self, tmp_path):
        manifests = _create_split_manifests(tmp_path)
        stats = compute_dataset_stats(manifests)
        dist = compute_split_class_distribution(manifests)
        props = analyze_image_properties(manifests)
        pixel_stats = compute_pixel_statistics(manifests, sample_size=3, seed=42)
        invalid = detect_invalid_images(manifests)
        viz_paths = generate_visualizations(stats, dist, props, manifests, tmp_path / "viz", seed=42)

        report_path = tmp_path / "eda_report.txt"
        generate_report(stats, dist, props, pixel_stats, invalid, viz_paths, report_path)
        assert report_path.exists()
        text = report_path.read_text(encoding="utf-8")
        assert "AgriMind AI" in text
        assert "Total images:" in text
        assert "Images per class:" in text
        assert "Images per split:" in text


class TestGenerateSummaryJson:
    def test_json_created(self, tmp_path):
        manifests = _create_split_manifests(tmp_path)
        stats = compute_dataset_stats(manifests)
        dist = compute_split_class_distribution(manifests)
        props = analyze_image_properties(manifests)
        pixel_stats = compute_pixel_statistics(manifests, sample_size=3, seed=42)
        invalid = detect_invalid_images(manifests)

        summary_path = tmp_path / "eda_summary.json"
        generate_summary_json(stats, dist, props, pixel_stats, invalid, summary_path)
        assert summary_path.exists()

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "total_images" in data
        assert "class_counts" in data
        assert "split_counts" in data
        assert "image_properties" in data
        assert "pixel_statistics" in data


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


class TestRunEda:
    def test_full_pipeline(self, tmp_path):
        manifests = _create_split_manifests(tmp_path, train_count=6, val_count=3, test_count=3)
        output_dir = tmp_path / "reports"
        config_path = tmp_path / "config.yaml"
        config_path.write_text("seed: 42\n", encoding="utf-8")

        result = run_eda(
            splits_dir=tmp_path / "splits",
            output_dir=output_dir,
            config_path=config_path,
        )

        assert result["report_path"].exists()
        assert result["summary_path"].exists()
        assert len(result["viz_paths"]) >= 4
        assert result["stats"]["total_images"] == 12
