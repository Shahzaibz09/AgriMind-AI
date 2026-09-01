"""
Tests for src.prepare_dataset — Dataset Preparation Pipeline.

Covers helper functions, validation logic, duplicate detection,
split generation, and report writing using temporary test images.
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.prepare_dataset import (
    CLASS_MAP,
    VALID_EXTENSIONS,
    check_class_dirs,
    check_dataset_exists,
    compute_md5,
    find_duplicates,
    generate_splits,
    generate_summary,
    is_verified_leaf_id,
    load_config,
    load_source_metadata,
    scan_dataset,
    validate_image,
    write_dataset_report,
    write_metadata_csv,
    write_split_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_test_image(
    path: Path,
    size: tuple[int, int] = (64, 64),
    color: tuple[int, int, int] = (255, 0, 0),
    mode: str = "RGB",
    unique_id: int | None = None,
) -> None:
    """Create a simple test image. If unique_id is set, a pixel is varied to ensure unique file bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size, color)
    if unique_id is not None:
        # Draw a unique pixel so each image has different bytes / MD5
        img.putpixel((0, 0), ((unique_id * 7) % 256, (unique_id * 13) % 256, (unique_id * 23) % 256))
    img.save(str(path))


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal dataset directory with test images in 3 classes."""
    base = tmp_path / "tomato"
    classes = {
        "Tomato___healthy": (255, 0, 0),
        "Tomato___Early_blight": (0, 255, 0),
        "Tomato___Late_blight": (0, 0, 255),
    }
    for class_name, color in classes.items():
        class_dir = base / class_name
        class_dir.mkdir(parents=True)
        for i in range(10):
            _create_test_image(
                class_dir / f"img_{i:03d}.jpg",
                color=color,
                unique_id=i,
            )
    return base


# ---------------------------------------------------------------------------
# compute_md5 tests
# ---------------------------------------------------------------------------


class TestComputeMD5:
    def test_known_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert compute_md5(f) == expected

    def test_identical_files_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"same content")
        f2.write_bytes(b"same content")
        assert compute_md5(f1) == compute_md5(f2)

    def test_different_files_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_md5(f1) != compute_md5(f2)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.md5(b"").hexdigest()
        assert compute_md5(f) == expected


# ---------------------------------------------------------------------------
# validate_image tests
# ---------------------------------------------------------------------------


class TestValidateImage:
    def test_valid_rgb_image(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path)
        result = validate_image(img_path)
        assert result is not None
        assert result["image_mode"] == "RGB"
        assert result["width"] == 64
        assert result["height"] == 64
        assert result["file_extension"] == ".jpg"
        assert result["image_path"] == str(img_path)
        assert len(result["md5"]) == 32

    def test_valid_png_image(self, tmp_path):
        img_path = tmp_path / "test.png"
        _create_test_image(img_path)
        result = validate_image(img_path)
        assert result is not None
        assert result["file_extension"] == ".png"

    def test_corrupted_file_returns_none(self, tmp_path):
        img_path = tmp_path / "bad.jpg"
        img_path.write_bytes(b"this is not a real image file at all")
        result = validate_image(img_path)
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        img_path = tmp_path / "empty.jpg"
        img_path.write_bytes(b"")
        result = validate_image(img_path)
        assert result is None

    def test_custom_dimensions(self, tmp_path):
        img_path = tmp_path / "big.jpg"
        _create_test_image(img_path, size=(128, 256))
        result = validate_image(img_path)
        assert result is not None
        assert result["width"] == 128
        assert result["height"] == 256


# ---------------------------------------------------------------------------
# is_verified_leaf_id tests
# ---------------------------------------------------------------------------


class TestIsVerifiedLeafId:
    def test_valid_leaf_id(self):
        assert is_verified_leaf_id("leaf_A") is True
        assert is_verified_leaf_id("Tomato___healthy:::1") is True
        assert is_verified_leaf_id("Blueberry___healthy:::42") is True

    def test_unknown_not_verified(self):
        assert is_verified_leaf_id("unknown") is False
        assert is_verified_leaf_id("Unknown") is False
        assert is_verified_leaf_id("UNKNOWN") is False

    def test_fallback_not_verified(self):
        assert is_verified_leaf_id("fallback_abc123") is False
        assert is_verified_leaf_id("fallback_xyz") is False
        assert is_verified_leaf_id("Fallback_XYZ") is False

    def test_empty_string_not_verified(self):
        assert is_verified_leaf_id("") is False

    def test_regular_leaf_ids_are_verified(self):
        assert is_verified_leaf_id("leaf_1") is True
        assert is_verified_leaf_id("leaf_group_42") is True


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_valid_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "seed: 99\ndata:\n  raw: my/raw\n  processed: my/proc\n  splits: my/splits\n",
            encoding="utf-8",
        )
        config = load_config(str(config_file))
        assert config["seed"] == 99
        assert config["data"]["raw"] == "my/raw"

    def test_missing_config_returns_defaults(self):
        config = load_config("nonexistent/path/config.yaml")
        assert config["seed"] == 42
        assert "data" in config


# ---------------------------------------------------------------------------
# check_dataset_exists tests
# ---------------------------------------------------------------------------


class TestCheckDatasetExists:
    def test_existing_directory(self, sample_dataset):
        assert check_dataset_exists(sample_dataset) is True

    def test_nonexistent_directory(self, tmp_path):
        fake = tmp_path / "does_not_exist" / "tomato"
        assert check_dataset_exists(fake) is False


# ---------------------------------------------------------------------------
# check_class_dirs tests
# ---------------------------------------------------------------------------


class TestCheckClassDirs:
    def test_all_present(self, sample_dataset):
        assert check_class_dirs(sample_dataset) is True

    def test_missing_class(self, tmp_path):
        base = tmp_path / "tomato"
        base.mkdir()
        (base / "Tomato___healthy").mkdir()
        # Missing Early_blight and Late_blight
        assert check_class_dirs(base) is False


# ---------------------------------------------------------------------------
# scan_dataset tests
# ---------------------------------------------------------------------------


class TestScanDataset:
    def test_scan_counts(self, sample_dataset):
        records, corrupted = scan_dataset(sample_dataset)
        assert len(records) == 30  # 10 per class x 3 classes
        assert len(corrupted) == 0

    def test_scan_class_labels(self, sample_dataset):
        records, _ = scan_dataset(sample_dataset)
        classes = {r["clean_class"] for r in records}
        assert classes == {"Healthy", "Early Blight", "Late Blight"}

    def test_scan_records_have_required_fields(self, sample_dataset):
        records, _ = scan_dataset(sample_dataset)
        required = {
            "image_path", "raw_class", "clean_class",
            "file_extension", "image_mode", "width", "height", "md5",
        }
        for record in records:
            assert required.issubset(set(record.keys()))

    def test_corrupted_image_logged(self, sample_dataset):
        bad_file = sample_dataset / "Tomato___healthy" / "corrupt.jpg"
        bad_file.write_bytes(b"not a real image")
        records, corrupted = scan_dataset(sample_dataset)
        assert len(corrupted) == 1
        assert str(bad_file) in corrupted

    def test_unsupported_format_skipped(self, sample_dataset):
        txt_file = sample_dataset / "Tomato___healthy" / "readme.txt"
        txt_file.write_text("hello")
        records, corrupted = scan_dataset(sample_dataset)
        # 30 valid images, txt file is skipped (not corrupted)
        assert len(records) == 30
        assert len(corrupted) == 0


# ---------------------------------------------------------------------------
# find_duplicates tests
# ---------------------------------------------------------------------------


class TestFindDuplicates:
    def test_no_duplicates(self):
        records = [
            {"image_path": "a.jpg", "md5": "aaa"},
            {"image_path": "b.jpg", "md5": "bbb"},
            {"image_path": "c.jpg", "md5": "ccc"},
        ]
        result = find_duplicates(records)
        assert len(result) == 0

    def test_exact_duplicates_detected(self):
        records = [
            {"image_path": "a.jpg", "md5": "same_hash"},
            {"image_path": "b.jpg", "md5": "same_hash"},
            {"image_path": "c.jpg", "md5": "unique"},
        ]
        result = find_duplicates(records)
        assert len(result) == 1
        assert "same_hash" in result
        assert len(result["same_hash"]) == 2

    def test_multiple_duplicate_groups(self):
        records = [
            {"image_path": "a1.jpg", "md5": "group1"},
            {"image_path": "a2.jpg", "md5": "group1"},
            {"image_path": "b1.jpg", "md5": "group2"},
            {"image_path": "b2.jpg", "md5": "group2"},
            {"image_path": "c.jpg", "md5": "solo"},
        ]
        result = find_duplicates(records)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# generate_summary tests
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    def test_basic_summary(self):
        records = [
            {
                "image_path": "a.jpg", "raw_class": "Tomato___healthy",
                "clean_class": "Healthy", "file_extension": ".jpg",
                "image_mode": "RGB", "width": 64, "height": 64,
                "md5": "hash_a",
            },
            {
                "image_path": "b.jpg", "raw_class": "Tomato___Early_blight",
                "clean_class": "Early Blight", "file_extension": ".jpg",
                "image_mode": "RGB", "width": 64, "height": 64,
                "md5": "hash_b",
            },
        ]
        summary = generate_summary(records, corrupted=[], duplicates={})
        assert summary["total_images"] == 2
        assert summary["corrupted_count"] == 0
        assert summary["duplicate_extra_files"] == 0
        assert "Healthy" in summary["class_counts"]
        assert "Early Blight" in summary["class_counts"]

    def test_with_corrupted_and_duplicates(self):
        records = [
            {
                "image_path": "a.jpg", "raw_class": "Tomato___healthy",
                "clean_class": "Healthy", "file_extension": ".jpg",
                "image_mode": "RGB", "width": 64, "height": 64,
                "md5": "hash_a",
            },
        ]
        corrupted = ["bad.jpg"]
        duplicates = {"hash_x": ["dup1.jpg", "dup2.jpg"]}
        summary = generate_summary(records, corrupted, duplicates)
        assert summary["corrupted_count"] == 1
        assert summary["duplicate_groups"] == 1
        assert summary["duplicate_extra_files"] == 1


# ---------------------------------------------------------------------------
# write_metadata_csv tests
# ---------------------------------------------------------------------------


class TestWriteMetadataCSV:
    def test_csv_written_with_correct_columns(self, tmp_path):
        records = [
            {
                "image_path": "img.jpg", "raw_class": "Tomato___healthy",
                "clean_class": "Healthy", "file_extension": ".jpg",
                "image_mode": "RGB", "width": 64, "height": 64,
                "md5": "abc123",
            },
        ]
        out = tmp_path / "meta.csv"
        write_metadata_csv(records, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 1
        expected_cols = {
            "image_path", "raw_class", "clean_class", "file_extension",
            "image_mode", "width", "height", "md5",
        }
        assert expected_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# write_dataset_report tests
# ---------------------------------------------------------------------------


class TestWriteDatasetReport:
    def test_report_contains_key_sections(self, tmp_path):
        summary = {
            "total_images": 100,
            "class_counts": {"Healthy": 40, "Early Blight": 30, "Late Blight": 30},
            "class_percentages": {
                "Healthy": 40.0, "Early Blight": 30.0, "Late Blight": 30.0,
            },
            "corrupted_count": 2,
            "corrupted_files": ["bad1.jpg", "bad2.jpg"],
            "duplicate_groups": 1,
            "duplicate_extra_files": 1,
            "format_counts": {".jpg": 100},
            "mode_counts": {"RGB": 100},
        }
        corrupted = ["bad1.jpg", "bad2.jpg"]
        duplicates = {"md5hash": ["dup1.jpg", "dup2.jpg"]}
        out = tmp_path / "report.txt"
        write_dataset_report(summary, corrupted, duplicates, out)
        text = out.read_text(encoding="utf-8")
        assert "AgriMind AI" in text
        assert "PlantVillage" in text
        assert "Healthy: 40" in text
        assert "bad1.jpg" in text
        assert "dup1.jpg" in text
        assert "NOT modified" in text


# ---------------------------------------------------------------------------
# generate_splits tests
# ---------------------------------------------------------------------------


class TestGenerateSplits:
    def _make_records(self, tmp_path, per_class=20):
        """Generate unique records with real image files."""
        records = []
        classes = [
            ("Tomato___healthy", "Healthy", (255, 0, 0)),
            ("Tomato___Early_blight", "Early Blight", (0, 255, 0)),
            ("Tomato___Late_blight", "Late Blight", (0, 0, 255)),
        ]
        for raw, clean, color in classes:
            class_dir = tmp_path / raw
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                img_path = class_dir / f"img_{i:03d}.jpg"
                _create_test_image(img_path, color=color, unique_id=i)
                records.append({
                    "image_path": str(img_path),
                    "raw_class": raw,
                    "clean_class": clean,
                    "file_extension": ".jpg",
                    "image_mode": "RGB",
                    "width": 64,
                    "height": 64,
                    "md5": compute_md5(img_path),
                })
        return records

    def test_split_ratios_approximate(self, tmp_path):
        records = self._make_records(tmp_path, per_class=30)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, duplicates={}, seed=42, splits_dir=splits_dir
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 90
        assert 0.60 <= len(train_df) / total <= 0.80
        assert 0.10 <= len(val_df) / total <= 0.25
        assert 0.10 <= len(test_df) / total <= 0.25

    def test_no_overlap_between_splits(self, tmp_path):
        records = self._make_records(tmp_path, per_class=20)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, duplicates={}, seed=42, splits_dir=splits_dir
        )
        train_set = set(train_df["image_path"])
        val_set = set(val_df["image_path"])
        test_set = set(test_df["image_path"])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_split_csv_files_created(self, tmp_path):
        records = self._make_records(tmp_path, per_class=20)
        splits_dir = tmp_path / "splits"
        generate_splits(records, duplicates={}, seed=42, splits_dir=splits_dir)
        assert (splits_dir / "train.csv").exists()
        assert (splits_dir / "val.csv").exists()
        assert (splits_dir / "test.csv").exists()

    def test_split_csv_has_correct_columns(self, tmp_path):
        records = self._make_records(tmp_path, per_class=20)
        splits_dir = tmp_path / "splits"
        generate_splits(records, duplicates={}, seed=42, splits_dir=splits_dir)
        df = pd.read_csv(splits_dir / "train.csv")
        assert set(df.columns) == {"image_path", "clean_class"}

    def test_duplicates_removed_before_split(self, tmp_path):
        records = self._make_records(tmp_path, per_class=10)
        # Inject a duplicate: same MD5 as the first record
        records.append({
            "image_path": str(tmp_path / "duplicate.jpg"),
            "raw_class": "Tomato___healthy",
            "clean_class": "Healthy",
            "file_extension": ".jpg",
            "image_mode": "RGB",
            "width": 64,
            "height": 64,
            "md5": records[0]["md5"],  # exact MD5 duplicate
        })
        duplicates = {records[0]["md5"]: [records[0]["image_path"], str(tmp_path / "duplicate.jpg")]}
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, duplicates=duplicates, seed=42, splits_dir=splits_dir
        )
        total = len(train_df) + len(val_df) + len(test_df)
        # Should equal the number of unique MD5s (dedup removes extras)
        expected_unique = len({r["md5"] for r in records})
        assert total == expected_unique

    def test_all_classes_present_in_splits(self, tmp_path):
        records = self._make_records(tmp_path, per_class=20)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, duplicates={}, seed=42, splits_dir=splits_dir
        )
        expected_classes = {"Healthy", "Early Blight", "Late Blight"}
        assert set(train_df["clean_class"]) == expected_classes
        assert set(val_df["clean_class"]) == expected_classes
        assert set(test_df["clean_class"]) == expected_classes

    # ---- leaf-grouping tests ----

    @staticmethod
    def _make_leaf_records(tmp_path):
        """Create records with leaf_id grouping (2 images per leaf, 6 leaves per class)."""
        records = []
        classes = [
            ("Tomato___healthy", "Healthy", (255, 0, 0)),
            ("Tomato___Early_blight", "Early Blight", (0, 255, 0)),
            ("Tomato___Late_blight", "Late Blight", (0, 0, 255)),
        ]
        uid = 0
        for raw, clean, color in classes:
            class_dir = tmp_path / raw
            class_dir.mkdir(parents=True, exist_ok=True)
            for leaf_idx in range(6):
                leaf_id = f"leaf_{raw}_{leaf_idx}"
                for img_idx in range(2):
                    img_path = class_dir / f"leaf{leaf_idx}_img{img_idx}.jpg"
                    _create_test_image(img_path, color=color, unique_id=uid)
                    uid += 1
                    records.append({
                        "image_path": str(img_path.resolve()),
                        "raw_class": raw,
                        "clean_class": clean,
                        "file_extension": ".jpg",
                        "image_mode": "RGB",
                        "width": 64,
                        "height": 64,
                        "md5": compute_md5(img_path),
                        "leaf_id": leaf_id,
                    })
        leaf_id_map = {r["image_path"]: r["leaf_id"] for r in records}
        return records, leaf_id_map

    def test_same_leaf_not_in_multiple_splits(self, tmp_path):
        records, leaf_id_map = self._make_leaf_records(tmp_path)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        train_leaves = set(leaf_id_map[p] for p in train_df["image_path"])
        val_leaves = set(leaf_id_map[p] for p in val_df["image_path"])
        test_leaves = set(leaf_id_map[p] for p in test_df["image_path"])
        assert len(train_leaves & val_leaves) == 0
        assert len(train_leaves & test_leaves) == 0
        assert len(val_leaves & test_leaves) == 0

    def test_all_images_of_leaf_in_same_split(self, tmp_path):
        records, leaf_id_map = self._make_leaf_records(tmp_path)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        train_paths = set(train_df["image_path"])
        val_paths = set(val_df["image_path"])
        test_paths = set(test_df["image_path"])
        leaves_seen: dict[str, set[str]] = {}
        for path in train_paths | val_paths | test_paths:
            lid = leaf_id_map[path]
            leaves_seen.setdefault(lid, set())
            if path in train_paths:
                leaves_seen[lid].add("train")
            if path in val_paths:
                leaves_seen[lid].add("val")
            if path in test_paths:
                leaves_seen[lid].add("test")
        for lid, splits in leaves_seen.items():
            assert len(splits) == 1, f"Leaf {lid} found in multiple splits: {splits}"

    def test_leaf_proportions_approximate(self, tmp_path):
        records, leaf_id_map = self._make_leaf_records(tmp_path)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 36
        assert 0.55 <= len(train_df) / total <= 0.85
        assert 0.05 <= len(val_df) / total <= 0.35
        assert 0.05 <= len(test_df) / total <= 0.35

    def test_leaf_all_classes_in_train(self, tmp_path):
        records, leaf_id_map = self._make_leaf_records(tmp_path)
        splits_dir = tmp_path / "splits"
        train_df, _, _ = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        assert set(train_df["clean_class"]) == {"Healthy", "Early Blight", "Late Blight"}

    def test_leaf_fallback_no_leaf_id_map(self, tmp_path):
        records = self._make_records(tmp_path, per_class=20)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=None,
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 60
        assert 0.60 <= len(train_df) / total <= 0.80

    def test_leaf_fallback_empty_leaf_id_map(self, tmp_path):
        records = self._make_records(tmp_path, per_class=20)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map={},
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 60
        train_set = set(train_df["image_path"])
        val_set = set(val_df["image_path"])
        assert len(train_set & val_set) == 0


# ---------------------------------------------------------------------------
# write_split_report tests
# ---------------------------------------------------------------------------


class TestWriteSplitReport:
    def test_report_pass_no_overlap(self, tmp_path):
        train_df = pd.DataFrame({
            "image_path": ["a.jpg", "b.jpg"],
            "clean_class": ["Healthy", "Early Blight"],
        })
        val_df = pd.DataFrame({
            "image_path": ["c.jpg"],
            "clean_class": ["Late Blight"],
        })
        test_df = pd.DataFrame({
            "image_path": ["d.jpg"],
            "clean_class": ["Healthy"],
        })
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out)
        text = out.read_text(encoding="utf-8")
        assert "Train: 2" in text
        assert "Validation: 1" in text
        assert "Test: 1" in text
        assert "PASS" in text

    def test_report_fail_with_overlap(self, tmp_path):
        train_df = pd.DataFrame({
            "image_path": ["a.jpg"],
            "clean_class": ["Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["a.jpg"],  # same path = leakage!
            "clean_class": ["Healthy"],
        })
        test_df = pd.DataFrame({
            "image_path": ["b.jpg"],
            "clean_class": ["Early Blight"],
        })
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out)
        text = out.read_text(encoding="utf-8")
        assert "FAIL" in text


# ---------------------------------------------------------------------------
# load_source_metadata tests
# ---------------------------------------------------------------------------


class TestLoadSourceMetadata:
    def test_load_valid_csv(self, tmp_path):
        csv_path = tmp_path / "source_metadata.csv"
        csv_path.write_text(
            "image_path,leaf_id,label\n"
            "/img/a.jpg,leaf_1,Tomato___healthy\n"
            "/img/b.jpg,leaf_2,Tomato___Early_blight\n",
            encoding="utf-8",
        )
        result = load_source_metadata(csv_path)
        assert len(result) == 2
        assert result[str(Path("/img/a.jpg").resolve())] == "leaf_1"
        assert result[str(Path("/img/b.jpg").resolve())] == "leaf_2"

    def test_missing_file_returns_empty(self, tmp_path):
        result = load_source_metadata(tmp_path / "nonexistent.csv")
        assert result == {}

    def test_missing_columns_returns_empty(self, tmp_path):
        csv_path = tmp_path / "bad_meta.csv"
        csv_path.write_text("path,label\n/img/a.jpg,healthy\n", encoding="utf-8")
        result = load_source_metadata(csv_path)
        assert result == {}


# ---------------------------------------------------------------------------
# Leaf-aware split report tests
# ---------------------------------------------------------------------------


class TestWriteSplitReportLeaf:
    def test_report_with_leaf_no_leakage(self, tmp_path):
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg", "/a/2.jpg", "/b/3.jpg"],
            "clean_class": ["Healthy", "Healthy", "Early Blight"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/c/4.jpg"],
            "clean_class": ["Late Blight"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/d/5.jpg"],
            "clean_class": ["Healthy"],
        })
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "leaf_A",
            str(Path("/a/2.jpg").resolve()): "leaf_A",
            str(Path("/b/3.jpg").resolve()): "leaf_B",
            str(Path("/c/4.jpg").resolve()): "leaf_C",
            str(Path("/d/5.jpg").resolve()): "leaf_D",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        assert "Leaf Grouping:" in text
        assert "Verified leaves: 4" in text
        assert "Overall leakage: PASS" in text
        assert "Verified leaf grouping was used" in text

    def test_report_with_leaf_leakage(self, tmp_path):
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg"],
            "clean_class": ["Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/a/2.jpg"],
            "clean_class": ["Healthy"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/b/3.jpg"],
            "clean_class": ["Early Blight"],
        })
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "leaf_A",
            str(Path("/a/2.jpg").resolve()): "leaf_A",
            str(Path("/b/3.jpg").resolve()): "leaf_B",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        assert "RESULT: FAIL - Verified leaf overlap detected!" in text
        assert "Overall leakage: FAIL" in text

    def test_report_without_leaf_id_backward_compat(self, tmp_path):
        train_df = pd.DataFrame({
            "image_path": ["a.jpg"],
            "clean_class": ["Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["b.jpg"],
            "clean_class": ["Early Blight"],
        })
        test_df = pd.DataFrame({
            "image_path": ["c.jpg"],
            "clean_class": ["Late Blight"],
        })
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out)
        text = out.read_text(encoding="utf-8")
        assert "Leaf Grouping:" not in text
        assert "PASS" in text

    def test_metadata_csv_includes_leaf_id(self, tmp_path):
        records = [{
            "image_path": "img.jpg",
            "raw_class": "Tomato___healthy",
            "clean_class": "Healthy",
            "file_extension": ".jpg",
            "image_mode": "RGB",
            "width": 64,
            "height": 64,
            "md5": "abc123",
            "leaf_id": "leaf_42",
        }]
        out = tmp_path / "meta.csv"
        write_metadata_csv(records, out)
        df = pd.read_csv(out)
        assert "leaf_id" in df.columns
        assert df["leaf_id"].iloc[0] == "leaf_42"


# ---------------------------------------------------------------------------
# Regression: report leakage logic must match generate_splits() grouping
# ---------------------------------------------------------------------------


class TestReportLeakageLogic:
    """Regression tests for write_split_report() leaf leakage check.

    The report must use the SAME effective grouping as generate_splits():
    - Verified leaves are checked for cross-split overlap.
    - "unknown" and "fallback_*" are NOT considered one physical leaf.
    - Each unverified image is its own effective group.
    """

    def test_unknown_ids_no_false_leakage(self, tmp_path):
        """Images with leaf_id='unknown' spread across splits must NOT
        trigger a false FAIL."""
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg", "/a/2.jpg"],
            "clean_class": ["Healthy", "Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/b/3.jpg"],
            "clean_class": ["Early Blight"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/c/4.jpg", "/c/5.jpg"],
            "clean_class": ["Late Blight", "Late Blight"],
        })
        # All images have leaf_id="unknown"
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "unknown",
            str(Path("/a/2.jpg").resolve()): "unknown",
            str(Path("/b/3.jpg").resolve()): "unknown",
            str(Path("/c/4.jpg").resolve()): "unknown",
            str(Path("/c/5.jpg").resolve()): "unknown",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        # No verified leaf overlap
        assert "RESULT: PASS" in text
        assert "Overall leakage: PASS" in text
        # Verified leaves = 0, ungrouped = 5
        assert "Verified leaves: 0" in text
        assert "Ungrouped (unverified) images: 5" in text
        assert "Effective groups: 5" in text

    def test_fallback_ids_no_false_leakage(self, tmp_path):
        """Images with fallback leaf IDs must NOT trigger false FAIL."""
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg"],
            "clean_class": ["Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/b/2.jpg"],
            "clean_class": ["Early Blight"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/c/3.jpg"],
            "clean_class": ["Late Blight"],
        })
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "fallback_abc123",
            str(Path("/b/2.jpg").resolve()): "fallback_def456",
            str(Path("/c/3.jpg").resolve()): "fallback_ghi789",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        assert "Overall leakage: PASS" in text
        assert "Verified leaves: 0" in text
        assert "Ungrouped (unverified) images: 3" in text

    def test_verified_leaf_overlap_causes_fail(self, tmp_path):
        """Same verified leaf in two splits MUST cause FAIL."""
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg"],
            "clean_class": ["Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/a/2.jpg"],  # same leaf as train
            "clean_class": ["Healthy"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/b/3.jpg"],
            "clean_class": ["Early Blight"],
        })
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "leaf_X",
            str(Path("/a/2.jpg").resolve()): "leaf_X",  # same leaf!
            str(Path("/b/3.jpg").resolve()): "leaf_Y",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        assert "RESULT: FAIL - Verified leaf overlap detected!" in text
        assert "Overall leakage: FAIL" in text

    def test_verified_leaves_one_split_causes_pass(self, tmp_path):
        """Verified leaves each appearing in only one split must PASS."""
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg", "/a/2.jpg"],
            "clean_class": ["Healthy", "Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/b/3.jpg"],
            "clean_class": ["Early Blight"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/c/4.jpg"],
            "clean_class": ["Late Blight"],
        })
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "leaf_A",
            str(Path("/a/2.jpg").resolve()): "leaf_A",
            str(Path("/b/3.jpg").resolve()): "leaf_B",
            str(Path("/c/4.jpg").resolve()): "leaf_C",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        assert "RESULT: PASS" in text
        assert "Overall leakage: PASS" in text
        assert "Verified leaves: 3" in text

    def test_image_overlap_causes_fail(self, tmp_path):
        """Same image path in two splits MUST cause FAIL."""
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg"],
            "clean_class": ["Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/a/1.jpg"],  # duplicate!
            "clean_class": ["Healthy"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/b/2.jpg"],
            "clean_class": ["Early Blight"],
        })
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "unknown",
            str(Path("/b/2.jpg").resolve()): "unknown",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        # Image overlap detected
        assert "RESULT: FAIL - Overlap detected!" in text
        assert "Overall leakage: FAIL" in text

    def test_mixed_verified_and_unknown_no_false_leakage(self, tmp_path):
        """Mix of verified leaves and unknown images, all properly placed."""
        # Verified leaf_A only in train, leaf_B only in val
        # Unknown images spread across all splits
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg", "/a/2.jpg", "/x/10.jpg"],
            "clean_class": ["Healthy", "Healthy", "Late Blight"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/b/3.jpg", "/y/20.jpg"],
            "clean_class": ["Early Blight", "Healthy"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/c/4.jpg", "/z/30.jpg"],
            "clean_class": ["Late Blight", "Early Blight"],
        })
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "leaf_A",
            str(Path("/a/2.jpg").resolve()): "leaf_A",
            str(Path("/b/3.jpg").resolve()): "leaf_B",
            str(Path("/c/4.jpg").resolve()): "leaf_C",
            str(Path("/x/10.jpg").resolve()): "unknown",
            str(Path("/y/20.jpg").resolve()): "unknown",
            str(Path("/z/30.jpg").resolve()): "unknown",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        assert "Overall leakage: PASS" in text
        assert "Verified leaves: 3" in text  # leaf_A, leaf_B, leaf_C
        assert "Ungrouped (unverified) images: 3" in text  # 3 unknowns
        assert "Effective groups: 6" in text  # 3 verified + 3 ungrouped

    def test_effective_groups_count(self, tmp_path):
        """Verify effective groups = verified leaves + ungrouped images."""
        train_df = pd.DataFrame({
            "image_path": ["/a/1.jpg", "/u/99.jpg"],
            "clean_class": ["Healthy", "Healthy"],
        })
        val_df = pd.DataFrame({
            "image_path": ["/b/2.jpg", "/v/88.jpg"],
            "clean_class": ["Early Blight", "Early Blight"],
        })
        test_df = pd.DataFrame({
            "image_path": ["/c/3.jpg", "/w/77.jpg"],
            "clean_class": ["Late Blight", "Late Blight"],
        })
        leaf_id_map = {
            str(Path("/a/1.jpg").resolve()): "leaf_A",
            str(Path("/b/2.jpg").resolve()): "leaf_B",
            str(Path("/c/3.jpg").resolve()): "leaf_C",
            str(Path("/u/99.jpg").resolve()): "unknown",
            str(Path("/v/88.jpg").resolve()): "fallback_xyz",
            str(Path("/w/77.jpg").resolve()): "unknown",
        }
        out = tmp_path / "split_report.txt"
        write_split_report(train_df, val_df, test_df, out, leaf_id_map=leaf_id_map)
        text = out.read_text(encoding="utf-8")
        assert "Verified leaves: 3" in text
        assert "Ungrouped (unverified) images: 3" in text
        assert "Effective groups: 6" in text
        assert "Overall leakage: PASS" in text


# ---------------------------------------------------------------------------
# Regression: unverified leaf IDs must not form a mega-group (issue: 45/10/45)
# ---------------------------------------------------------------------------


class TestUnverifiedLeafGrouping:
    """Regression tests for the split proportion bug.

    When many images have leaf_id="unknown", they must NOT form one giant
    leaf group that dumps all images into a single split.
    """

    @staticmethod
    def _make_mixed_records(tmp_path, n_verified=60, n_unknown=60):
        """Create records: half verified (grouped by leaf), half unknown."""
        records = []
        classes = [
            ("Tomato___healthy", "Healthy", (255, 0, 0)),
            ("Tomato___Early_blight", "Early Blight", (0, 255, 0)),
            ("Tomato___Late_blight", "Late Blight", (0, 0, 255)),
        ]
        uid = 0
        leaf_id_map: dict[str, str] = {}

        # Verified images: 3 classes, 10 leaves each, 2 images per leaf
        per_class = n_verified // 3
        leaves_per_class = max(per_class // 2, 1)
        for raw, clean, color in classes:
            class_dir = tmp_path / raw
            class_dir.mkdir(parents=True, exist_ok=True)
            for leaf_idx in range(leaves_per_class):
                leaf_id = f"{raw}:::{leaf_idx + 1}"
                for img_idx in range(2):
                    img_path = class_dir / f"v_{uid}.jpg"
                    _create_test_image(img_path, color=color, unique_id=uid)
                    uid += 1
                    p = str(img_path.resolve())
                    records.append({
                        "image_path": p,
                        "raw_class": raw,
                        "clean_class": clean,
                        "file_extension": ".jpg",
                        "image_mode": "RGB",
                        "width": 64,
                        "height": 64,
                        "md5": compute_md5(img_path),
                    })
                    leaf_id_map[p] = leaf_id

        # Unknown images: 3 classes, equal distribution
        per_class_unk = n_unknown // 3
        for raw, clean, color in classes:
            class_dir = tmp_path / raw
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class_unk):
                img_path = class_dir / f"u_{uid}.jpg"
                _create_test_image(img_path, color=color, unique_id=uid)
                uid += 1
                p = str(img_path.resolve())
                records.append({
                    "image_path": p,
                    "raw_class": raw,
                    "clean_class": clean,
                    "file_extension": ".jpg",
                    "image_mode": "RGB",
                    "width": 64,
                    "height": 64,
                    "md5": compute_md5(img_path),
                })
                leaf_id_map[p] = "unknown"

        return records, leaf_id_map

    def test_unknown_does_not_mega_group(self, tmp_path):
        """Unknown images must be distributed across ALL three splits."""
        records, leaf_id_map = self._make_mixed_records(
            tmp_path, n_verified=60, n_unknown=60,
        )
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        # Count unknown images per split
        unknown_paths = {
            p for p, lid in leaf_id_map.items() if lid == "unknown"
        }
        train_unk = len(
            set(train_df["image_path"]) & unknown_paths
        )
        val_unk = len(
            set(val_df["image_path"]) & unknown_paths
        )
        test_unk = len(
            set(test_df["image_path"]) & unknown_paths
        )
        # All three splits must have some unknown images
        assert train_unk > 0, "Train has zero unknown-leaf images"
        assert val_unk > 0, "Val has zero unknown-leaf images"
        assert test_unk > 0, "Test has zero unknown-leaf images"

    def test_split_proportions_approximately_70_15_15(self, tmp_path):
        """Split proportions should be close to 70/15/15."""
        records, leaf_id_map = self._make_mixed_records(
            tmp_path, n_verified=90, n_unknown=90,
        )
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 180
        train_pct = len(train_df) / total
        val_pct = len(val_df) / total
        test_pct = len(test_df) / total
        # Tolerant bounds for leaf-group-aware splitting
        assert 0.55 <= train_pct <= 0.85, (
            f"Train {train_pct:.1%} outside [55%, 85%]"
        )
        assert 0.05 <= val_pct <= 0.30, (
            f"Val {val_pct:.1%} outside [5%, 30%]"
        )
        assert 0.05 <= test_pct <= 0.30, (
            f"Test {test_pct:.1%} outside [5%, 30%]"
        )

    def test_no_image_overlap(self, tmp_path):
        records, leaf_id_map = self._make_mixed_records(tmp_path)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        t = set(train_df["image_path"])
        v = set(val_df["image_path"])
        s = set(test_df["image_path"])
        assert len(t & v) == 0
        assert len(t & s) == 0
        assert len(v & s) == 0

    def test_verified_leaf_no_overlap(self, tmp_path):
        """Verified leaves must appear in exactly one split."""
        records, leaf_id_map = self._make_mixed_records(tmp_path)
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        verified = {
            p: lid for p, lid in leaf_id_map.items()
            if lid != "unknown" and not lid.lower().startswith("fallback_")
        }
        t = set(train_df["image_path"])
        v = set(val_df["image_path"])
        s = set(test_df["image_path"])
        leaf_splits: dict[str, set[str]] = {}
        for p, lid in verified.items():
            leaf_splits.setdefault(lid, set())
            if p in t:
                leaf_splits[lid].add("train")
            if p in v:
                leaf_splits[lid].add("val")
            if p in s:
                leaf_splits[lid].add("test")
        for lid, sp in leaf_splits.items():
            assert len(sp) == 1, f"Verified leaf {lid} in {sp}"

    def test_all_classes_in_train(self, tmp_path):
        records, leaf_id_map = self._make_mixed_records(tmp_path)
        splits_dir = tmp_path / "splits"
        train_df, _, _ = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        assert set(train_df["clean_class"]) == {
            "Healthy", "Early Blight", "Late Blight",
        }

    def test_dedup_still_works(self, tmp_path):
        """Duplicate removal happens before splitting."""
        records, leaf_id_map = self._make_mixed_records(
            tmp_path, n_verified=60, n_unknown=0,
        )
        original_count = len(records)
        # Inject a duplicate MD5
        records[1]["md5"] = records[0]["md5"]
        dup_groups = {records[0]["md5"]: [
            records[0]["image_path"], records[1]["image_path"],
        ]}
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, dup_groups, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        total = len(train_df) + len(val_df) + len(test_df)
        # Dedup removed at least 1 record (the injected duplicate)
        assert total < original_count

    def test_backward_compat_no_leaf_map(self, tmp_path):
        """Without leaf_id_map, standard stratified split works."""
        records, _ = self._make_mixed_records(
            tmp_path, n_verified=30, n_unknown=30,
        )
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=None,
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 60
        assert 0.55 <= len(train_df) / total <= 0.85

    def test_deterministic_with_seed(self, tmp_path):
        """Same seed produces identical splits."""
        records, leaf_id_map = self._make_mixed_records(tmp_path)
        s1 = tmp_path / "s1"
        s2 = tmp_path / "s2"
        t1, v1, _ = generate_splits(
            records, {}, 42, s1, leaf_id_map=leaf_id_map,
        )
        t2, v2, _ = generate_splits(
            records, {}, 42, s2, leaf_id_map=leaf_id_map,
        )
        assert list(t1["image_path"]) == list(t2["image_path"])
        assert list(v1["image_path"]) == list(v2["image_path"])

    def test_fallback_leaf_id_treated_like_unknown(self, tmp_path):
        """fallback_ leaf IDs are also individually distributed."""
        records = []
        leaf_id_map: dict[str, str] = {}
        class_dir = tmp_path / "Tomato___healthy"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(30):
            img_path = class_dir / f"fb_{i}.jpg"
            _create_test_image(img_path, unique_id=i)
            p = str(img_path.resolve())
            records.append({
                "image_path": p,
                "raw_class": "Tomato___healthy",
                "clean_class": "Healthy",
                "file_extension": ".jpg",
                "image_mode": "RGB",
                "width": 64,
                "height": 64,
                "md5": compute_md5(img_path),
            })
            leaf_id_map[p] = f"fallback_{i}"
        # Add some verified leaves too
        for i in range(30):
            img_path = class_dir / f"v_{i}.jpg"
            _create_test_image(img_path, unique_id=100 + i)
            p = str(img_path.resolve())
            records.append({
                "image_path": p,
                "raw_class": "Tomato___healthy",
                "clean_class": "Healthy",
                "file_extension": ".jpg",
                "image_mode": "RGB",
                "width": 64,
                "height": 64,
                "md5": compute_md5(img_path),
            })
            leaf_id_map[p] = f"Tomato___healthy:::{i + 1}"
        splits_dir = tmp_path / "splits"
        train_df, val_df, test_df = generate_splits(
            records, {}, 42, splits_dir, leaf_id_map=leaf_id_map,
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 60
        # Both train and test should have images
        assert len(train_df) > 0
        assert len(test_df) > 0
