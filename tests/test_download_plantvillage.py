"""
Tests for src.download_plantvillage — Direct PlantVillage Acquisition.

Covers all helper functions and the full pipeline using mock/local ZIP
files.  Never downloads the real 2.18 GB dataset.
"""

import json
import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.download_plantvillage import (
    CLASS_MAP,
    TARGET_CLASSES,
    TARGET_CLASS_SET,
    check_output_exists,
    cleanup_generated_output,
    combine_splits,
    download_pipeline,
    extract_class_from_path,
    extract_short_id,
    extract_target_images,
    filter_target_paths,
    load_leaf_map,
    parse_split_file,
    resolve_all_leaf_ids,
    resolve_leaf_id,
    write_source_metadata_csv,
)


# ---------------------------------------------------------------------------
# Helpers — create test artifacts
# ---------------------------------------------------------------------------

# Realistic PlantVillage-style paths for tests
_TRAIN_LINES = [
    "raw/color/Tomato___healthy/uuid-001___RS_HL 9973.JPG",
    "raw/color/Tomato___healthy/uuid-002___RS_HL 9971.JPG",
    "raw/color/Tomato___healthy/uuid-003___GH_HL Leaf 268.JPG",
    "raw/color/Tomato___Early_blight/uuid-010___RS_Erly.B 7544.JPG",
    "raw/color/Tomato___Early_blight/uuid-011___RS_Erly.B 7541.JPG",
    "raw/color/Tomato___Late_blight/uuid-020___RS_Late.B 5095.JPG",
    "raw/color/Apple___Apple_scab/uuid-100___RS_AS 1234.JPG",
    "raw/color/Potato___healthy/uuid-200___RS_PH 5555.JPG",
]

_TEST_LINES = [
    "raw/color/Tomato___healthy/uuid-301___RS_HL 9980.JPG",
    "raw/color/Tomato___Late_blight/uuid-310___RS_Late.B 5100.JPG",
    "raw/color/Apple___healthy/uuid-400___RS_AH 9999.JPG",
]

# Leaf-map: lowercase short IDs -> list of "class:::leaf_num"
_LEAF_MAP = {
    "rs_hl 9973": ["Tomato___healthy:::245.0"],
    "rs_hl 9971": ["Tomato___healthy:::245.0"],
    "rs_erly.b 7544": ["Tomato___Early_blight:::76.0"],
    "rs_erly.b 7541": ["Tomato___Early_blight:::76.0"],
    "rs_late.b 5095": ["Tomato___Late_blight:::51.0"],
    "rs_hl 9980": ["Tomato___healthy:::246.0"],
    "rs_late.b 5100": ["Tomato___Late_blight:::52.0"],
    "fallback_img 123": ["Tomato___healthy:::fallback_1"],
}


def _write_split_file(path: Path, lines: list[str]) -> None:
    """Write a mock split file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_leaf_map(path: Path, data: dict) -> None:
    """Write a mock leaf-map.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _create_test_zip(
    zip_path: Path,
    member_paths: list[str],
) -> None:
    """Create a ZIP file containing small dummy image files at given paths."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "w") as zf:
        for member in member_paths:
            # Write a tiny fake JPEG (SOI + EOI markers)
            zf.writestr(member, b"\xff\xd8\xff\xe0FAKE_JPEG\xff\xd9")


def _create_test_image(path: Path, size=(64, 64)) -> None:
    """Create a minimal test JPEG."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 0, 0)).save(str(path))


# ---------------------------------------------------------------------------
# parse_split_file tests
# ---------------------------------------------------------------------------


class TestParseSplitFile:
    def test_parses_normal_lines(self, tmp_path):
        f = tmp_path / "train.txt"
        _write_split_file(f, _TRAIN_LINES)
        result = parse_split_file(f)
        assert len(result) == len(_TRAIN_LINES)
        assert result[0] == _TRAIN_LINES[0]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "train.txt"
        _write_split_file(f, ["", "raw/color/Tomato___healthy/u___X.JPG", ""])
        result = parse_split_file(f)
        assert len(result) == 1

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert parse_split_file(f) == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_split_file(tmp_path / "nope.txt")

    def test_malformed_line_raises(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("no-slash-here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed"):
            parse_split_file(f)


# ---------------------------------------------------------------------------
# extract_class_from_path tests
# ---------------------------------------------------------------------------


class TestExtractClassFromPath:
    def test_standard_path(self):
        assert (
            extract_class_from_path(
                "raw/color/Tomato___healthy/uuid.JPG"
            )
            == "Tomato___healthy"
        )

    def test_backslash_path(self):
        assert (
            extract_class_from_path(
                "raw\\color\\Tomato___Early_blight\\uuid.JPG"
            )
            == "Tomato___Early_blight"
        )

    def test_short_path_returns_empty(self):
        assert extract_class_from_path("raw/color") == ""

    def test_empty_string(self):
        assert extract_class_from_path("") == ""


# ---------------------------------------------------------------------------
# filter_target_paths tests
# ---------------------------------------------------------------------------


class TestFilterTargetPaths:
    def test_keeps_target_only(self):
        result = filter_target_paths(_TRAIN_LINES, TARGET_CLASS_SET)
        assert len(result) == 6  # 3 healthy + 2 early + 1 late
        for p in result:
            assert "Tomato___" in p

    def test_empty_input(self):
        assert filter_target_paths([], TARGET_CLASS_SET) == []

    def test_no_matches(self):
        paths = ["raw/color/Apple___scab/x.JPG"]
        assert filter_target_paths(paths, TARGET_CLASS_SET) == []


# ---------------------------------------------------------------------------
# combine_splits tests
# ---------------------------------------------------------------------------


class TestCombineSplits:
    def test_tags_source_split(self):
        records = combine_splits(_TRAIN_LINES[:1], _TEST_LINES[:1])
        assert records[0]["source_split"] == "train"
        assert records[1]["source_split"] == "test"

    def test_extracts_class_name(self):
        records = combine_splits(
            ["raw/color/Tomato___healthy/u___X.JPG"], []
        )
        assert records[0]["class_name"] == "Tomato___healthy"

    def test_deduplicates(self):
        dup = "raw/color/Tomato___healthy/u___X.JPG"
        records = combine_splits([dup], [dup])
        assert len(records) == 1
        assert records[0]["source_split"] == "train"  # first wins

    def test_empty_splits(self):
        assert combine_splits([], []) == []


# ---------------------------------------------------------------------------
# extract_short_id tests
# ---------------------------------------------------------------------------


class TestExtractShortId:
    def test_standard_filename(self):
        path = "raw/color/Tomato___healthy/uuid-001___RS_HL 9973.JPG"
        assert extract_short_id(path) == "RS_HL 9973"

    def test_no_uuid(self):
        path = "raw/color/Tomato___healthy/RS_HL 9973.JPG"
        assert extract_short_id(path) == "RS_HL 9973"

    def test_multiple_triple_underscores(self):
        # rsplit("___", 1) takes the last segment
        path = "raw/color/Tomato___healthy/uuid___RS_HL 9973.JPG"
        assert extract_short_id(path) == "RS_HL 9973"

    def test_no_extension(self):
        path = "raw/color/Tomato___healthy/uuid___ShortID"
        assert extract_short_id(path) == "ShortID"

    def test_no_separator(self):
        path = "raw/color/Tomato___healthy/plainfile.jpg"
        assert extract_short_id(path) == "plainfile"


# ---------------------------------------------------------------------------
# resolve_leaf_id tests
# ---------------------------------------------------------------------------


class TestResolveLeafId:
    def test_verified_match(self):
        leaf_id, status = resolve_leaf_id("RS_HL 9973", _LEAF_MAP)
        assert status == "verified"
        assert leaf_id == "Tomato___healthy:::245"

    def test_case_insensitive(self):
        leaf_id, status = resolve_leaf_id("rs_hl 9973", _LEAF_MAP)
        assert status == "verified"

    def test_unknown_when_missing(self):
        leaf_id, status = resolve_leaf_id("nonexistent id", _LEAF_MAP)
        assert status == "unknown"
        assert leaf_id == "unknown"

    def test_empty_short_id(self):
        leaf_id, status = resolve_leaf_id("", _LEAF_MAP)
        assert status == "unknown"

    def test_fallback_detected(self):
        leaf_id, status = resolve_leaf_id("fallback_img 123", _LEAF_MAP)
        assert status == "fallback"

    def test_leaf_num_parsed_as_int(self):
        leaf_id, status = resolve_leaf_id("RS_Erly.B 7544", _LEAF_MAP)
        assert status == "verified"
        assert leaf_id == "Tomato___Early_blight:::76"

    def test_empty_value_list(self):
        lm = {"some_key": []}
        leaf_id, status = resolve_leaf_id("some_key", lm)
        assert status == "unknown"


# ---------------------------------------------------------------------------
# resolve_all_leaf_ids tests
# ---------------------------------------------------------------------------


class TestResolveAllLeafIds:
    def test_adds_leaf_id_and_status(self):
        records = [
            {
                "source_path": "raw/color/Tomato___healthy/uuid-001___RS_HL 9973.JPG",
                "source_split": "train",
                "class_name": "Tomato___healthy",
            },
            {
                "source_path": "raw/color/Tomato___healthy/uuid-003___GH_HL Leaf 268.JPG",
                "source_split": "train",
                "class_name": "Tomato___healthy",
            },
        ]
        result = resolve_all_leaf_ids(records, _LEAF_MAP)
        assert result[0]["leaf_id"] == "Tomato___healthy:::245"
        assert result[0]["leaf_id_status"] == "verified"
        assert result[1]["leaf_id_status"] == "unknown"

    def test_empty_records(self):
        assert resolve_all_leaf_ids([], _LEAF_MAP) == []


# ---------------------------------------------------------------------------
# load_leaf_map tests
# ---------------------------------------------------------------------------


class TestLoadLeafMap:
    def test_loads_valid_json(self, tmp_path):
        f = tmp_path / "leaf-map.json"
        _write_leaf_map(f, {"key1": ["val1"]})
        result = load_leaf_map(f)
        assert result == {"key1": ["val1"]}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_leaf_map(tmp_path / "nope.json")

    def test_non_dict_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="dict"):
            load_leaf_map(f)


# ---------------------------------------------------------------------------
# extract_target_images tests
# ---------------------------------------------------------------------------


class TestExtractTargetImages:
    @staticmethod
    def _make_zip_and_records(tmp_path):
        """Create a test ZIP and matching records."""
        zip_path = tmp_path / "data.zip"
        paths = [
            "raw/color/Tomato___healthy/uuid-001___RS_HL 9973.JPG",
            "raw/color/Tomato___Early_blight/uuid-010___RS_Erly.B 7544.JPG",
            "raw/color/Apple___Apple_scab/uuid-100___RS_AS 1234.JPG",
        ]
        _create_test_zip(zip_path, paths)

        records = [
            {
                "source_path": paths[0],
                "source_split": "train",
                "class_name": "Tomato___healthy",
                "leaf_id": "Tomato___healthy:::245",
                "leaf_id_status": "verified",
            },
            {
                "source_path": paths[1],
                "source_split": "train",
                "class_name": "Tomato___Early_blight",
                "leaf_id": "Tomato___Early_blight:::76",
                "leaf_id_status": "verified",
            },
        ]
        return zip_path, records

    def test_extracts_target_images(self, tmp_path):
        zip_path, records = self._make_zip_and_records(tmp_path)
        out = tmp_path / "output"
        saved = extract_target_images(zip_path, records, out)
        assert len(saved) == 2
        for entry in saved:
            assert Path(entry["image_path"]).exists()

    def test_does_not_extract_non_target(self, tmp_path):
        zip_path, records = self._make_zip_and_records(tmp_path)
        out = tmp_path / "output"
        extract_target_images(zip_path, records, out)
        assert not (out / "Apple___Apple_scab").exists()

    def test_skip_existing(self, tmp_path):
        zip_path, records = self._make_zip_and_records(tmp_path)
        out = tmp_path / "output"
        # First extraction
        extract_target_images(zip_path, records, out, skip_existing=True)
        # Get file content
        first_path = Path(saved[0]["image_path"]) if (
            saved := extract_target_images(
                zip_path, records, out, skip_existing=True
            )
        ) else None
        assert first_path is not None
        content1 = first_path.read_bytes()
        # Second extraction — file should be unchanged
        saved2 = extract_target_images(
            zip_path, records, out, skip_existing=True
        )
        assert Path(saved2[0]["image_path"]).read_bytes() == content1

    def test_missing_zip_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_target_images(
                tmp_path / "nope.zip", [], tmp_path / "out"
            )

    def test_corrupted_zip_raises(self, tmp_path):
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_text("not a zip", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Corrupted"):
            extract_target_images(bad_zip, [], tmp_path / "out")

    def test_missing_member_in_zip(self, tmp_path):
        zip_path = tmp_path / "data.zip"
        _create_test_zip(
            zip_path,
            ["raw/color/Tomato___healthy/exists.JPG"],
        )
        records = [{
            "source_path": "raw/color/Tomato___healthy/missing.JPG",
            "source_split": "train",
            "class_name": "Tomato___healthy",
            "leaf_id": "unknown",
            "leaf_id_status": "unknown",
        }]
        saved = extract_target_images(
            zip_path, records, tmp_path / "out"
        )
        assert len(saved) == 0  # missing member skipped

    def test_saved_metadata_fields(self, tmp_path):
        zip_path, records = self._make_zip_and_records(tmp_path)
        saved = extract_target_images(
            zip_path, records, tmp_path / "output"
        )
        entry = saved[0]
        assert "image_path" in entry
        assert "leaf_id" in entry
        assert "original_label" in entry
        assert "clean_class" in entry
        assert "source_split" in entry
        assert "crop" in entry
        assert "disease" in entry
        assert "leaf_id_status" in entry
        assert entry["crop"] == "Tomato"
        assert entry["clean_class"] == "Healthy"


# ---------------------------------------------------------------------------
# write_source_metadata_csv tests
# ---------------------------------------------------------------------------


class TestWriteSourceMetadataCSV:
    def test_writes_correct_csv(self, tmp_path):
        records = [
            {
                "image_path": "/img/a.jpg",
                "leaf_id": "Tomato___healthy:::245",
                "original_label": "Tomato___healthy",
                "clean_class": "Healthy",
                "source_split": "train",
                "crop": "Tomato",
                "disease": "healthy",
                "leaf_id_status": "verified",
            },
        ]
        out = tmp_path / "meta.csv"
        write_source_metadata_csv(records, out)
        df = pd.read_csv(out)
        assert len(df) == 1
        expected_cols = {
            "image_path", "leaf_id", "original_label", "clean_class",
            "source_split", "crop", "disease", "leaf_id_status",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_empty_records(self, tmp_path):
        out = tmp_path / "meta.csv"
        write_source_metadata_csv([], out)
        assert out.exists()
        assert len(pd.read_csv(out)) == 0

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "meta.csv"
        write_source_metadata_csv([], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# check_output_exists tests
# ---------------------------------------------------------------------------


class TestCheckOutputExists:
    def test_empty_directory(self, tmp_path):
        for cls in TARGET_CLASSES:
            (tmp_path / cls).mkdir(parents=True)
        assert check_output_exists(tmp_path, TARGET_CLASSES) is False

    def test_with_images(self, tmp_path):
        _create_test_image(tmp_path / "Tomato___healthy" / "img.jpg")
        assert check_output_exists(tmp_path, TARGET_CLASSES) is True

    def test_nonexistent_directory(self, tmp_path):
        assert check_output_exists(tmp_path / "nope", TARGET_CLASSES) is False

    def test_non_image_files_ignored(self, tmp_path):
        (tmp_path / "Tomato___healthy").mkdir(parents=True)
        (tmp_path / "Tomato___healthy" / "readme.txt").write_text("hello")
        assert check_output_exists(tmp_path, TARGET_CLASSES) is False


# ---------------------------------------------------------------------------
# cleanup_generated_output tests
# ---------------------------------------------------------------------------


class TestCleanupGeneratedOutput:
    def test_removes_images(self, tmp_path):
        _create_test_image(tmp_path / "Tomato___healthy" / "img.jpg")
        meta = tmp_path / "meta.csv"
        meta.write_text("data", encoding="utf-8")
        cleanup_generated_output(tmp_path, meta, TARGET_CLASSES)
        assert not (tmp_path / "Tomato___healthy" / "img.jpg").exists()
        assert not meta.exists()

    def test_preserves_unrelated_files(self, tmp_path):
        _create_test_image(tmp_path / "Tomato___healthy" / "img.jpg")
        readme = tmp_path / "Tomato___healthy" / "notes.txt"
        readme.write_text("keep me", encoding="utf-8")
        cleanup_generated_output(tmp_path, tmp_path / "no.csv", TARGET_CLASSES)
        assert readme.exists()


# ---------------------------------------------------------------------------
# Pipeline integration tests (mocked downloads, local ZIP)
# ---------------------------------------------------------------------------


def _mock_download_pipeline(tmp_path):
    """Create all mock files and return a patcher for download_repo_file."""
    zip_path = tmp_path / "cache" / "data.zip"
    leaf_map_path = tmp_path / "cache" / "leaf-map.json"
    train_path = tmp_path / "cache" / "color_train.txt"
    test_path = tmp_path / "cache" / "color_test.txt"

    # All paths that appear in the split files must exist in the ZIP
    all_paths = _TRAIN_LINES + _TEST_LINES
    _create_test_zip(zip_path, all_paths)
    _write_leaf_map(leaf_map_path, _LEAF_MAP)
    _write_split_file(train_path, _TRAIN_LINES)
    _write_split_file(test_path, _TEST_LINES)

    file_map = {
        "data.zip": str(zip_path),
        "leaf_grouping/leaf-map.json": str(leaf_map_path),
        "splits/color_train.txt": str(train_path),
        "splits/color_test.txt": str(test_path),
    }

    def fake_download(filename, repo_id=None):
        if filename not in file_map:
            raise FileNotFoundError(f"Mock: {filename} not found")
        return Path(file_map[filename])

    return patch(
        "src.download_plantvillage.download_repo_file",
        side_effect=fake_download,
    )


class TestDownloadPipeline:
    def test_full_pipeline(self, tmp_path):
        output_dir = tmp_path / "tomato"
        metadata_path = tmp_path / "processed" / "source_metadata.csv"

        with _mock_download_pipeline(tmp_path):
            download_pipeline(output_dir, metadata_path)

        # Check images extracted for target classes only
        for cls in TARGET_CLASSES:
            class_dir = output_dir / cls
            assert class_dir.exists()
            assert len(list(class_dir.iterdir())) > 0

        # Non-target classes NOT extracted
        assert not (output_dir / "Apple___Apple_scab").exists()
        assert not (output_dir / "Potato___healthy").exists()

        # Metadata CSV correct
        assert metadata_path.exists()
        df = pd.read_csv(metadata_path)
        expected_cols = {
            "image_path", "leaf_id", "original_label", "clean_class",
            "source_split", "crop", "disease", "leaf_id_status",
        }
        assert expected_cols.issubset(set(df.columns))
        assert len(df) > 0
        # All rows are tomato
        for label in df["original_label"]:
            assert label in TARGET_CLASSES

    def test_source_split_preserved(self, tmp_path):
        output_dir = tmp_path / "tomato"
        metadata_path = tmp_path / "meta.csv"

        with _mock_download_pipeline(tmp_path):
            download_pipeline(output_dir, metadata_path)

        df = pd.read_csv(metadata_path)
        splits = set(df["source_split"])
        assert "train" in splits
        assert "test" in splits

    def test_leaf_id_status_present(self, tmp_path):
        output_dir = tmp_path / "tomato"
        metadata_path = tmp_path / "meta.csv"

        with _mock_download_pipeline(tmp_path):
            download_pipeline(output_dir, metadata_path)

        df = pd.read_csv(metadata_path)
        assert "leaf_id_status" in df.columns
        statuses = set(df["leaf_id_status"])
        # Our test data has both verified and unknown
        assert "verified" in statuses

    def test_skip_existing_idempotent(self, tmp_path):
        output_dir = tmp_path / "tomato"
        metadata_path = tmp_path / "meta.csv"

        with _mock_download_pipeline(tmp_path):
            download_pipeline(output_dir, metadata_path)
        first_csv = metadata_path.read_text(encoding="utf-8")

        # Second run — existing images skipped
        with _mock_download_pipeline(tmp_path):
            download_pipeline(output_dir, metadata_path)
        second_csv = metadata_path.read_text(encoding="utf-8")
        assert first_csv == second_csv

    def test_metadata_compatible_with_load_source(self, tmp_path):
        """Verify CSV integrates with prepare_dataset.load_source_metadata."""
        output_dir = tmp_path / "tomato"
        metadata_path = tmp_path / "meta.csv"

        from src.prepare_dataset import load_source_metadata

        with _mock_download_pipeline(tmp_path):
            download_pipeline(output_dir, metadata_path)

        leaf_map = load_source_metadata(metadata_path)
        assert len(leaf_map) > 0
        for path, leaf_id in leaf_map.items():
            assert Path(path).exists()
            # leaf_id is either verified ("class:::N") or "unknown"
            assert ":::" in leaf_id or leaf_id == "unknown"

    def test_missing_zip_member_logged(self, tmp_path):
        """Records with missing ZIP members are skipped gracefully."""
        # Create ZIP with only some of the paths
        zip_path = tmp_path / "cache" / "data.zip"
        leaf_map_path = tmp_path / "cache" / "leaf-map.json"
        train_path = tmp_path / "cache" / "color_train.txt"
        test_path = tmp_path / "cache" / "color_test.txt"

        _create_test_zip(zip_path, _TRAIN_LINES[:2])  # only 2 of 8
        _write_leaf_map(leaf_map_path, _LEAF_MAP)
        _write_split_file(train_path, _TRAIN_LINES)
        _write_split_file(test_path, _TEST_LINES)

        file_map = {
            "data.zip": str(zip_path),
            "leaf_grouping/leaf-map.json": str(leaf_map_path),
            "splits/color_train.txt": str(train_path),
            "splits/color_test.txt": str(test_path),
        }

        def fake_download(filename, repo_id=None):
            return Path(file_map[filename])

        output_dir = tmp_path / "tomato"
        metadata_path = tmp_path / "meta.csv"

        with patch(
            "src.download_plantvillage.download_repo_file",
            side_effect=fake_download,
        ):
            download_pipeline(output_dir, metadata_path)

        # Only the 2 available images extracted (both healthy)
        df = pd.read_csv(metadata_path)
        assert len(df) == 2

    def test_no_target_class_fails(self, tmp_path):
        """Pipeline fails if a target class has zero records."""
        zip_path = tmp_path / "cache" / "data.zip"
        leaf_map_path = tmp_path / "cache" / "leaf-map.json"
        train_path = tmp_path / "cache" / "color_train.txt"
        test_path = tmp_path / "cache" / "color_test.txt"

        # Only Apple paths — no tomato
        apple_lines = [
            "raw/color/Apple___Apple_scab/uuid-100___RS_AS 1234.JPG",
        ]
        _create_test_zip(zip_path, apple_lines)
        _write_leaf_map(leaf_map_path, _LEAF_MAP)
        _write_split_file(train_path, apple_lines)
        _write_split_file(test_path, [])

        file_map = {
            "data.zip": str(zip_path),
            "leaf_grouping/leaf-map.json": str(leaf_map_path),
            "splits/color_train.txt": str(train_path),
            "splits/color_test.txt": str(test_path),
        }

        def fake_download(filename, repo_id=None):
            return Path(file_map[filename])

        with patch(
            "src.download_plantvillage.download_repo_file",
            side_effect=fake_download,
        ):
            with pytest.raises(SystemExit):
                download_pipeline(
                    tmp_path / "tomato", tmp_path / "meta.csv"
                )


# ---------------------------------------------------------------------------
# Force flag tests
# ---------------------------------------------------------------------------


class TestForceFlag:
    def test_force_cleans_and_rebuilds(self, tmp_path):
        output_dir = tmp_path / "tomato"
        metadata_path = tmp_path / "meta.csv"

        # First run
        with _mock_download_pipeline(tmp_path):
            download_pipeline(output_dir, metadata_path)
        assert check_output_exists(output_dir, TARGET_CLASSES)

        # Force cleanup
        cleanup_generated_output(
            output_dir, metadata_path, TARGET_CLASSES
        )
        assert not check_output_exists(output_dir, TARGET_CLASSES)

        # Rebuild
        with _mock_download_pipeline(tmp_path):
            download_pipeline(output_dir, metadata_path)
        assert check_output_exists(output_dir, TARGET_CLASSES)


# ---------------------------------------------------------------------------
# Duplicate path tests
# ---------------------------------------------------------------------------


class TestDuplicatePaths:
    def test_duplicates_dropped(self):
        dup = "raw/color/Tomato___healthy/uuid___RS_HL 9973.JPG"
        records = combine_splits([dup, dup], [dup])
        assert len(records) == 1

    def test_same_path_different_split_first_wins(self):
        dup = "raw/color/Tomato___healthy/uuid___RS_HL 9973.JPG"
        records = combine_splits([dup], [dup])
        assert records[0]["source_split"] == "train"


# ---------------------------------------------------------------------------
# Malformed input tests
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    def test_split_file_malformed_line(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("no_slash_here\n", encoding="utf-8")
        with pytest.raises(ValueError):
            parse_split_file(f)

    def test_leaf_map_non_dict(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError):
            load_leaf_map(f)


# ---------------------------------------------------------------------------
# Missing leaf-map tests
# ---------------------------------------------------------------------------


class TestMissingLeafMap:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_leaf_map(tmp_path / "nonexistent.json")

    def test_images_extracted_without_leaf_ids(self, tmp_path):
        """Pipeline works even if no leaf-map matches exist."""
        zip_path = tmp_path / "cache" / "data.zip"
        leaf_map_path = tmp_path / "cache" / "leaf-map.json"
        train_path = tmp_path / "cache" / "color_train.txt"
        test_path = tmp_path / "cache" / "color_test.txt"

        lines = [
            "raw/color/Tomato___healthy/uuid___NoMatch1.JPG",
            "raw/color/Tomato___Early_blight/uuid___NoMatch2.JPG",
            "raw/color/Tomato___Late_blight/uuid___NoMatch3.JPG",
        ]
        _create_test_zip(zip_path, lines)
        _write_leaf_map(leaf_map_path, {})  # empty leaf-map
        _write_split_file(train_path, lines)
        _write_split_file(test_path, [])

        file_map = {
            "data.zip": str(zip_path),
            "leaf_grouping/leaf-map.json": str(leaf_map_path),
            "splits/color_train.txt": str(train_path),
            "splits/color_test.txt": str(test_path),
        }

        def fake_download(filename, repo_id=None):
            return Path(file_map[filename])

        with patch(
            "src.download_plantvillage.download_repo_file",
            side_effect=fake_download,
        ):
            download_pipeline(tmp_path / "tomato", tmp_path / "meta.csv")

        df = pd.read_csv(tmp_path / "meta.csv")
        assert len(df) == 3
        assert all(df["leaf_id_status"] == "unknown")
