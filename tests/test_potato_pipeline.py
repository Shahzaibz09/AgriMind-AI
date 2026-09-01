"""
Tests for the Potato multi-crop extension — pipeline integrity.

Covers the potato crop configuration, class mapping, trilingual knowledge
base (structure + content safety, mirroring the tomato checks in
tests/test_farmer_assistant.py), guard threshold calibration claims, and
the generated potato artifacts (splits, checkpoint, training metadata).
Tomato remains the production crop: its artifacts are asserted untouched.

Data-dependent tests skip when artifacts have not been generated yet
(same convention as TestRealDataSafety in tests/test_guard.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from app.assistant import load_knowledge
from app.crop_config import get_crop_config, list_crops
from app.guard import (
    QUALITY_REJECT,
    GuardThresholds,
    compute_quality_metrics,
    evaluate_quality,
    load_guard_thresholds,
)
from src.prepare_dataset import CLASS_MAP, CROP_CLASS_MAPS, POTATO_CLASS_MAP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = PROJECT_ROOT / "configs" / "crops"
KNOWLEDGE_DIR = PROJECT_ROOT / "configs" / "knowledge"
POTATO_CROP_CFG_PATH = CROPS_DIR / "potato.yaml"
TOMATO_CROP_CFG_PATH = CROPS_DIR / "tomato.yaml"
POTATO_SPLITS_DIR = PROJECT_ROOT / "data" / "splits_potato"
POTATO_CHECKPOINT = PROJECT_ROOT / "models" / "potato_best_model.pth"
POTATO_METADATA = PROJECT_ROOT / "models" / "potato_training_metadata.json"
TOMATO_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pth"
POTATO_TEST_CSV = POTATO_SPLITS_DIR / "test.csv"

EXPECTED_CLASSES = {"Healthy", "Early Blight", "Late Blight"}

# Potato test-distribution minima measured during guard calibration
# (324 test images; see the calibration table in configs/crops/potato.yaml).
# Every reject threshold must sit at or below half its real-data minimum
# so no real leaf image is ever rejected.
_POTATO_REAL_MINIMA = {
    "laplacian_variance": 563.9,
    "green_fraction": 0.0887,
    "adjacent_correlation": 0.5213,
    "saturation": 0.0886,
}


# ---------------------------------------------------------------------------
# Helpers (mirrored from tests/test_farmer_assistant.py)
# ---------------------------------------------------------------------------


def _has_urdu_script(text: str) -> bool:
    """Return True if the text contains Arabic-script (Urdu) characters."""
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _iter_strings(node):
    """Yield every string inside a nested YAML structure."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _iter_strings(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _iter_strings(value)


# ---------------------------------------------------------------------------
# Crop configuration
# ---------------------------------------------------------------------------


class TestPotatoCropConfig:
    def test_list_crops_includes_both_crops(self):
        crops = list_crops(str(CROPS_DIR))
        assert "Tomato" in crops
        assert "Potato" in crops
        assert "Apple" in crops
        assert len(crops) == 3

    def test_potato_config_has_required_keys(self):
        cfg = get_crop_config("Potato", crops_dir=str(CROPS_DIR))
        for key in (
            "crop_name", "model_path", "num_classes", "class_names", "image_size",
        ):
            assert key in cfg, key
        assert cfg["crop_name"] == "Potato"
        assert set(cfg["class_names"]) == EXPECTED_CLASSES
        assert cfg["num_classes"] == 3
        assert cfg["image_size"] == 224

    def test_potato_model_file_exists(self):
        cfg = get_crop_config("Potato", crops_dir=str(CROPS_DIR))
        assert (PROJECT_ROOT / cfg["model_path"]).exists()

    def test_potato_model_isolated_from_tomato(self):
        cfg = get_crop_config("Potato", crops_dir=str(CROPS_DIR))
        assert cfg["model_path"] != "models/best_model.pth"
        # Tomato's own artifacts are untouched by the potato extension.
        assert TOMATO_CHECKPOINT.exists()
        assert TOMATO_CROP_CFG_PATH.exists()

    def test_tomato_config_unchanged(self):
        raw = yaml.safe_load(TOMATO_CROP_CFG_PATH.read_text(encoding="utf-8"))
        assert raw["crop_name"] == "Tomato"
        assert raw["model_path"] == "models/best_model.pth"
        assert set(raw["class_names"]) == EXPECTED_CLASSES


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------


class TestPotatoClassMap:
    def test_potato_class_map_maps_all_folders(self):
        assert POTATO_CLASS_MAP == {
            "Potato___healthy": "Healthy",
            "Potato___Early_blight": "Early Blight",
            "Potato___Late_blight": "Late Blight",
        }

    def test_crop_class_maps_registry(self):
        assert set(CROP_CLASS_MAPS) == {"tomato", "potato", "apple"}
        assert CROP_CLASS_MAPS["tomato"] is CLASS_MAP
        assert CROP_CLASS_MAPS["potato"] is POTATO_CLASS_MAP

    def test_clean_class_names_shared_with_tomato(self):
        # Identical clean class names are what lets the dataset, training,
        # and evaluation modules work unchanged for both crops.
        assert set(CLASS_MAP.values()) == EXPECTED_CLASSES
        assert set(POTATO_CLASS_MAP.values()) == EXPECTED_CLASSES

    def test_tomato_class_map_unchanged(self):
        assert CLASS_MAP == {
            "Tomato___healthy": "Healthy",
            "Tomato___Early_blight": "Early Blight",
            "Tomato___Late_blight": "Late Blight",
        }


# ---------------------------------------------------------------------------
# Knowledge base — structure and trilingual completeness
# ---------------------------------------------------------------------------


class TestPotatoKnowledge:
    def test_potato_knowledge_loads(self):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        assert knowledge["crop_name"] == "Potato"
        assert "diseases" in knowledge

    def test_case_insensitive_crop_match(self):
        knowledge = load_knowledge("potato", KNOWLEDGE_DIR)
        assert knowledge["crop_name"] == "Potato"

    def test_all_three_diseases_present(self):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        assert set(knowledge["diseases"]) == EXPECTED_CLASSES

    def test_all_languages_available_per_disease(self):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            assert set(entry) == {"en", "ur", "roman_ur"}, disease

    def test_required_sections_per_language(self):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        required = {"title", "explanation", "symptoms", "management", "expert_advice"}
        for disease, entry in knowledge["diseases"].items():
            for lang, content in entry.items():
                assert required.issubset(content), (disease, lang)
                assert content["symptoms"], (disease, lang)
                assert content["management"], (disease, lang)

    def test_fallbacks_declared_in_knowledge(self):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        fallbacks = knowledge.get("fallbacks", {})
        for kind in ("missing_prediction", "unknown_disease", "low_confidence_note"):
            assert kind in fallbacks
            assert "en" in fallbacks[kind]

    def test_urdu_entries_use_urdu_script(self):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            assert _has_urdu_script(entry["ur"]["explanation"]), disease

    def test_roman_urdu_entries_are_ascii(self):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            assert entry["roman_ur"]["explanation"].isascii(), disease


# ---------------------------------------------------------------------------
# Knowledge base — content safety (mirrors tomato's checks)
# ---------------------------------------------------------------------------


class TestPotatoContentSafety:
    @pytest.mark.parametrize(
        "pattern",
        [
            r"\d+\s*(?:ml|g|kg|gram|grams|liter|litre|liters|litres)\b",
            r"\b(?:dose|dosage)\b",
            r"\bspray\s+per\s+acre\b",
            r"(?:خوراک|گرام|ملی\s*لیٹر)",
        ],
    )
    def test_no_dosage_patterns_anywhere(self, pattern):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        for text in _iter_strings(knowledge):
            assert re.search(pattern, text, re.IGNORECASE) is None, (
                f"Pattern {pattern!r} matched in: {text!r}"
            )

    def test_management_lists_are_cultural_only(self):
        """Management guidance must not prescribe chemical products."""
        banned = ("pesticide", "fungicide", "insecticide", "spray", "chemical")
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            for lang, content in entry.items():
                for item in content["management"]:
                    lower = item.lower()
                    for word in banned:
                        assert word not in lower, (disease, lang, item)

    def test_pathogens_mentioned(self):
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        eb = knowledge["diseases"]["Early Blight"]["en"]["explanation"]
        lb = knowledge["diseases"]["Late Blight"]["en"]["explanation"]
        assert "Alternaria" in eb
        assert "Phytophthora" in lb

    def test_potato_specific_guidance_present(self):
        """Potato guidance must cover tuber/hilling/rotation (early blight)
        and destruction/storage-rot risks (late blight)."""
        knowledge = load_knowledge("Potato", KNOWLEDGE_DIR)
        eb_mgmt = " ".join(knowledge["diseases"]["Early Blight"]["en"]["management"])
        lb_mgmt = " ".join(knowledge["diseases"]["Late Blight"]["en"]["management"])
        lb_advice = knowledge["diseases"]["Late Blight"]["en"]["expert_advice"]

        assert "tuber" in eb_mgmt.lower()
        assert "hilling" in eb_mgmt.lower()
        assert "Rotate" in eb_mgmt  # crop rotation

        assert "destroy" in lb_mgmt.lower() or "bury" in lb_mgmt.lower()
        assert "harvest" in lb_mgmt.lower()  # do not harvest diseased plants
        assert "storage" in lb_advice.lower() or "rot" in lb_advice.lower()


# ---------------------------------------------------------------------------
# Guard threshold calibration
# ---------------------------------------------------------------------------


class TestPotatoGuardConfig:
    def _thresholds(self) -> GuardThresholds:
        cfg = get_crop_config("Potato", crops_dir=str(CROPS_DIR))
        return load_guard_thresholds(cfg)

    def test_guard_section_present_with_all_thresholds(self):
        raw = yaml.safe_load(POTATO_CROP_CFG_PATH.read_text(encoding="utf-8"))
        guard = raw.get("guard")
        assert isinstance(guard, dict)
        for key in (
            "min_resolution",
            "saturation_min",
            "flatness_lapvar_min",
            "green_fraction_min",
            "correlation_min",
            "blur_lapvar_warn",
            "confidence_warn",
            "margin_warn",
        ):
            assert key in guard, key

    def test_thresholds_load_from_config(self):
        t = self._thresholds()
        assert t != GuardThresholds()  # potato-specific calibration applied
        assert t.flatness_lapvar_min == 280.0
        assert t.green_fraction_min == 0.04
        assert t.correlation_min == 0.25
        assert t.saturation_min == 0.04
        assert t.blur_lapvar_warn == 750.0

    def test_reject_thresholds_at_or_below_half_real_minimum(self):
        t = self._thresholds()
        assert (
            t.flatness_lapvar_min
            <= _POTATO_REAL_MINIMA["laplacian_variance"] / 2
        )
        assert t.green_fraction_min <= _POTATO_REAL_MINIMA["green_fraction"] / 2
        assert (
            t.correlation_min
            <= _POTATO_REAL_MINIMA["adjacent_correlation"] / 2
        )
        assert t.saturation_min <= _POTATO_REAL_MINIMA["saturation"] / 2

    def test_warn_thresholds_sane(self):
        t = self._thresholds()
        assert t.blur_lapvar_warn > t.flatness_lapvar_min
        assert 0.0 < t.confidence_warn < 1.0
        assert 0.0 < t.margin_warn < 1.0

    def test_tomato_guard_defaults_unchanged(self):
        raw = yaml.safe_load(TOMATO_CROP_CFG_PATH.read_text(encoding="utf-8"))
        t = load_guard_thresholds(raw)
        assert t == GuardThresholds()  # tomato config mirrors the defaults


# ---------------------------------------------------------------------------
# Generated potato artifacts (splits, checkpoint, metadata)
# ---------------------------------------------------------------------------


class TestPotatoArtifacts:
    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_split_csv_exists_with_valid_classes(self, split):
        csv_path = POTATO_SPLITS_DIR / f"{split}.csv"
        if not csv_path.exists():
            pytest.skip("potato split manifest not available")
        df = pd.read_csv(csv_path)
        assert set(df.columns) == {"image_path", "clean_class"}
        assert set(df["clean_class"]) == EXPECTED_CLASSES

    def test_split_report_passes_leakage_check(self):
        report_path = POTATO_SPLITS_DIR / "split_report.txt"
        if not report_path.exists():
            pytest.skip("potato split report not available")
        report = report_path.read_text(encoding="utf-8")
        assert "Overall leakage: PASS" in report

    def test_source_metadata_rows_are_potato(self):
        csv_path = POTATO_SPLITS_DIR / "source_metadata.csv"
        if not csv_path.exists():
            pytest.skip("potato source metadata not available")
        df = pd.read_csv(csv_path)
        assert len(df) == 2152
        assert set(df["crop"]) == {"Potato"}
        assert set(df["original_label"]) == set(POTATO_CLASS_MAP)

    def test_checkpoint_and_metadata_exist(self):
        if not POTATO_CHECKPOINT.exists() or not POTATO_METADATA.exists():
            pytest.skip("potato training artifacts not available")
        # Tomato artifacts untouched.
        assert TOMATO_CHECKPOINT.exists()
        meta = json.loads(POTATO_METADATA.read_text(encoding="utf-8"))
        assert meta["architecture"] == "mobilenet_v3_small"
        assert meta["num_classes"] == 3
        assert set(meta["class_mapping"].values()) == EXPECTED_CLASSES


# ---------------------------------------------------------------------------
# Real-data guard safety + inference (potato model end-to-end)
# ---------------------------------------------------------------------------


class TestPotatoRealDataSafety:
    def test_all_test_images_pass_reject_gates(self):
        """Every real potato test leaf must pass all reject-level checks."""
        if not POTATO_TEST_CSV.exists():
            pytest.skip("potato test split manifest not available")

        df = pd.read_csv(POTATO_TEST_CSV)
        paths = df["image_path"].tolist()
        assert len(paths) > 300

        thresholds = load_guard_thresholds(
            get_crop_config("Potato", crops_dir=str(CROPS_DIR))
        )
        rejected = []
        for p in paths:
            p = Path(p)
            img_path = p if p.is_absolute() else PROJECT_ROOT / p
            with Image.open(img_path) as f:
                img = f.convert("RGB")
            verdict = evaluate_quality(compute_quality_metrics(img), thresholds)
            if verdict["status"] == QUALITY_REJECT:
                rejected.append((str(p), verdict["reasons"]))
        assert rejected == []

    def test_trained_model_predicts_each_class(self):
        """One real leaf per class must load the checkpoint and predict."""
        if not POTATO_CHECKPOINT.exists():
            pytest.skip("potato checkpoint not available")

        from src.inference import predict_image
        from src.model import load_checkpoint

        cfg = get_crop_config("Potato", crops_dir=str(CROPS_DIR))
        model = load_checkpoint(cfg["model_path"], device="cpu")
        model.eval()

        for class_dir, expected in (
            ("Potato___healthy", "Healthy"),
            ("Potato___Early_blight", "Early Blight"),
            ("Potato___Late_blight", "Late Blight"),
        ):
            samples = sorted((PROJECT_ROOT / "data" / "raw" / "potato" / class_dir).glob("*.JPG"))
            if not samples:
                pytest.skip(f"no raw potato images for {class_dir}")
            result = predict_image(
                model=model,
                image_path=str(samples[0]),
                device="cpu",
                image_size=cfg["image_size"],
                class_names=cfg["class_names"],
                gradcam=False,
            )
            assert result["predicted_class"] == expected
