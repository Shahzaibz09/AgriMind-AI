"""
Tests for the Apple multi-crop extension — pipeline integrity.

Covers the apple crop configuration, class mapping (including the
requirement that apple disease names never mix with the tomato/potato
disease names), trilingual knowledge base (structure + content safety,
mirroring the tomato/potato checks), guard threshold calibration
claims, and the generated apple artifacts (splits, checkpoint, training
metadata). Tomato and potato remain untouched: their artifacts are
asserted present and their registries unchanged.

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
from src.prepare_dataset import (
    APPLE_CLASS_MAP,
    CLASS_MAP,
    CROP_CLASS_MAPS,
    POTATO_CLASS_MAP,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = PROJECT_ROOT / "configs" / "crops"
KNOWLEDGE_DIR = PROJECT_ROOT / "configs" / "knowledge"
APPLE_CROP_CFG_PATH = CROPS_DIR / "apple.yaml"
POTATO_CROP_CFG_PATH = CROPS_DIR / "potato.yaml"
TOMATO_CROP_CFG_PATH = CROPS_DIR / "tomato.yaml"
APPLE_SPLITS_DIR = PROJECT_ROOT / "data" / "splits_apple"
APPLE_CHECKPOINT = PROJECT_ROOT / "models" / "apple_best_model.pth"
APPLE_METADATA = PROJECT_ROOT / "models" / "apple_training_metadata.json"
TOMATO_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pth"
POTATO_CHECKPOINT = PROJECT_ROOT / "models" / "potato_best_model.pth"
APPLE_TEST_CSV = APPLE_SPLITS_DIR / "test.csv"

APPLE_CLASSES = {"Healthy", "Apple Scab", "Black Rot"}
TOMATO_POTATO_DISEASES = {"Early Blight", "Late Blight"}

# Apple test-distribution minima measured during guard calibration
# (428 test images; see the calibration table in configs/crops/apple.yaml).
# Every reject threshold must sit at or below half its real-data minimum
# so no real leaf image is ever rejected.
_APPLE_REAL_MINIMA = {
    "laplacian_variance": 94.99,
    "green_fraction": 0.0068,
    "adjacent_correlation": 0.5956,
    "saturation": 0.0326,
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


class TestAppleCropConfig:
    def test_list_crops_includes_all_three_crops(self):
        crops = list_crops(str(CROPS_DIR))
        assert "Tomato" in crops
        assert "Potato" in crops
        assert "Apple" in crops
        assert len(crops) == 3

    def test_apple_config_has_required_keys(self):
        cfg = get_crop_config("Apple", crops_dir=str(CROPS_DIR))
        for key in (
            "crop_name", "model_path", "num_classes", "class_names", "image_size",
        ):
            assert key in cfg, key
        assert cfg["crop_name"] == "Apple"
        assert set(cfg["class_names"]) == APPLE_CLASSES
        assert cfg["num_classes"] == 3
        assert cfg["image_size"] == 224

    def test_apple_model_file_exists(self):
        cfg = get_crop_config("Apple", crops_dir=str(CROPS_DIR))
        assert (PROJECT_ROOT / cfg["model_path"]).exists()

    def test_apple_model_isolated_from_other_crops(self):
        cfg = get_crop_config("Apple", crops_dir=str(CROPS_DIR))
        assert cfg["model_path"] != "models/best_model.pth"
        assert cfg["model_path"] != "models/potato_best_model.pth"
        # Tomato and potato artifacts remain untouched by the apple extension.
        assert TOMATO_CHECKPOINT.exists()
        assert POTATO_CHECKPOINT.exists()
        assert POTATO_CROP_CFG_PATH.exists()

    def test_tomato_and_potato_configs_unchanged(self):
        tomato = yaml.safe_load(TOMATO_CROP_CFG_PATH.read_text(encoding="utf-8"))
        assert tomato["crop_name"] == "Tomato"
        assert tomato["model_path"] == "models/best_model.pth"
        assert set(tomato["class_names"]) == {"Healthy", "Early Blight", "Late Blight"}
        potato = yaml.safe_load(POTATO_CROP_CFG_PATH.read_text(encoding="utf-8"))
        assert potato["crop_name"] == "Potato"
        assert potato["model_path"] == "models/potato_best_model.pth"
        assert set(potato["class_names"]) == {"Healthy", "Early Blight", "Late Blight"}


# ---------------------------------------------------------------------------
# Class mapping — apple names must never mix with tomato/potato classes
# ---------------------------------------------------------------------------


class TestAppleClassMap:
    def test_apple_class_map_maps_all_folders(self):
        assert APPLE_CLASS_MAP == {
            "Apple___healthy": "Healthy",
            "Apple___Apple_scab": "Apple Scab",
            "Apple___Black_rot": "Black Rot",
        }

    def test_crop_class_maps_registry(self):
        assert set(CROP_CLASS_MAPS) == {"tomato", "potato", "apple"}
        assert CROP_CLASS_MAPS["tomato"] is CLASS_MAP
        assert CROP_CLASS_MAPS["potato"] is POTATO_CLASS_MAP
        assert CROP_CLASS_MAPS["apple"] is APPLE_CLASS_MAP

    def test_apple_disease_names_disjoint_from_tomato_potato(self):
        """Crop-specific disease names must never mix across crops."""
        apple_diseases = set(APPLE_CLASS_MAP.values()) - {"Healthy"}
        assert apple_diseases == {"Apple Scab", "Black Rot"}
        assert apple_diseases.isdisjoint(TOMATO_POTATO_DISEASES)
        # And in the other direction: no apple name leaked into the
        # tomato/potato registries.
        assert set(CLASS_MAP.values()).isdisjoint(apple_diseases)
        assert set(POTATO_CLASS_MAP.values()).isdisjoint(apple_diseases)

    def test_apple_config_class_names_match_class_map(self):
        cfg = get_crop_config("Apple", crops_dir=str(CROPS_DIR))
        assert set(cfg["class_names"]) == set(APPLE_CLASS_MAP.values())

    def test_default_label_mapping_unchanged(self):
        # The module-level tomato/potato default mapping stays intact.
        from src.dataset import CLASS_TO_IDX

        assert CLASS_TO_IDX == {
            "Healthy": 0,
            "Early Blight": 1,
            "Late Blight": 2,
        }

    def test_tomato_class_map_unchanged(self):
        assert CLASS_MAP == {
            "Tomato___healthy": "Healthy",
            "Tomato___Early_blight": "Early Blight",
            "Tomato___Late_blight": "Late Blight",
        }


# ---------------------------------------------------------------------------
# Knowledge base — structure and trilingual completeness
# ---------------------------------------------------------------------------


class TestAppleKnowledge:
    def test_apple_knowledge_loads(self):
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        assert knowledge["crop_name"] == "Apple"
        assert "diseases" in knowledge

    def test_case_insensitive_crop_match(self):
        knowledge = load_knowledge("apple", KNOWLEDGE_DIR)
        assert knowledge["crop_name"] == "Apple"

    def test_all_three_diseases_present(self):
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        assert set(knowledge["diseases"]) == APPLE_CLASSES

    def test_all_languages_available_per_disease(self):
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            assert set(entry) == {"en", "ur", "roman_ur"}, disease

    def test_required_sections_per_language(self):
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        required = {"title", "explanation", "symptoms", "management", "expert_advice"}
        for disease, entry in knowledge["diseases"].items():
            for lang, content in entry.items():
                assert required.issubset(content), (disease, lang)
                assert content["symptoms"], (disease, lang)
                assert content["management"], (disease, lang)

    def test_fallbacks_declared_in_knowledge(self):
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        fallbacks = knowledge.get("fallbacks", {})
        for kind in ("missing_prediction", "unknown_disease", "low_confidence_note"):
            assert kind in fallbacks
            assert "en" in fallbacks[kind]

    def test_urdu_entries_use_urdu_script(self):
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            assert _has_urdu_script(entry["ur"]["explanation"]), disease

    def test_roman_urdu_entries_are_ascii(self):
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            assert entry["roman_ur"]["explanation"].isascii(), disease


# ---------------------------------------------------------------------------
# Knowledge base — content safety (mirrors tomato/potato checks)
# ---------------------------------------------------------------------------


class TestAppleContentSafety:
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
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        for text in _iter_strings(knowledge):
            assert re.search(pattern, text, re.IGNORECASE) is None, (
                f"Pattern {pattern!r} matched in: {text!r}"
            )

    def test_management_lists_are_cultural_only(self):
        """Management guidance must not prescribe chemical products."""
        banned = ("pesticide", "fungicide", "insecticide", "spray", "chemical")
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            for lang, content in entry.items():
                for item in content["management"]:
                    lower = item.lower()
                    for word in banned:
                        assert word not in lower, (disease, lang, item)

    def test_pathogens_mentioned(self):
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        scab = knowledge["diseases"]["Apple Scab"]["en"]["explanation"]
        rot = knowledge["diseases"]["Black Rot"]["en"]["explanation"]
        assert "Venturia" in scab
        assert "Botryosphaeria" in rot

    def test_apple_specific_guidance_present(self):
        """Apple guidance must cover sanitation/pruning (apple scab) and
        canker/mummified-fruit removal (black rot)."""
        knowledge = load_knowledge("Apple", KNOWLEDGE_DIR)
        scab_mgmt = " ".join(knowledge["diseases"]["Apple Scab"]["en"]["management"])
        rot_mgmt = " ".join(knowledge["diseases"]["Black Rot"]["en"]["management"])

        assert "fallen leaves" in scab_mgmt.lower()  # overwintering sanitation
        assert "prune" in scab_mgmt.lower()  # airflow through the canopy

        assert "canker" in rot_mgmt.lower() or "dead" in rot_mgmt.lower()
        assert "mummified" in rot_mgmt.lower()  # remove mummified fruit


# ---------------------------------------------------------------------------
# Guard threshold calibration
# ---------------------------------------------------------------------------


class TestAppleGuardConfig:
    def _thresholds(self) -> GuardThresholds:
        cfg = get_crop_config("Apple", crops_dir=str(CROPS_DIR))
        return load_guard_thresholds(cfg)

    def test_guard_section_present_with_all_thresholds(self):
        raw = yaml.safe_load(APPLE_CROP_CFG_PATH.read_text(encoding="utf-8"))
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
        assert t != GuardThresholds()  # apple-specific calibration applied
        assert t.flatness_lapvar_min == 45.0
        assert t.green_fraction_min == 0.003
        assert t.correlation_min == 0.25
        assert t.saturation_min == 0.015
        assert t.blur_lapvar_warn == 180.0

    def test_reject_thresholds_at_or_below_half_real_minimum(self):
        t = self._thresholds()
        assert (
            t.flatness_lapvar_min
            <= _APPLE_REAL_MINIMA["laplacian_variance"] / 2
        )
        assert t.green_fraction_min <= _APPLE_REAL_MINIMA["green_fraction"] / 2
        assert (
            t.correlation_min
            <= _APPLE_REAL_MINIMA["adjacent_correlation"] / 2
        )
        assert t.saturation_min <= _APPLE_REAL_MINIMA["saturation"] / 2

    def test_warn_thresholds_sane(self):
        t = self._thresholds()
        assert t.blur_lapvar_warn > t.flatness_lapvar_min
        assert 0.0 < t.confidence_warn < 1.0
        assert 0.0 < t.margin_warn < 1.0

    def test_tomato_and_potato_guard_configs_unchanged(self):
        tomato = yaml.safe_load(TOMATO_CROP_CFG_PATH.read_text(encoding="utf-8"))
        assert load_guard_thresholds(tomato) == GuardThresholds()
        potato = yaml.safe_load(POTATO_CROP_CFG_PATH.read_text(encoding="utf-8"))
        t = load_guard_thresholds(potato)
        assert t.flatness_lapvar_min == 280.0
        assert t.blur_lapvar_warn == 750.0


# ---------------------------------------------------------------------------
# Generated apple artifacts (splits, checkpoint, metadata)
# ---------------------------------------------------------------------------


class TestAppleArtifacts:
    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_split_csv_exists_with_valid_classes(self, split):
        csv_path = APPLE_SPLITS_DIR / f"{split}.csv"
        if not csv_path.exists():
            pytest.skip("apple split manifest not available")
        df = pd.read_csv(csv_path)
        assert set(df.columns) == {"image_path", "clean_class"}
        assert set(df["clean_class"]) == APPLE_CLASSES

    def test_split_report_passes_leakage_check(self):
        report_path = APPLE_SPLITS_DIR / "split_report.txt"
        if not report_path.exists():
            pytest.skip("apple split report not available")
        report = report_path.read_text(encoding="utf-8")
        assert "Overall leakage: PASS" in report

    def test_source_metadata_rows_are_apple(self):
        csv_path = APPLE_SPLITS_DIR / "source_metadata.csv"
        if not csv_path.exists():
            pytest.skip("apple source metadata not available")
        df = pd.read_csv(csv_path)
        assert len(df) == 2896
        assert set(df["crop"]) == {"Apple"}
        assert set(df["original_label"]) == set(APPLE_CLASS_MAP)

    def test_checkpoint_and_metadata_exist(self):
        if not APPLE_CHECKPOINT.exists() or not APPLE_METADATA.exists():
            pytest.skip("apple training artifacts not available")
        # Tomato and potato artifacts untouched.
        assert TOMATO_CHECKPOINT.exists()
        assert POTATO_CHECKPOINT.exists()
        meta = json.loads(APPLE_METADATA.read_text(encoding="utf-8"))
        assert meta["architecture"] == "mobilenet_v3_small"
        assert meta["num_classes"] == 3
        assert set(meta["class_mapping"].values()) == APPLE_CLASSES

    def test_apple_class_mapping_order_is_config_order(self):
        """Training metadata class_mapping must follow the config order so
        the app's class_names indexing matches the model's output logits."""
        if not APPLE_METADATA.exists():
            pytest.skip("apple training artifacts not available")
        meta = json.loads(APPLE_METADATA.read_text(encoding="utf-8"))
        cfg = get_crop_config("Apple", crops_dir=str(CROPS_DIR))
        expected = {
            str(i): name for i, name in enumerate(cfg["class_names"])
        }
        assert meta["class_mapping"] == expected


# ---------------------------------------------------------------------------
# Real-data guard safety + inference (apple model end-to-end)
# ---------------------------------------------------------------------------


class TestAppleRealDataSafety:
    def test_all_test_images_pass_reject_gates(self):
        """Every real apple test leaf must pass all reject-level checks."""
        if not APPLE_TEST_CSV.exists():
            pytest.skip("apple test split manifest not available")

        df = pd.read_csv(APPLE_TEST_CSV)
        paths = df["image_path"].tolist()
        assert len(paths) > 400

        thresholds = load_guard_thresholds(
            get_crop_config("Apple", crops_dir=str(CROPS_DIR))
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
        if not APPLE_CHECKPOINT.exists():
            pytest.skip("apple checkpoint not available")

        from src.inference import predict_image
        from src.model import load_checkpoint

        cfg = get_crop_config("Apple", crops_dir=str(CROPS_DIR))
        model = load_checkpoint(cfg["model_path"], device="cpu")
        model.eval()

        for class_dir, expected in (
            ("Apple___healthy", "Healthy"),
            ("Apple___Apple_scab", "Apple Scab"),
            ("Apple___Black_rot", "Black Rot"),
        ):
            samples = sorted(
                (PROJECT_ROOT / "data" / "raw" / "apple" / class_dir).glob("*.JPG")
            )
            if not samples:
                pytest.skip(f"no raw apple images for {class_dir}")
            result = predict_image(
                model=model,
                image_path=str(samples[0]),
                device="cpu",
                image_size=cfg["image_size"],
                class_names=cfg["class_names"],
                gradcam=False,
            )
            assert result["predicted_class"] == expected
            assert set(result["probabilities"]) == APPLE_CLASSES
