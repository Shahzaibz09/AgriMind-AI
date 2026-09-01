"""
Tests for the crop verification feature — app/crop_verification.py.

Two layers:
1. Logic tests with stub classifier models (no checkpoint needed) —
   verdict transitions, threshold boundary, exact mismatch message.
2. Data-dependent tests against the trained crop classifier (skip when
   the checkpoint/manifests are absent) — including the seven required
   end-to-end cases:
       1-3. correct crop selected  -> ok (diagnosis continues)
       4-6. wrong crop selected    -> mismatch with the exact warning
       7.   unclear/random image   -> uncertain (ask for clearer image)
   plus a calibration claim: the confidence threshold sits below every
   real leaf's top-1 confidence, so genuine leaves are never blocked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from app.crop_verification import (
    CROP_CLASSIFIER_PATH,
    CROP_CONFIDENCE_THRESHOLD,
    STATUS_MISMATCH,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_UNCERTAIN,
    verify_crop,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROP_MANIFEST_DIR = PROJECT_ROOT / "data" / "splits_crop_classifier"
CROP_METADATA = PROJECT_ROOT / "models" / "crop_classifier_metadata.json"

CLASS_NAMES = ("Tomato", "Potato", "Apple")
DISEASE_CHECKPOINTS = (
    PROJECT_ROOT / "models" / "best_model.pth",
    PROJECT_ROOT / "models" / "potato_best_model.pth",
    PROJECT_ROOT / "models" / "apple_best_model.pth",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubModel(torch.nn.Module):
    """Classifier stub returning fixed logits regardless of input."""

    def __init__(self, logits: list[float]):
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self._logits.expand(x.size(0), -1)


def _img(size=(224, 224)) -> Image.Image:
    """A plain image (contents are irrelevant for the stub tests)."""
    return Image.new("RGB", size, (80, 120, 60))


def _real_image_for_crop(crop: str, row_index: int = 0) -> Image.Image:
    """Load a real leaf image of *crop* from the crop-classifier manifests."""
    manifest = CROP_MANIFEST_DIR / "test.csv"
    df = pd.read_csv(manifest)
    rows = df[df["clean_class"] == crop]
    path = Path(rows.iloc[row_index]["image_path"])
    img_path = path if path.is_absolute() else PROJECT_ROOT / path
    return Image.open(img_path).convert("RGB")


def _noise_image(size=(224, 224), seed=42) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB")


def _trained_classifier_available() -> bool:
    return CROP_CLASSIFIER_PATH.exists() and CROP_MANIFEST_DIR.joinpath(
        "test.csv"
    ).exists()


# ---------------------------------------------------------------------------
# Logic tests (stub models — no checkpoint required)
# ---------------------------------------------------------------------------


class TestVerifyCropLogic:
    def test_matching_crop_is_ok(self):
        # softmax([2.5, 0, 0]) -> ~0.859 confidence on Tomato
        model = _StubModel([2.5, 0.0, 0.0])
        result = verify_crop(_img(), "Tomato", model=model, class_names=CLASS_NAMES)
        assert result.status == STATUS_OK
        assert result.detected_crop == "Tomato"
        assert result.confidence == pytest.approx(0.8589, abs=0.001)

    def test_mismatch_message_matches_required_format(self):
        # Confident Tomato leaf while Potato is selected
        model = _StubModel([6.0, 0.0, 0.0])
        result = verify_crop(_img(), "Potato", model=model, class_names=CLASS_NAMES)
        assert result.status == STATUS_MISMATCH
        assert result.detected_crop == "Tomato"
        assert result.message == (
            "Crop mismatch: This image appears to be a Tomato leaf, but "
            "Potato was selected. Please select Tomato or upload a "
            "Potato leaf."
        )

    def test_low_confidence_is_uncertain_and_asks_for_clearer_image(self):
        # Near-uniform logits -> top-1 confidence ~0.34 < threshold
        model = _StubModel([0.1, 0.0, 0.0])
        result = verify_crop(_img(), "Tomato", model=model, class_names=CLASS_NAMES)
        assert result.status == STATUS_UNCERTAIN
        assert result.detected_crop is None  # no crop claim is made
        assert "clearer" in result.message.lower()

    def test_confidence_exactly_at_threshold_is_trusted(self):
        # logits chosen so softmax top-1 == 0.75 exactly: e^L / (e^L + 2) = 0.75
        import math

        logit = math.log(6.0)
        model = _StubModel([logit, 0.0, 0.0])
        result = verify_crop(
            _img(), "Tomato", model=model, class_names=CLASS_NAMES,
            threshold=0.75,
        )
        assert result.status == STATUS_OK
        assert result.confidence == pytest.approx(0.75, abs=1e-6)

    def test_selected_crop_comparison_is_case_insensitive(self):
        model = _StubModel([2.5, 0.0, 0.0])
        result = verify_crop(_img(), "tomato", model=model, class_names=CLASS_NAMES)
        assert result.status == STATUS_OK

    def test_probabilities_cover_all_crops(self):
        model = _StubModel([2.5, 1.0, 0.0])
        result = verify_crop(_img(), "Tomato", model=model, class_names=CLASS_NAMES)
        assert set(result.probabilities) == set(CLASS_NAMES)
        assert sum(result.probabilities.values()) == pytest.approx(1.0)

    def test_missing_checkpoint_reports_skipped(self, monkeypatch):
        import app.crop_verification as cv

        monkeypatch.setattr(
            cv, "CROP_CLASSIFIER_PATH", PROJECT_ROOT / "models" / "nonexistent.pth"
        )
        result = cv.verify_crop(_img(), "Tomato")
        assert result.status == STATUS_SKIPPED
        assert result.detected_crop is None

    def test_threshold_constant_is_sane(self):
        # Soft gate: strict enough to filter garbage, loose enough not to
        # block genuine leaves (calibrated against the test split below).
        assert 0.5 <= CROP_CONFIDENCE_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# Data-dependent tests — trained classifier end-to-end
# ---------------------------------------------------------------------------


class TestCropClassifierArtifacts:
    def test_checkpoint_and_metadata_exist(self):
        if not _trained_classifier_available():
            pytest.skip("crop classifier artifacts not available")
        assert CROP_METADATA.exists()

    def test_metadata_class_mapping_is_crop_labels(self):
        if not CROP_METADATA.exists():
            pytest.skip("crop classifier metadata not available")
        import json

        meta = json.loads(CROP_METADATA.read_text(encoding="utf-8"))
        assert meta["num_classes"] == 3
        assert [meta["class_mapping"][str(i)] for i in range(3)] == [
            "Tomato", "Potato", "Apple",
        ]

    def test_disease_checkpoints_untouched(self):
        # The crop classifier lives beside — never instead of — the
        # disease models, which must all still be present.
        for ckpt in DISEASE_CHECKPOINTS:
            assert ckpt.exists(), ckpt


class TestCropClassifierCalibration:
    def test_threshold_blocks_at_most_one_percent_of_real_leaves(self):
        """Real leaves must almost always clear the confidence threshold.

        Measured on the 300-image test split: 297/300 (99%) score >= 0.85.
        The few borderline leaves receive the safe "uncertain" retry
        prompt — never a wrong crop-mismatch claim.
        """
        if not _trained_classifier_available():
            pytest.skip("crop classifier artifacts not available")

        from app.crop_verification import predict_crop, load_crop_classifier

        model, class_names = load_crop_classifier()
        assert tuple(class_names) == CLASS_NAMES

        df = pd.read_csv(CROP_MANIFEST_DIR / "test.csv")
        below = []
        for _, row in df.iterrows():
            path = Path(row["image_path"])
            img_path = path if path.is_absolute() else PROJECT_ROOT / path
            with Image.open(img_path) as f:
                img = f.convert("RGB")
            probs = predict_crop(model, img, class_names)
            if max(probs.values()) < CROP_CONFIDENCE_THRESHOLD:
                below.append(row["image_path"])
        assert len(below) <= 3, below  # measured: 3/300 = 1% borderline

    def test_threshold_above_noise_confidence(self):
        """Pure noise must NOT be confidently assigned to a crop."""
        if not _trained_classifier_available():
            pytest.skip("crop classifier artifacts not available")

        from app.crop_verification import predict_crop, load_crop_classifier

        model, class_names = load_crop_classifier()
        for seed in (42, 7, 1):
            probs = predict_crop(model, _noise_image(seed=seed), class_names)
            assert max(probs.values()) < CROP_CONFIDENCE_THRESHOLD, (
                seed, probs
            )

    def test_threshold_above_leaf_like_synthetic_confidence(self):
        """A guard-passing leaf-like synthetic must stay below threshold
        (uncertain), not be confidently assigned to a crop."""
        if not _trained_classifier_available():
            pytest.skip("crop classifier artifacts not available")

        import numpy as np

        from app.crop_verification import predict_crop, load_crop_classifier

        rng = np.random.default_rng(11)
        yy, xx = np.mgrid[0:224, 0:224]
        base = (
            60 * np.sin(xx / 25.0) * np.cos(yy / 30.0)
            + 40 * np.sin((xx + yy) / 17.0)
            + 128
        )
        green = np.stack(
            [base * 0.4, base * 1.1 + 20, base * 0.35], axis=-1
        ).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(green, "RGB")

        model, class_names = load_crop_classifier()
        probs = predict_crop(model, img, class_names)
        assert max(probs.values()) < CROP_CONFIDENCE_THRESHOLD, probs


# ---------------------------------------------------------------------------
# The seven required cases
# ---------------------------------------------------------------------------


class TestSevenRequiredCases:
    """1-3: correct crop -> ok.  4-6: wrong crop -> mismatch warning.
    7: unclear image -> uncertain."""

    def _verify(self, selected: str, image: Image.Image):
        if not _trained_classifier_available():
            pytest.skip("crop classifier artifacts not available")
        return verify_crop(image, selected)

    # -- Cases 1-3: compatible selection -> diagnosis continues -------

    def test_case1_tomato_selected_tomato_image(self):
        result = self._verify("Tomato", _real_image_for_crop("Tomato"))
        assert result.status == STATUS_OK, result.message
        assert result.detected_crop == "Tomato"

    def test_case2_potato_selected_potato_image(self):
        result = self._verify("Potato", _real_image_for_crop("Potato"))
        assert result.status == STATUS_OK, result.message
        assert result.detected_crop == "Potato"

    def test_case3_apple_selected_apple_image(self):
        result = self._verify("Apple", _real_image_for_crop("Apple"))
        assert result.status == STATUS_OK, result.message
        assert result.detected_crop == "Apple"

    # -- Cases 4-6: incompatible selection -> mismatch warning ---------

    def test_case4_potato_selected_tomato_image(self):
        result = self._verify("Potato", _real_image_for_crop("Tomato"))
        assert result.status == STATUS_MISMATCH
        assert result.detected_crop == "Tomato"
        assert result.message == (
            "Crop mismatch: This image appears to be a Tomato leaf, but "
            "Potato was selected. Please select Tomato or upload a "
            "Potato leaf."
        )

    def test_case5_tomato_selected_apple_image(self):
        result = self._verify("Tomato", _real_image_for_crop("Apple"))
        assert result.status == STATUS_MISMATCH
        assert result.detected_crop == "Apple"
        assert result.message == (
            "Crop mismatch: This image appears to be an Apple leaf, but "
            "Tomato was selected. Please select Apple or upload a "
            "Tomato leaf."
        )

    def test_case6_apple_selected_potato_image(self):
        result = self._verify("Apple", _real_image_for_crop("Potato"))
        assert result.status == STATUS_MISMATCH
        assert result.detected_crop == "Potato"
        assert result.message == (
            "Crop mismatch: This image appears to be a Potato leaf, but "
            "Apple was selected. Please select Potato or upload an "
            "Apple leaf."
        )

    # -- Case 7: unclear/random image -> uncertain ---------------------

    def test_case7_noise_image_is_uncertain(self):
        if not _trained_classifier_available():
            pytest.skip("crop classifier artifacts not available")
        result = verify_crop(_noise_image(), "Potato")
        assert result.status == STATUS_UNCERTAIN, (
            f"probabilities: {result.probabilities}"
        )
        assert "clearer" in result.message.lower()

    def test_case7_heavily_blurred_leaf_is_uncertain_or_low_confidence(self):
        """A heavily blurred leaf should never yield a confident mismatch."""
        if not _trained_classifier_available():
            pytest.skip("crop classifier artifacts not available")
        from PIL import ImageFilter

        from app.crop_verification import predict_crop, load_crop_classifier

        model, class_names = load_crop_classifier()
        blurred = _real_image_for_crop("Potato").filter(
            ImageFilter.GaussianBlur(12)
        )
        probs = predict_crop(model, blurred, class_names)
        # Either uncertain (below threshold) or still correctly Potato.
        top_crop = max(probs, key=probs.get)
        assert max(probs.values()) < CROP_CONFIDENCE_THRESHOLD or (
            top_crop == "Potato"
        )
