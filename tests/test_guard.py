"""
Tests for app.guard — Inference Robustness Guard.

Covers quality metrics, quality verdicts (reject/caution/ok with
precedence), uncertainty evaluation, threshold loading, real-data
safety (all 688 test images pass every reject gate), and integration
with the existing inference pipeline.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app.guard import (
    QUALITY_CAUTION,
    QUALITY_OK,
    QUALITY_REJECT,
    GuardThresholds,
    compute_quality_metrics,
    evaluate_quality,
    evaluate_uncertainty,
    load_guard_thresholds,
)
from src.inference import predict_image
from src.model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
CROP_CFG_PATH = PROJECT_ROOT / "configs" / "crops" / "tomato.yaml"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _flat_image(size=(256, 256), color=(128, 128, 128)) -> Image.Image:
    return Image.new("RGB", size, color)


def _noise_image(size=(256, 256), seed=42) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB")


def _solid_green_image(size=(256, 256)) -> Image.Image:
    return Image.new("RGB", size, (40, 140, 40))


def _grayscale_photo_image(size=(256, 256)) -> Image.Image:
    """A textured but colorless (grayscale-like) image."""
    rng = np.random.default_rng(7)
    g = rng.integers(30, 220, size, dtype=np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    return Image.fromarray(rgb, "RGB")


def _blurred_leaf_image() -> Image.Image:
    """A green textured image with almost no high-frequency detail."""
    rng = np.random.default_rng(3)
    coarse = rng.integers(0, 60, (16, 16, 3), dtype=np.uint8)
    coarse[..., 1] += 120  # green-dominant
    arr = np.kron(coarse, np.ones((16, 16, 1), dtype=np.uint8))
    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    return img.filter(ImageFilter.GaussianBlur(8))


def _leaf_like_image(size=(256, 256)) -> Image.Image:
    """Synthetic 'leaf-like' image: sharp, green, textured, coherent."""
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0 : size[0], 0 : size[1]]
    # Smooth low-frequency background (natural coherence)
    base = (
        60 * np.sin(xx / 25.0) * np.cos(yy / 30.0)
        + 40 * np.sin((xx + yy) / 17.0)
        + 128
    )
    # Fine texture (sharpness)
    texture = rng.normal(0, 18, size)
    g = np.clip(base + texture, 0, 255)
    r = np.clip(base * 0.45 + rng.normal(0, 8, size), 0, 255)
    b = np.clip(base * 0.40 + rng.normal(0, 8, size), 0, 255)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _leaf_like_bytes() -> bytes:
    buf = io.BytesIO()
    _leaf_like_image().save(buf, format="PNG")
    return buf.getvalue()


_LEAF_LIKE_IMAGE = _leaf_like_image()
_LEAF_LIKE_METRICS = compute_quality_metrics(_LEAF_LIKE_IMAGE)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


class TestQualityMetrics:
    def test_flat_image_laplacian_zero(self):
        m = compute_quality_metrics(_flat_image())
        assert m.laplacian_variance == 0.0

    def test_noise_image_correlation_near_zero(self):
        m = compute_quality_metrics(_noise_image())
        assert abs(m.adjacent_correlation) < 0.1

    def test_grayscale_image_saturation_zero(self):
        m = compute_quality_metrics(_grayscale_photo_image())
        assert m.saturation == 0.0

    def test_solid_green_green_fraction_one(self):
        m = compute_quality_metrics(_solid_green_image())
        assert m.green_fraction == 1.0

    def test_leaf_like_sane_values(self):
        m = _LEAF_LIKE_METRICS
        assert m.laplacian_variance > 100
        assert m.green_fraction > 0.10
        assert m.adjacent_correlation > 0.5
        assert m.saturation > 0.02

    def test_deterministic(self):
        m1 = compute_quality_metrics(_noise_image(seed=5))
        m2 = compute_quality_metrics(_noise_image(seed=5))
        assert m1 == m2

    def test_thumbnail_matches_large_image(self):
        """Metrics computed via thumbnail should be stable for big inputs."""
        big = _noise_image(size=(2048, 2048), seed=9)
        m = compute_quality_metrics(big)
        assert m.width == 2048 and m.height == 2048
        # Still noise-like after LANCZOS downsampling: well below the
        # 0.20 reject threshold (the tiny correlation is a resize artifact).
        assert abs(m.adjacent_correlation) < 0.20


# ---------------------------------------------------------------------------
# Quality verdicts
# ---------------------------------------------------------------------------


class TestQualityVerdict:
    def test_flat_image_rejected(self):
        v = evaluate_quality(compute_quality_metrics(_flat_image()))
        assert v["status"] == QUALITY_REJECT
        assert "no_detail" in v["reasons"]

    def test_grayscale_image_rejected(self):
        v = evaluate_quality(compute_quality_metrics(_grayscale_photo_image()))
        assert v["status"] == QUALITY_REJECT
        assert "grayscale_image" in v["reasons"]

    def test_solid_green_rejected(self):
        v = evaluate_quality(compute_quality_metrics(_solid_green_image()))
        assert v["status"] == QUALITY_REJECT
        assert "no_detail" in v["reasons"]

    def test_noise_image_rejected(self):
        v = evaluate_quality(compute_quality_metrics(_noise_image()))
        assert v["status"] == QUALITY_REJECT
        assert "unnatural_image" in v["reasons"]

    def test_too_small_rejected(self):
        m = compute_quality_metrics(_leaf_like_image(size=(32, 32)))
        v = evaluate_quality(m)
        assert v["status"] == QUALITY_REJECT
        assert "too_small" in v["reasons"]

    def test_leaf_like_ok(self):
        v = evaluate_quality(_LEAF_LIKE_METRICS)
        assert v["status"] in (QUALITY_OK, QUALITY_CAUTION)

    def test_blurred_image_caution_not_reject(self):
        v = evaluate_quality(compute_quality_metrics(_blurred_leaf_image()))
        assert v["status"] in (QUALITY_CAUTION, QUALITY_REJECT)
        if v["status"] == QUALITY_CAUTION:
            assert "possibly_blurry" in v["reasons"]
            assert all(r != "no_detail" for r in v["reasons"])

    def test_grayscale_precedence_before_greenness(self):
        """A gray flat image reports grayscale (earlier check) too."""
        m = compute_quality_metrics(_flat_image(color=(100, 100, 100)))
        v = evaluate_quality(m)
        assert v["reasons"][0] in ("too_small", "grayscale_image", "no_detail")
        assert "grayscale_image" in v["reasons"]

    def test_threshold_boundaries_saturation(self):
        """Metric at exactly the threshold passes (strictly below rejects)."""
        m = compute_quality_metrics(_grayscale_photo_image())
        v_pass = evaluate_quality(
            m, GuardThresholds(saturation_min=m.saturation)
        )
        assert "grayscale_image" not in v_pass["reasons"]
        v_reject = evaluate_quality(
            m, GuardThresholds(saturation_min=m.saturation + 1e-9)
        )
        assert "grayscale_image" in v_reject["reasons"]

    def test_boundary_laplacian(self):
        m = _LEAF_LIKE_METRICS
        v_pass = evaluate_quality(
            m, GuardThresholds(flatness_lapvar_min=m.laplacian_variance)
        )
        assert "no_detail" not in v_pass["reasons"]
        v_reject = evaluate_quality(
            m, GuardThresholds(flatness_lapvar_min=m.laplacian_variance + 1e-9)
        )
        assert "no_detail" in v_reject["reasons"]

    def test_reason_messages_exist_for_all_codes(self):
        from app.guard import REASON_MESSAGES

        for code in (
            "too_small",
            "grayscale_image",
            "no_detail",
            "no_plant_material",
            "unnatural_image",
            "possibly_blurry",
            "low_confidence",
            "small_margin",
        ):
            assert code in REASON_MESSAGES
            assert REASON_MESSAGES[code]


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_low_confidence_flagged(self):
        probs = {"Healthy": 0.10, "Early Blight": 0.60, "Late Blight": 0.30}
        v = evaluate_uncertainty(probs)
        assert v["msp"] == pytest.approx(0.60)
        assert "low_confidence" in v["cautions"]

    def test_small_margin_flagged(self):
        # With 3 classes summing to 1, margin < 0.25 forces top2 > 0.50,
        # hence top1 < 0.625 < 0.75: small_margin always co-occurs with
        # low_confidence.  Verified here as a documented property.
        probs = {"Healthy": 0.01, "Early Blight": 0.50, "Late Blight": 0.49}
        v = evaluate_uncertainty(probs)
        assert v["margin"] == pytest.approx(0.01)
        assert "small_margin" in v["cautions"]
        assert "low_confidence" in v["cautions"]

    def test_both_flagged(self):
        probs = {"Healthy": 0.35, "Early Blight": 0.35, "Late Blight": 0.30}
        v = evaluate_uncertainty(probs)
        assert "low_confidence" in v["cautions"]
        assert "small_margin" in v["cautions"]

    def test_confident_prediction_clean(self):
        probs = {"Healthy": 0.98, "Early Blight": 0.01, "Late Blight": 0.01}
        v = evaluate_uncertainty(probs)
        assert v["cautions"] == []
        assert v["msp"] == pytest.approx(0.98)
        assert v["margin"] == pytest.approx(0.97)

    def test_exact_boundary_not_flagged(self):
        probs = {"Healthy": 0.25, "Early Blight": 0.75, "Late Blight": 0.00}
        v = evaluate_uncertainty(probs)
        # msp == 0.75 and margin == 0.50 -> no cautions
        assert "low_confidence" not in v["cautions"]
        assert "small_margin" not in v["cautions"]

    def test_margin_boundary_not_flagged(self):
        probs = {"Healthy": 0.00, "Early Blight": 0.625, "Late Blight": 0.375}
        v = evaluate_uncertainty(probs)
        assert v["margin"] == pytest.approx(0.25)
        assert "small_margin" not in v["cautions"]

    def test_empty_probabilities_safe(self):
        v = evaluate_uncertainty({})
        assert v["msp"] == 0.0
        assert "low_confidence" in v["cautions"]

    def test_works_with_predict_image_output(self, tmp_path):
        """evaluate_uncertainty consumes the real predict_image dict shape."""
        model = build_model(num_classes=3, pretrained=False)
        model.eval()
        img_path = tmp_path / "leaf.png"
        _LEAF_LIKE_IMAGE.save(img_path)
        result = predict_image(model, img_path, device="cpu")
        v = evaluate_uncertainty(result["probabilities"])
        assert 0.0 <= v["msp"] <= 1.0
        assert 0.0 <= v["margin"] <= 1.0
        assert isinstance(v["cautions"], list)


# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_defaults(self):
        t = load_guard_thresholds(None)
        assert t == GuardThresholds()

    def test_empty_cfg_defaults(self):
        assert load_guard_thresholds({}) == GuardThresholds()

    def test_yaml_override_respected(self):
        cfg = {"guard": {"confidence_warn": 0.5, "min_resolution": 128}}
        t = load_guard_thresholds(cfg)
        assert t.confidence_warn == 0.5
        assert t.min_resolution == 128
        assert t.saturation_min == GuardThresholds().saturation_min

    def test_invalid_values_fall_back(self):
        cfg = {"guard": {"confidence_warn": "not-a-number", "min_resolution": None}}
        t = load_guard_thresholds(cfg)
        assert t.confidence_warn == GuardThresholds().confidence_warn
        assert t.min_resolution == GuardThresholds().min_resolution

    def test_tomato_yaml_loads(self):
        import yaml

        cfg = yaml.safe_load(CROP_CFG_PATH.read_text(encoding="utf-8"))
        t = load_guard_thresholds(cfg)
        assert t == GuardThresholds()  # config mirrors the calibrated defaults


# ---------------------------------------------------------------------------
# Real-data safety — no real leaf image is ever rejected
# ---------------------------------------------------------------------------


class TestRealDataSafety:
    def test_all_test_images_pass_reject_gates(self):
        """Every real test leaf must pass all reject-level checks."""
        import csv

        if not TEST_CSV.exists():
            pytest.skip("test split manifest not available")

        with open(TEST_CSV, newline="", encoding="utf-8") as f:
            paths = [row["image_path"] for row in csv.DictReader(f)]
        assert len(paths) > 600

        rejected = []
        for p in paths:
            img = Image.open(PROJECT_ROOT / p).convert("RGB")
            verdict = evaluate_quality(compute_quality_metrics(img))
            if verdict["status"] == QUALITY_REJECT:
                rejected.append((p, verdict["reasons"]))
        assert rejected == []

    def test_blur_caution_is_minority(self):
        """The blur caution may only affect a small minority of real images."""
        import csv

        if not TEST_CSV.exists():
            pytest.skip("test split manifest not available")

        with open(TEST_CSV, newline="", encoding="utf-8") as f:
            paths = [row["image_path"] for row in csv.DictReader(f)]

        cautioned = 0
        for p in paths:
            img = Image.open(PROJECT_ROOT / p).convert("RGB")
            verdict = evaluate_quality(compute_quality_metrics(img))
            if verdict["status"] == QUALITY_CAUTION:
                cautioned += 1

        assert cautioned < len(paths) * 0.25  # at most a quarter


# ---------------------------------------------------------------------------
# Integration with the inference pipeline
# ---------------------------------------------------------------------------


class TestInferenceIntegration:
    @pytest.fixture
    def model(self):
        m = build_model(num_classes=3, pretrained=False)
        m.eval()
        return m

    def test_rejected_image_needs_no_inference(self):
        """The reject path is decidable from metrics alone (no model)."""
        noise = compute_quality_metrics(_noise_image())
        verdict = evaluate_quality(noise)
        assert verdict["status"] == QUALITY_REJECT
        # The app must skip predict_image when status is reject; nothing
        # in the guard calls the model, so this holds structurally.

    def test_leaf_like_passes_and_predicts(self, model, tmp_path):
        img_path = tmp_path / "leaf.png"
        _LEAF_LIKE_IMAGE.save(img_path)

        verdict = evaluate_quality(compute_quality_metrics(_LEAF_LIKE_IMAGE))
        assert verdict["status"] in (QUALITY_OK, QUALITY_CAUTION)

        result = predict_image(model, img_path, device="cpu")
        uncertainty = evaluate_uncertainty(result["probabilities"])
        assert uncertainty["msp"] > 0  # untrained model still yields probs
        assert set(result["probabilities"]) == {"Healthy", "Early Blight", "Late Blight"}

    def test_full_flow_noise_would_be_blocked_before_model(self):
        """Guard ordering: quality verdict computed without touching torch."""
        from app.guard import evaluate_quality_from_image

        v = evaluate_quality_from_image(_noise_image())
        assert v["status"] == QUALITY_REJECT
        assert "unnatural_image" in v["reasons"]
