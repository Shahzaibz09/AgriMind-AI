"""
AgriMind AI — Inference Robustness Guard.

Lightweight real-world robustness layer around the existing inference
pipeline.  Two pure-function gates, both CPU-only and dependency-free
(numpy + Pillow only):

1. **Quality gate** (pre-inference): rejects images that clearly differ
   from the PlantVillage training distribution before the model runs —
   grayscale photos/screenshots, flat/empty frames, images without any
   plant-like green content, and random noise.
2. **Uncertainty gate** (post-inference): flags low-confidence and
   small-margin predictions with caution messages instead of blocking.

Thresholds were calibrated against the full 688-image test
distribution (every reject threshold sits at or below half the
real-data minimum, so no real leaf image is rejected):

==========================  ========  ========  ========
metric                      real min  real p1   threshold
==========================  ========  ========  ========
Laplacian variance          41.2      159.7     10 (reject)
Green fraction (ExG>0.10)   0.025     0.076     0.01 (reject)
Adjacent-pixel correlation  0.297     0.411     0.20 (reject)
Saturation                  0.0405    0.0505    0.02 (reject)
==========================  ========  ========  ========

Known limitations (documented deliberately)
-------------------------------------------
- Adversarially crafted "plant-like" images (e.g. random green-dominant
  color blocks) can pass every quality gate; catching those requires a
  learned OOD detector, which is out of MVP scope.
- In-distribution Early-Blight/Late-Blight confusion errors run up to
  99.8% confidence and pass every check here.  That is model quality,
  not an input-quality problem, and is covered by the AI disclaimer.
- The gate verifies "plant-like photo", **not** "tomato leaf": a healthy
  leaf of another species will pass and receive a tomato-class answer.
- Real field photos (soil background, hand-held leaves, multiple
  leaves) differ from lab-style PlantVillage images but are exactly
  what farmers upload; they are plant-like, so they pass by design —
  distribution shift surfaces as lower confidence (uncertainty gate),
  not as rejection.

This module has no Streamlit dependency and never modifies the model,
checkpoint, or inference pipeline: the app calls it before/after the
existing ``predict_image``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Verdict statuses
QUALITY_OK = "ok"
QUALITY_CAUTION = "caution"
QUALITY_REJECT = "reject"

# Metrics are computed on a thumbnail of at most this size to bound
# memory and compute for large phone photos.
_MAX_METRIC_SIZE = 1024

# Excess-Green threshold for the "plant-like green pixel" test.
_EXG_THRESHOLD = 0.10

# Human-readable messages for each reason code (rendered in the UI).
REASON_MESSAGES: dict[str, str] = {
    "unreadable_image": "The uploaded file could not be read as an image.",
    "too_small": "The image is too small (under 64 pixels on a side).",
    "grayscale_image": "The image appears to be grayscale or black-and-white.",
    "no_detail": "The image appears flat or empty (no visible detail).",
    "no_plant_material": "No plant-like green content was found in the image.",
    "unnatural_image": "The image looks like random noise rather than a photo.",
    "possibly_blurry": "The image may be blurry or lacking fine detail; "
    "the prediction may be less reliable.",
    "low_confidence": "The model is not fully confident in this prediction "
    "(confidence below 75%).",
    "small_margin": "This was a close call between the two most likely "
    "classes.",
}

# Guard-threshold keys expected in a crop config ``guard:`` section
# (attribute name -> YAML key).
_THRESHOLD_YAML_KEYS = {
    "min_resolution": "min_resolution",
    "saturation_min": "saturation_min",
    "flatness_lapvar_min": "flatness_lapvar_min",
    "green_fraction_min": "green_fraction_min",
    "correlation_min": "correlation_min",
    "blur_lapvar_warn": "blur_lapvar_warn",
    "confidence_warn": "confidence_warn",
    "margin_warn": "margin_warn",
}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardThresholds:
    """Calibrated thresholds for the robustness guard.

    Reject-level values sit at or below half the real-data minimum
    (see module docstring) so legitimate leaf images are never
    rejected.  Caution-level values only trigger warnings.
    """

    # -- reject-level quality checks ------------------------------------
    min_resolution: int = 64
    saturation_min: float = 0.02
    flatness_lapvar_min: float = 10.0
    green_fraction_min: float = 0.01
    correlation_min: float = 0.20

    # -- caution-level checks -------------------------------------------
    blur_lapvar_warn: float = 300.0
    confidence_warn: float = 0.75
    margin_warn: float = 0.25


def load_guard_thresholds(crop_cfg: dict[str, Any] | None) -> GuardThresholds:
    """Build thresholds from a crop config's optional ``guard:`` section.

    Missing section, missing keys, or invalid values fall back to the
    calibrated defaults; the guard never fails because of config
    problems.
    """
    defaults = GuardThresholds()
    if not crop_cfg:
        return defaults

    section = crop_cfg.get("guard")
    if not isinstance(section, dict):
        return defaults

    values: dict[str, Any] = {}
    for attr, key in _THRESHOLD_YAML_KEYS.items():
        if key not in section:
            continue
        raw = section[key]
        expected_type = float if isinstance(getattr(defaults, attr), float) else int
        try:
            value = expected_type(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid guard threshold %s=%r (expected %s)",
                key, raw, expected_type.__name__,
            )
            continue
        values[attr] = value

    return GuardThresholds(**values)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityMetrics:
    """Image-space quality metrics used by the quality gate.

    Attributes
    ----------
    width, height : int
        Original image dimensions.
    laplacian_variance : float
        Variance of the Laplacian on grayscale (higher = sharper).
    green_fraction : float
        Fraction of pixels with Excess-Green (2G-R-B) above 0.10.
    adjacent_correlation : float
        Correlation of horizontally adjacent grayscale pixels
        (natural photos are high; noise is near zero).
    saturation : float
        Mean (|R-G| + |G-B|) in [0, 2]; ~0 for grayscale images.
    """

    width: int
    height: int
    laplacian_variance: float
    green_fraction: float
    adjacent_correlation: float
    saturation: float


def compute_quality_metrics(pil_image: Image.Image) -> QualityMetrics:
    """Compute quality metrics on a (<=1024px) thumbnail of the image.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Opened RGB-converted image.

    Returns
    -------
    QualityMetrics computed on the thumbnail (dimensions reflect the
    original image).
    """
    width, height = pil_image.size

    thumb = pil_image
    if max(width, height) > _MAX_METRIC_SIZE:
        scale = _MAX_METRIC_SIZE / max(width, height)
        thumb = pil_image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.LANCZOS,
        )

    gray = np.asarray(thumb.convert("L"), dtype=np.float64)
    rgb = np.asarray(thumb.convert("RGB"), dtype=np.float64) / 255.0

    # Laplacian variance (sharpness / detail)
    lap = (
        np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
        - 4 * gray
    )
    laplacian_variance = float(lap.var())

    # Green fraction (plant-like content)
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    green_fraction = float((exg > _EXG_THRESHOLD).mean())

    # Adjacent-pixel correlation (natural photo coherence)
    a = gray[:, :-1].ravel()
    b = gray[:, 1:].ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        adjacent_correlation = 1.0  # constant image — flatness check handles it
    else:
        adjacent_correlation = float(np.corrcoef(a, b)[0, 1])

    # Saturation (grayscale detection)
    saturation = float(
        (np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])).mean()
    )

    return QualityMetrics(
        width=width,
        height=height,
        laplacian_variance=laplacian_variance,
        green_fraction=green_fraction,
        adjacent_correlation=adjacent_correlation,
        saturation=saturation,
    )


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def evaluate_quality(
    metrics: QualityMetrics,
    thresholds: GuardThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate image quality against the thresholds.

    Parameters
    ----------
    metrics : QualityMetrics
        Metrics from :func:`compute_quality_metrics`.
    thresholds : GuardThresholds, optional
        Thresholds (defaults when ``None``).

    Returns
    -------
    dict with:
        - ``status``: ``"ok"``, ``"caution"``, or ``"reject"``
        - ``reasons``: list of reason codes (reject reasons when
          rejecting, caution reasons otherwise)
    """
    if thresholds is None:
        thresholds = GuardThresholds()

    reject_reasons: list[str] = []
    caution_reasons: list[str] = []

    if metrics.width < thresholds.min_resolution or metrics.height < thresholds.min_resolution:
        reject_reasons.append("too_small")
    if metrics.saturation < thresholds.saturation_min:
        reject_reasons.append("grayscale_image")
    if metrics.laplacian_variance < thresholds.flatness_lapvar_min:
        reject_reasons.append("no_detail")
    if metrics.green_fraction < thresholds.green_fraction_min:
        reject_reasons.append("no_plant_material")
    if metrics.adjacent_correlation < thresholds.correlation_min:
        reject_reasons.append("unnatural_image")

    if reject_reasons:
        return {"status": QUALITY_REJECT, "reasons": reject_reasons}

    if metrics.laplacian_variance < thresholds.blur_lapvar_warn:
        caution_reasons.append("possibly_blurry")

    if caution_reasons:
        return {"status": QUALITY_CAUTION, "reasons": caution_reasons}
    return {"status": QUALITY_OK, "reasons": []}


def evaluate_quality_from_image(
    pil_image: Image.Image,
    thresholds: GuardThresholds | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: compute metrics then evaluate."""
    return evaluate_quality(compute_quality_metrics(pil_image), thresholds)


# ---------------------------------------------------------------------------
# Uncertainty gate
# ---------------------------------------------------------------------------


def evaluate_uncertainty(
    probabilities: dict[str, float],
    thresholds: GuardThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate prediction uncertainty from a probability dict.

    Works directly on the ``probabilities`` dict returned by
    ``src.inference.predict_image`` — no pipeline changes.

    Returns
    -------
    dict with:
        - ``msp``: maximum softmax probability (top-1 confidence)
        - ``margin``: top1 minus top2 probability
        - ``cautions``: list of reason codes (possibly empty)
    """
    if thresholds is None:
        thresholds = GuardThresholds()

    if not probabilities:
        return {"msp": 0.0, "margin": 0.0, "cautions": ["low_confidence", "small_margin"]}

    probs = sorted((float(p) for p in probabilities.values()), reverse=True)
    msp = probs[0]
    margin = probs[0] - probs[1] if len(probs) > 1 else probs[0]

    cautions: list[str] = []
    if msp < thresholds.confidence_warn:
        cautions.append("low_confidence")
    if margin < thresholds.margin_warn:
        cautions.append("small_margin")

    return {"msp": msp, "margin": margin, "cautions": cautions}
