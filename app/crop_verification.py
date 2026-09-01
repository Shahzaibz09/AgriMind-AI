"""
AgriMind AI — Crop Verification.

Pre-diagnosis check: decides whether the uploaded image is compatible
with the crop the user selected in the app.  A lightweight 3-class
classifier (Tomato / Potato / Apple, MobileNetV3-Small) trained on the
existing PlantVillage crop data powers the check — see
``src/build_crop_classifier_dataset.py`` and
``configs/config_crop_classifier.yaml``.

Verdicts
--------
ok        Detected crop matches the selected crop (confidence >=
           threshold) — disease analysis may continue.
mismatch  Confidently a *different* crop — diagnosis must stop and the
           user is told which crop the image appears to be.
uncertain Confidence below the threshold — no crop claim is made; the
           user is asked for a clearer image.
skipped   Classifier checkpoint missing — verification disabled (MVP
           fallback so a fresh clone still runs).

This module is Streamlit-independent so it can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.dataset import get_eval_transforms
from src.model import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROP_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "crop_classifier.pth"
IMAGE_SIZE = 224

# Confidence threshold for trusting the crop classifier's top-1 label.
# Calibrated on the crop-classifier test split (300 real leaves, 100 per
# crop) against out-of-distribution fixtures:
#   - 99% of real leaves (297/300) score >= 0.85 (min 0.739); the 3
#     borderline leaves fall into the safe "uncertain" prompt instead of
#     a wrong crop claim.
#   - Out-of-distribution inputs all score BELOW 0.85 -> uncertain:
#     random noise <= 0.821, leaf-like synthetic 0.729, heavily blurred
#     leaf 0.527, dark image 0.560.  (Pure noise is usually rejected
#     even earlier by the image-quality guard in app/app.py.)
# Never claim certainty — this is a soft, confidence-gated check.
CROP_CONFIDENCE_THRESHOLD = 0.85

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_UNCERTAIN = "uncertain"
STATUS_SKIPPED = "skipped"


@dataclass
class CropVerificationResult:
    """Outcome of a crop verification check."""

    status: str
    selected_crop: str
    detected_crop: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    message: str = ""


@lru_cache(maxsize=1)
def load_crop_classifier(
    checkpoint_path: str | None = None,
) -> tuple[Any, tuple[str, ...]]:
    """Load the crop classifier and its ordered class names (cached).

    The class-name order is read from the checkpoint's training metadata
    (``metadata.class_mapping``), so predictions are always labelled with
    the order the model was trained on.
    """
    path = Path(checkpoint_path) if checkpoint_path else CROP_CLASSIFIER_PATH
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mapping = (checkpoint.get("metadata") or {}).get("class_mapping") or {}
    class_names = tuple(
        str(mapping[key]) for key in sorted(mapping, key=int)
    )
    if not class_names:
        raise ValueError(f"No class mapping in checkpoint: {path}")
    model = load_checkpoint(str(path), device="cpu")
    model.eval()
    return model, class_names


def predict_crop(
    model: Any,
    image: Image.Image,
    class_names: tuple[str, ...] | list[str],
    image_size: int = IMAGE_SIZE,
) -> dict[str, float]:
    """Return crop-name -> softmax probability for a PIL image."""
    transform = get_eval_transforms(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {name: float(probs[i]) for i, name in enumerate(class_names)}


def verify_crop(
    image: Image.Image,
    selected_crop: str,
    *,
    model: Any | None = None,
    class_names: tuple[str, ...] | list[str] | None = None,
    threshold: float = CROP_CONFIDENCE_THRESHOLD,
) -> CropVerificationResult:
    """Verify that *image* is compatible with *selected_crop*.

    Parameters
    ----------
    image : PIL.Image.Image
        The uploaded image (RGB conversion is handled internally).
    selected_crop : str
        Crop name selected in the app (e.g. "Potato"); compared
        case-insensitively against the classifier's labels.
    model, class_names : optional
        Pre-loaded classifier (used by tests to inject stub models).
        When omitted, the real classifier is loaded — or the check is
        skipped if the checkpoint is absent.
    threshold : float
        Minimum top-1 confidence to trust a crop label.

    Returns
    -------
    CropVerificationResult with status ok / mismatch / uncertain / skipped.
    """
    selected = str(selected_crop).strip()

    if model is None:
        if not CROP_CLASSIFIER_PATH.exists():
            return CropVerificationResult(
                status=STATUS_SKIPPED,
                selected_crop=selected,
                message="Crop classifier unavailable — verification skipped.",
            )
        model, class_names = load_crop_classifier()

    probabilities = predict_crop(model, image, class_names)
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    detected, confidence = ranked[0]

    if confidence < threshold:
        return CropVerificationResult(
            status=STATUS_UNCERTAIN,
            selected_crop=selected,
            detected_crop=None,
            confidence=confidence,
            probabilities=probabilities,
            message=(
                "This image could not be recognized confidently as a "
                "Tomato, Potato, or Apple leaf. Please upload a clearer, "
                "close-up photo of a single leaf and try again."
            ),
        )

    if detected.strip().lower() != selected.lower():
        det_article = "an" if detected[:1].lower() in "aeiou" else "a"
        sel_article = "an" if selected[:1].lower() in "aeiou" else "a"
        return CropVerificationResult(
            status=STATUS_MISMATCH,
            selected_crop=selected,
            detected_crop=detected,
            confidence=confidence,
            probabilities=probabilities,
            message=(
                f"Crop mismatch: This image appears to be {det_article} "
                f"{detected} leaf, but {selected} was selected. Please "
                f"select {detected} or upload {sel_article} {selected} "
                f"leaf."
            ),
        )

    return CropVerificationResult(
        status=STATUS_OK,
        selected_crop=selected,
        detected_crop=detected,
        confidence=confidence,
        probabilities=probabilities,
        message=f"Crop check passed: this looks like a {detected} leaf.",
    )
