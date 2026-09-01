"""
AgriMind AI — Real-World Validation Harness (Tier 1).

Systematic validation of the frozen MVP across 10 real-world image
categories.  Drives the exact production code paths in the same order
as ``app/app.py``:

    load_guard_thresholds -> compute_quality_metrics -> evaluate_quality
    -> (reject: stop) -> predict_image(gradcam=True, save_gradcam=True)
    -> evaluate_uncertainty

No project files are modified; the harness only reads the checkpoint,
crop config, and data splits, and writes new artifacts under
``reports/validation/``.

Fixture provenance:
    - test-split    : held-out images from data/splits/test.csv
    - derived       : deterministic transforms of a real leaf
                      (blur / darkening / downscale / JPEG / composite)
    - synthetic     : procedurally generated (seeded)
    - user-supplied : validation/fixtures/<NN>_*/ drop-in photos

Verdicts:
    PASS             behavior matches the approved expectation matrix
    DEVIATION        behavior differs (empirical finding, not hidden)
    KNOWN-LIMITATION documented guard/model limitation confirmed
    INFO             sweep/boundary measurement row

Usage::

    .venv\\Scripts\\python.exe validation\\validate.py
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Path bootstrap (same pattern as app/app.py)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.crop_config import get_crop_config  # noqa: E402
from app.guard import (  # noqa: E402
    compute_quality_metrics,
    evaluate_quality,
    evaluate_uncertainty,
    load_guard_thresholds,
)
from src.inference import predict_image  # noqa: E402
from src.model import load_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROP_NAME = "Tomato"
CROPS_DIR = _PROJECT_ROOT / "configs" / "crops"
TEST_CSV = _PROJECT_ROOT / "data" / "splits" / "test.csv"

VALIDATION_DIR = _PROJECT_ROOT / "reports" / "validation"
FIXTURE_DIR = VALIDATION_DIR / "fixtures"
GRADCAM_DIR = VALIDATION_DIR / "gradcam"
RESULTS_JSON = VALIDATION_DIR / "validation_results.json"
REPORT_MD = VALIDATION_DIR / "validation_report.md"

USER_FIXTURE_DIR = _PROJECT_ROOT / "validation" / "fixtures"

_SAMPLE_SEED = 42
_PER_CLASS_N = 5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------------------
# Expectation table (approved validation matrix)
# ---------------------------------------------------------------------------

_EXP_REAL_LEAF = {
    "guard_status": ["ok", "caution"],
    "require_reasons": [],
    "label": "pass",
    "note": "held-out test-split leaf: caution (blur) acceptable by "
            "design, reject would be a failure",
}
_EXP_BLUR_MODERATE = {
    "guard_status": ["caution"],
    "require_reasons": [],
    "label": "pass",
    "note": "moderate blur should warn (possibly_blurry), never block",
}
_EXP_BLUR_EXTREME = {
    "guard_status": ["reject"],
    "require_reasons": ["no_detail"],
    "label": "pass",
    "note": "extreme blur should be rejected before inference",
}
_EXP_DARK_MODERATE = {
    "guard_status": ["ok", "caution"],
    "require_reasons": [],
    "label": "pass",
    "note": "poor lighting: measured outcome; caution or pass expected",
}
_EXP_DARK_EXTREME = {
    "guard_status": ["reject"],
    "require_reasons": ["no_plant_material"],
    "label": "pass",
    "note": "extreme darkness collapses ExG/saturation: rejection "
            "expected (extra reasons acceptable)",
}
_EXP_NOISE = {
    "guard_status": ["reject"],
    "require_reasons": ["unnatural_image"],
    "label": "pass",
    "note": "uniform noise has green fraction ~0.4, so rejection must "
            "rest on near-zero adjacent-pixel correlation alone",
}
_EXP_NONLEAF_SCENE = {
    "guard_status": ["reject"],
    "require_reasons": ["no_plant_material"],
    "label": "pass",
    "note": "structured but green-free scene: rejected for missing "
            "plant material",
}
_EXP_GREEN_NONLEAF = {
    "guard_status": ["ok", "caution"],
    "require_reasons": [],
    "label": "known_limitation",
    "note": "guard verifies plant-like photo, not leaf identity: green "
            "non-leaf texture passes and receives a confident closed-"
            "world answer (documented limitation)",
}
_EXP_OTHER_PLANT = {
    "guard_status": ["ok", "caution"],
    "require_reasons": [],
    "label": "known_limitation",
    "note": "guard is not species-aware: a non-tomato leaf passes and "
            "receives a tomato-class answer (documented limitation)",
}
_EXP_PHONE_STYLE = {
    "guard_status": ["ok", "caution"],
    "require_reasons": [],
    "label": "pass",
    "note": "field-style photos must never be rejected; distribution "
            "shift may surface as lower confidence or cautions",
}
_EXP_AMBIG_MILD = {
    "guard_status": ["ok", "caution"],
    "require_reasons": [],
    "label": "pass",
    "note": "low-quality leaf: accepted, uncertainty messaging may fire",
}
_EXP_AMBIG_SEVERE = {
    "guard_status": ["reject", "caution"],
    "require_reasons": [],
    "label": "pass",
    "note": "tolerant expectation: blocked or warned, both consistent "
            "with the graduated-response design",
}

# ---------------------------------------------------------------------------
# Test-split sampling
# ---------------------------------------------------------------------------


def _load_test_split() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(TEST_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["image_path"], row["clean_class"]))
    return rows


def _sample_per_class(n: int = _PER_CLASS_N, seed: int = _SAMPLE_SEED):
    by_class: dict[str, list[str]] = {}
    for path, cls in _load_test_split():
        by_class.setdefault(cls, []).append(path)
    rng = random.Random(seed)
    return {
        cls: rng.sample(sorted(paths), min(n, len(paths)))
        for cls, paths in sorted(by_class.items())
    }


# ---------------------------------------------------------------------------
# Image transforms and synthetic generators (all seeded/deterministic)
# ---------------------------------------------------------------------------


def _blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img.copy()
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def _darken(img: Image.Image, k: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float64) * k
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _gen_noise(size: int = 300, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _gen_nonleaf_scene(size: int = 256) -> Image.Image:
    """Structured scene with sky, sun, clouds and soil — zero plant green."""
    y, x = np.mgrid[0:size, 0:size]
    yn, xn = y / size, x / size
    horizon = size * 0.62

    img = np.zeros((size, size, 3), dtype=np.float64)
    t = np.clip(y / horizon, 0, 1)[..., None]
    sky = np.array([0.42, 0.63, 0.86]) * (1 - t) + np.array(
        [0.80, 0.88, 0.96]
    ) * t
    tb = np.clip((y - horizon) / (size - horizon), 0, 1)[..., None]
    soil = np.array([0.60, 0.46, 0.33]) * (1 - tb) + np.array(
        [0.36, 0.26, 0.17]
    ) * tb
    img = np.where((y <= horizon)[..., None], sky, soil)

    # Pale sun (ExG of pale yellow-white stays below the green threshold)
    d = np.sqrt((xn - 0.22) ** 2 + (yn - 0.16) ** 2)
    img[d < 0.05] = [1.00, 0.94, 0.84]
    # Two pale clouds
    for cx, cy, r in [(0.55, 0.14, 0.11), (0.78, 0.24, 0.07)]:
        dc = np.sqrt(((xn - cx) * 1.7) ** 2 + (yn - cy) ** 2)
        img[dc < r] = [0.93, 0.95, 0.97]
    # Fine horizontal soil furrows (detail without green)
    soil_rows = y > horizon
    img[soil_rows] += (
        0.045 * np.sin(y[soil_rows] * 1.2)
    )[:, None] * np.array([-1.0, -0.9, -0.7])

    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def _gen_green_texture(size: int = 256, seed: int = 5) -> Image.Image:
    """Grass-like vertical green bands — plant-like but not a leaf."""
    x = np.arange(size, dtype=np.float64)
    r = 0.28 + 0.05 * np.sin(x / 11.0 + 0.6)
    g = 0.52 + 0.10 * np.sin(x / 8.5) + 0.04 * np.sin(x / 3.3 + 1.7)
    b = 0.20 + 0.03 * np.sin(x / 9.5 + 2.2)

    img = np.zeros((size, size, 3), dtype=np.float64)
    img[..., 0] = r[None, :]
    img[..., 1] = g[None, :]
    img[..., 2] = b[None, :]

    y = np.arange(size, dtype=np.float64)
    img *= (1.0 + 0.05 * np.sin(y / 19.0))[:, None, None]

    rng = np.random.default_rng(seed)
    streaks = np.zeros((size, size))
    for _ in range(60):
        x0 = int(rng.integers(0, size))
        w = int(rng.integers(1, 3))
        streaks[:, x0 : x0 + w] = rng.uniform(0.02, 0.08)
    img += streaks[..., None] * np.array([0.0, 1.0, 0.2])
    img += rng.normal(0, 0.008, img.shape)

    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def _gen_synthetic_leaf(size: int = 256, seed: int = 9) -> Image.Image:
    """Leaf-shaped green object with veins on a plain background."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    ey = (yy - cy) / 105.0
    ex = (xx - cx) / 82.0
    inside = (ex**2 + ey**2) <= 1.0

    img = np.full((size, size, 3), 0.52, dtype=np.float64)
    radial = np.clip(1.0 - 0.25 * np.sqrt(ex**2 + ey**2), 0, 1)
    leaf = np.array([0.30, 0.55, 0.24])[None, None, :] * radial[..., None]
    img[inside] = leaf[inside]

    # Central vein
    img[inside & (np.abs(xx - cx) <= 2)] = [0.55, 0.74, 0.46]
    # Side veins
    for ang in np.deg2rad([-75, -55, -35, -15, 15, 35, 55, 75]):
        t = np.linspace(0, 92, 100)
        vx = np.round(cx + np.cos(ang) * t).astype(int)
        vy = np.round(cy + np.sin(ang) * t * 1.25).astype(int)
        ok = (
            (vy >= 0) & (vy < size) & (vx >= 0) & (vx < size)
        )
        pts_v, pts_x = vy[ok], vx[ok]
        keep = inside[pts_v, pts_x]
        img[pts_v[keep], pts_x[keep]] = [0.50, 0.72, 0.44]

    rng = np.random.default_rng(seed)
    img += rng.normal(0, 0.008, img.shape)
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def _gen_field_composite(
    leaf_img: Image.Image, size: int = 256, seed: int = 13
) -> Image.Image:
    """Real leaf pasted on a textured soil background, rotated/zoomed."""
    leaf = leaf_img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(leaf, dtype=np.float64) / 255.0

    # Leaf mask: PlantVillage backgrounds are neutral gray, leaves are green
    exg = 2 * arr[..., 1] - arr[..., 0] - arr[..., 2]
    mask = exg > 0.06

    # Soil: brown base + smoothed noise + darker clumps
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (size, size))
    noise_img = Image.fromarray(
        ((noise - noise.min()) / np.ptp(noise) * 255).astype(np.uint8)
    ).filter(ImageFilter.GaussianBlur(2.5))
    nz = (np.asarray(noise_img, dtype=np.float64) / 255.0)
    nz = (nz - nz.mean()) * 0.12
    soil = np.zeros((size, size, 3), dtype=np.float64)
    soil[..., 0] = 0.50 + nz
    soil[..., 1] = 0.38 + nz * 0.9
    soil[..., 2] = 0.28 + nz * 0.8
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(8):
        cy = int(rng.integers(0, size))
        cx = int(rng.integers(0, size))
        r = int(rng.integers(6, 18))
        d = (yy - cy) ** 2 + (xx - cx) ** 2
        dark = np.clip(1 - d / (r * r), 0, 1) * 0.10
        soil[..., 0] -= dark
        soil[..., 1] -= dark * 0.9
        soil[..., 2] -= dark * 0.8

    out = soil.copy()
    out[mask] = arr[mask]
    img = Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))

    # Slight zoom (crop 4%, resize back) then slight rotation
    w, h = img.size
    img = img.crop(
        (int(w * 0.04), int(h * 0.04), int(w * 0.96), int(h * 0.96))
    ).resize((w, h), Image.BILINEAR)
    img = img.rotate(8, resample=Image.BILINEAR, fillcolor=(128, 97, 71))

    # Sensor-style noise
    arr2 = np.asarray(img, dtype=np.float64) + rng.normal(
        0, 2.5, (h, w, 3)
    )
    return Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# User-supplied fixtures (optional drop-in)
# ---------------------------------------------------------------------------


def _user_fixtures(category: int) -> list[Path]:
    if not USER_FIXTURE_DIR.exists():
        return []
    out: list[Path] = []
    for d in sorted(USER_FIXTURE_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(f"{category:02d}"):
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS:
                    out.append(p)
    return out


_USER_EXP = {
    1: _EXP_REAL_LEAF, 2: _EXP_REAL_LEAF, 3: _EXP_REAL_LEAF,
    4: {"guard_status": ["caution", "reject"], "require_reasons": [],
        "label": "pass", "note": "user-supplied blurry photo"},
    5: {"guard_status": ["ok", "caution", "reject"], "require_reasons": [],
        "label": "pass", "note": "user-supplied dark photo"},
    6: _EXP_NOISE,
    7: {"guard_status": ["reject", "caution", "ok"], "require_reasons": [],
        "label": "pass", "note": "user-supplied non-leaf photo"},
    8: _EXP_OTHER_PLANT, 9: _EXP_PHONE_STYLE,
    10: {"guard_status": ["ok", "caution", "reject"], "require_reasons": [],
         "label": "pass", "note": "user-supplied ambiguous photo"},
}


# ---------------------------------------------------------------------------
# Fixture list
# ---------------------------------------------------------------------------


def build_fixtures() -> tuple[list[dict], Path]:
    """Return (fixture specs, base leaf path used for derived fixtures)."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    sampled = _sample_per_class()
    base_leaf_path = Path(sampled["Healthy"][0])
    base_leaf = Image.open(base_leaf_path).convert("RGB")
    base_size = base_leaf.size

    fixtures: list[dict] = []

    # -- Categories 1-3: real held-out leaves (user drop-in preferred) ------
    for cat, cls in [(1, "Healthy"), (2, "Early Blight"), (3, "Late Blight")]:
        user = _user_fixtures(cat)
        if user:
            for i, p in enumerate(user, start=1):
                fixtures.append({
                    "category": cat, "name": f"{cat:02d}_{_slug(cls)}_user{i}",
                    "path": p, "provenance": "user-supplied",
                    "ground_truth": None, "expected": _USER_EXP[cat],
                })
        else:
            for i, p in enumerate(sampled[cls], start=1):
                fixtures.append({
                    "category": cat,
                    "name": f"{cat:02d}_{_slug(cls)}_{i}",
                    "path": Path(p), "provenance": "test-split",
                    "ground_truth": cls, "expected": _EXP_REAL_LEAF,
                })

    def _save(img: Image.Image, name: str, fmt: str = "PNG", **kw) -> Path:
        path = FIXTURE_DIR / f"{name}.{fmt.lower()}"
        img.save(path, format=fmt, **kw)
        return path

    def _add(cat, name, path, provenance, expected, ground_truth=None):
        fixtures.append({
            "category": cat, "name": name, "path": path,
            "provenance": provenance, "ground_truth": ground_truth,
            "expected": expected,
        })

    # -- Category 4: blur ---------------------------------------------------
    _add(4, "04a_blur_moderate_s6", _save(_blur(base_leaf, 6), "04a_blur_moderate_s6"),
         "derived", _EXP_BLUR_MODERATE)
    _add(4, "04b_blur_extreme_s20", _save(_blur(base_leaf, 20), "04b_blur_extreme_s20"),
         "derived", _EXP_BLUR_EXTREME)

    # -- Category 5: darkness ----------------------------------------------
    _add(5, "05a_dark_moderate_k030", _save(_darken(base_leaf, 0.30), "05a_dark_moderate_k030"),
         "derived", _EXP_DARK_MODERATE)
    _add(5, "05b_dark_extreme_k010", _save(_darken(base_leaf, 0.10), "05b_dark_extreme_k010"),
         "derived", _EXP_DARK_EXTREME)

    # -- Category 6: noise --------------------------------------------------
    _add(6, "06_random_noise", _save(_gen_noise(), "06_random_noise"),
         "synthetic", _EXP_NOISE)

    # -- Category 7: non-leaf (scene must reject; green texture is a
    #    documented pass) ----------------------------------------------------
    _add(7, "07a_nonleaf_scene", _save(_gen_nonleaf_scene(), "07a_nonleaf_scene"),
         "synthetic", _EXP_NONLEAF_SCENE)
    _add(7, "07b_green_nonleaf_texture",
         _save(_gen_green_texture(), "07b_green_nonleaf_texture"),
         "synthetic", _EXP_GREEN_NONLEAF)

    # -- Category 8: non-tomato leaf ---------------------------------------
    user8 = _user_fixtures(8)
    if user8:
        for i, p in enumerate(user8, start=1):
            _add(8, f"08_other_plant_user{i}", p, "user-supplied",
                 _EXP_OTHER_PLANT)
    else:
        _add(8, "08_other_plant_synthetic",
             _save(_gen_synthetic_leaf(), "08_other_plant_synthetic"),
             "synthetic", _EXP_OTHER_PLANT)

    # -- Category 9: phone-style photo --------------------------------------
    user9 = _user_fixtures(9)
    if user9:
        for i, p in enumerate(user9, start=1):
            _add(9, f"09_phone_photo_user{i}", p, "user-supplied",
                 _EXP_PHONE_STYLE)
    else:
        _add(9, "09_phone_style_composite",
             _save(_gen_field_composite(base_leaf), "09_phone_style_composite",
                   fmt="JPEG", quality=92),
             "derived-composite", _EXP_PHONE_STYLE)

    # -- Category 10: ambiguous/low-quality ---------------------------------
    downscaled = base_leaf.resize((32, 32), Image.LANCZOS).resize(
        base_size, Image.LANCZOS
    )
    mild = _jpeg_roundtrip(downscaled, quality=30)
    _add(10, "10a_ambiguous_mild", _save(mild, "10a_ambiguous_mild", fmt="JPEG", quality=30),
         "derived", _EXP_AMBIG_MILD)
    _add(10, "10b_ambiguous_severe",
         _save(_darken(mild, 0.5), "10b_ambiguous_severe", fmt="JPEG", quality=30),
         "derived", _EXP_AMBIG_SEVERE)

    return fixtures, base_leaf_path


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Pipeline execution (exact app.py gate order)
# ---------------------------------------------------------------------------


def run_fixture(fx: dict, model, thresholds, cfg: dict) -> dict:
    img = Image.open(fx["path"]).convert("RGB")
    metrics = compute_quality_metrics(img)
    quality = evaluate_quality(metrics, thresholds)

    row: dict = {
        "category": fx["category"],
        "fixture": fx["name"],
        "source": str(fx["path"]),
        "provenance": fx["provenance"],
        "ground_truth": fx.get("ground_truth"),
        "guard_status": quality["status"],
        "guard_reasons": quality["reasons"],
        "metrics": {
            "width": metrics.width,
            "height": metrics.height,
            "laplacian_variance": round(metrics.laplacian_variance, 1),
            "green_fraction": round(metrics.green_fraction, 4),
            "adjacent_correlation": round(metrics.adjacent_correlation, 4),
            "saturation": round(metrics.saturation, 4),
        },
        "inference_ran": False,
        "predicted_class": None,
        "correct": None,
        "confidence": None,
        "probabilities": None,
        "msp": None,
        "margin": None,
        "uncertainty_cautions": [],
        "gradcam_saved": False,
        "gradcam_overlay": None,
        "inference_ms": None,
    }

    if quality["status"] != "reject":
        t0 = time.perf_counter()
        result = predict_image(
            model=model,
            image_path=str(fx["path"]),
            device="cpu",
            image_size=cfg["image_size"],
            class_names=cfg["class_names"],
            gradcam=True,
            save_gradcam=True,
            gradcam_output_dir=GRADCAM_DIR,
            gradcam_alpha=cfg.get("gradcam_alpha", 0.4),
        )
        row["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        unc = evaluate_uncertainty(result["probabilities"], thresholds)
        row.update(
            inference_ran=True,
            predicted_class=result["predicted_class"],
            confidence=round(result["confidence"], 4),
            probabilities={
                k: round(v, 4) for k, v in result["probabilities"].items()
            },
            msp=round(unc["msp"], 4),
            margin=round(unc["margin"], 4),
            uncertainty_cautions=unc["cautions"],
            gradcam_saved="gradcam_paths" in result,
        )
        if "gradcam_paths" in result:
            row["gradcam_overlay"] = result["gradcam_paths"].get("overlay")
        if fx.get("ground_truth"):
            row["correct"] = (
                result["predicted_class"] == fx["ground_truth"]
            )

    return row


def classify(row: dict, expected: dict) -> tuple[str, list[str]]:
    issues: list[str] = []
    status = row["guard_status"]
    acceptable = expected["guard_status"]
    if status not in acceptable:
        issues.append(
            f"guard status '{status}', expected one of {acceptable}"
        )
    if status == "reject":
        missing = [
            r for r in expected.get("require_reasons", [])
            if r not in row["guard_reasons"]
        ]
        if missing:
            issues.append(f"missing reject reasons {missing}")
        if row["inference_ran"]:
            issues.append("inference ran on a rejected image")
    else:
        if not row["inference_ran"]:
            issues.append("inference did not run on an accepted image")

    if issues:
        return "DEVIATION", issues
    if expected.get("label") == "known_limitation":
        return "KNOWN-LIMITATION", []
    return "PASS", []


# ---------------------------------------------------------------------------
# Guard-only sweeps
# ---------------------------------------------------------------------------


def sweep(base_img: Image.Image, transform, values, thresholds) -> list[dict]:
    rows = []
    for v in values:
        img = transform(base_img, v)
        m = compute_quality_metrics(img)
        verdict = evaluate_quality(m, thresholds)
        rows.append({
            "param": v,
            "laplacian_variance": round(m.laplacian_variance, 1),
            "green_fraction": round(m.green_fraction, 4),
            "adjacent_correlation": round(m.adjacent_correlation, 4),
            "saturation": round(m.saturation, 4),
            "status": verdict["status"],
            "reasons": verdict["reasons"],
        })
    return rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt(v, pct=False, dash="-"):
    if v is None:
        return dash
    if pct:
        return f"{v:.1%}"
    return f"{v}"


def write_report(doc: dict) -> None:
    lines: list[str] = []
    meta = doc["meta"]
    lines.append("# AgriMind AI — Real-World Validation Report")
    lines.append("")
    lines.append(f"*Generated:* {meta['generated']}  ")
    lines.append(f"*Model:* {meta['display'].get('model_architecture', 'N/A')} "
                 f"(test accuracy {meta['display'].get('test_accuracy', 'N/A')}%, "
                 f"{meta['display'].get('test_samples', 'N/A')} test samples, "
                 f"{meta['display'].get('dataset', 'N/A')})  ")
    lines.append(f"*Checkpoint:* `{meta['checkpoint']}`  ")
    lines.append(f"*Device:* CPU (torch)")
    lines.append("")
    lines.append("## Guard thresholds (from configs/crops/tomato.yaml)")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    for k, v in meta["thresholds"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    # -- Main matrix ---------------------------------------------------------
    lines.append("## Per-category results")
    lines.append("")
    lines.append("| Cat | Fixture | Guard (reasons) | GT | Prediction | "
                 "Confidence | Cautions | Grad-CAM | ms | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in doc["results"]:
        guard = r["guard_status"]
        if r["guard_reasons"]:
            guard += f" ({', '.join(r['guard_reasons'])})"
        gt = r["ground_truth"] or "-"
        pred = _fmt(r["predicted_class"])
        if r["correct"] is True:
            pred += " [correct]"
        elif r["correct"] is False:
            pred += " [WRONG]"
        conf = _fmt(r["confidence"], pct=True)
        cautions = ", ".join(r["uncertainty_cautions"]) or "-"
        cam = "yes" if r["gradcam_saved"] else "no"
        ms = _fmt(r["inference_ms"])
        lines.append(
            f"| {r['category']} | `{r['fixture']}` | {guard} | {gt} | "
            f"{pred} | {conf} | {cautions} | {cam} | {ms} | "
            f"**{r['verdict']}** |"
        )
    lines.append("")

    # -- Issues ---------------------------------------------------------------
    deviations = [r for r in doc["results"] if r["verdict"] == "DEVIATION"]
    if deviations:
        lines.append("### Deviations (empirical findings)")
        lines.append("")
        for r in deviations:
            lines.append(f"- **{r['fixture']}**: {'; '.join(r['issues'])}")
        lines.append("")

    # -- Sweeps ----------------------------------------------------------------
    lines.append("## Guard sweeps (quality gate only, no inference)")
    lines.append("")
    for name, rows in doc["sweeps"].items():
        param = "blur sigma" if name == "blur" else "brightness factor"
        lines.append(f"### {name.capitalize()} sweep")
        lines.append("")
        lines.append(f"| {param} | laplacian var | green frac | corr | "
                     f"saturation | status (reasons) |")
        lines.append("|---|---|---|---|---|---|")
        for s in rows:
            status = s["status"]
            if s["reasons"]:
                status += f" ({', '.join(s['reasons'])})"
            lines.append(
                f"| {s['param']} | {s['laplacian_variance']} | "
                f"{s['green_fraction']} | {s['adjacent_correlation']} | "
                f"{s['saturation']} | {status} |"
            )
        lines.append("")

    # -- Boundary cases ---------------------------------------------------------
    if doc.get("boundary"):
        lines.append("## Boundary cases (most degraded accepted input)")
        lines.append("")
        for r in doc["boundary"]:
            lines.append(
                f"- **{r['fixture']}** — guard `{r['guard_status']}`, "
                f"prediction {r['predicted_class']} "
                f"({_fmt(r['confidence'], pct=True)}), "
                f"cautions {r['uncertainty_cautions'] or 'none'}, "
                f"{_fmt(r['inference_ms'])} ms"
            )
        lines.append("")

    # -- Known limitations -------------------------------------------------------
    kls = [r for r in doc["results"] if r["verdict"] == "KNOWN-LIMITATION"]
    if kls:
        lines.append("## Known limitations confirmed")
        lines.append("")
        for r in kls:
            note = next(
                (f["note"] for f in doc["meta"]["fixture_expectations"]
                 if f["name"] == r["fixture"]), ""
            )
            lines.append(
                f"- **{r['fixture']}**: predicted "
                f"{r['predicted_class']} at "
                f"{_fmt(r['confidence'], pct=True)} — {note}"
            )
        lines.append("")

    # -- Summary ------------------------------------------------------------------
    s = doc["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Fixtures: {s['n_fixtures']} "
                 f"(PASS {s['n_pass']}, DEVIATION {s['n_deviation']}, "
                 f"KNOWN-LIMITATION {s['n_known_limitation']})")
    if s["n_correct"] is not None:
        lines.append(
            f"- Held-out leaf correctness: {s['n_correct']}/{s['n_total_gt']} "
            f"on sampled test-split images"
        )
    if s["mean_inference_ms"] is not None:
        lines.append(f"- Mean CPU inference (incl. Grad-CAM): "
                     f"{s['mean_inference_ms']:.0f} ms over "
                     f"{s['n_inference']} runs")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Fixtures are stored in `reports/validation/fixtures/`, "
                 "Grad-CAM overlays in `reports/validation/gradcam/`, and "
                 "raw numbers in `reports/validation/validation_results.json`. "
                 "All synthetic/derived fixtures are deterministic (seeded).*")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    cfg = get_crop_config(CROP_NAME, crops_dir=str(CROPS_DIR))
    thresholds = load_guard_thresholds(cfg)

    checkpoint = _PROJECT_ROOT / cfg["model_path"]
    print(f"Loading model: {checkpoint}")
    model = load_checkpoint(str(checkpoint), device="cpu")
    model.eval()

    fixtures, base_leaf_path = build_fixtures()
    print(f"Fixtures built: {len(fixtures)} "
          f"(base leaf for derived: {base_leaf_path.name})")

    results: list[dict] = []
    for fx in fixtures:
        row = run_fixture(fx, model, thresholds, cfg)
        verdict, issues = classify(row, fx["expected"])
        row["verdict"] = verdict
        row["issues"] = issues
        row["expectation"] = {
            "guard_status": fx["expected"]["guard_status"],
            "label": fx["expected"].get("label", "pass"),
            "note": fx["expected"].get("note", ""),
        }
        results.append(row)
        conf = f" conf={row['confidence']:.1%}" if row["confidence"] else ""
        pred = f" pred={row['predicted_class']}" if row["predicted_class"] else ""
        print(f"  [{verdict:>17}] cat{fx['category']:>2} {fx['name']:<32} "
              f"guard={row['guard_status']}{pred}{conf}")

    # -- Guard-only sweeps ----------------------------------------------------
    base_leaf = Image.open(base_leaf_path).convert("RGB")
    blur_rows = sweep(base_leaf, _blur, [2, 4, 6, 8, 12, 16, 20], thresholds)
    dark_rows = sweep(base_leaf, _darken, [0.5, 0.3, 0.2, 0.1, 0.05], thresholds)
    print("Sweeps completed: blur sigma 2..20, brightness 0.5..0.05")

    # -- Boundary cases: most degraded accepted input ---------------------------
    boundary = []
    for name, rows, values, transform in [
        ("04s_blur_boundary", blur_rows, [2, 4, 6, 8, 12, 16, 20], _blur),
        ("05s_dark_boundary", dark_rows, [0.5, 0.3, 0.2, 0.1, 0.05], _darken),
    ]:
        accepted = [s for s in rows if s["status"] != "reject"]
        if not accepted:
            continue
        if name.startswith("04s"):
            param = max(s["param"] for s in accepted)  # most blur accepted
        else:
            param = min(s["param"] for s in accepted)  # most dark accepted
        img = transform(base_leaf, param)
        path = FIXTURE_DIR / f"{name}.png"
        img.save(path, format="PNG")
        row = run_fixture(
            {"category": int(name[:2]), "name": name, "path": path,
             "provenance": "derived", "ground_truth": None,
             "expected": {}},
            model, thresholds, cfg,
        )
        row["verdict"] = "INFO"
        row["issues"] = []
        row["boundary_param"] = param
        boundary.append(row)
        print(f"  [            INFO] {name} (param={param}) "
              f"guard={row['guard_status']} pred={row['predicted_class']}")

    # -- Summary -----------------------------------------------------------------
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_dev = sum(1 for r in results if r["verdict"] == "DEVIATION")
    n_kl = sum(1 for r in results if r["verdict"] == "KNOWN-LIMITATION")
    gt_rows = [r for r in results if r["correct"] is not None]
    inf_ms = [r["inference_ms"] for r in results + boundary
              if r["inference_ms"] is not None]
    summary = {
        "n_fixtures": len(results),
        "n_pass": n_pass,
        "n_deviation": n_dev,
        "n_known_limitation": n_kl,
        "n_correct": sum(1 for r in gt_rows if r["correct"]) if gt_rows else None,
        "n_total_gt": len(gt_rows) if gt_rows else None,
        "n_inference": len(inf_ms),
        "mean_inference_ms": (sum(inf_ms) / len(inf_ms)) if inf_ms else None,
    }

    doc = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "crop": CROP_NAME,
            "checkpoint": str(checkpoint),
            "display": cfg.get("display", {}),
            "thresholds": asdict(thresholds),
            "device": "cpu",
            "fixture_expectations": [
                {"name": f["name"],
                 "guard_status": f["expected"]["guard_status"],
                 "label": f["expected"].get("label", "pass"),
                 "note": f["expected"].get("note", "")}
                for f in fixtures
            ],
        },
        "results": results,
        "sweeps": {"blur": blur_rows, "dark": dark_rows},
        "boundary": boundary,
        "summary": summary,
    }

    RESULTS_JSON.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(doc)

    print()
    print(f"PASS {n_pass} | DEVIATION {n_dev} | KNOWN-LIMITATION {n_kl} "
          f"| INFO {len(boundary)}")
    if summary["n_correct"] is not None:
        print(f"Held-out correctness: {summary['n_correct']}/"
              f"{summary['n_total_gt']}")
    print(f"Reports: {RESULTS_JSON}")
    print(f"         {REPORT_MD}")


if __name__ == "__main__":
    main()
