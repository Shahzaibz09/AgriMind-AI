"""
AgriMind AI — FastAPI Backend Application.

Serves HTTP REST endpoints for the Flutter mobile application:
- GET  /health              — Service health check
- GET  /api/v1/crops        — Available crops and metadata
- POST /api/v1/analyze      — Guarded leaf analysis (Quality -> Crop Verification -> Disease Model -> Grad-CAM)
- POST /api/v1/assistant    — Deterministic trilingual Farmer Assistant
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap — ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.assistant import (
    FarmerQuestion,
    answer_farmer_question,
    normalize_language,
)
from app.crop_config import get_crop_config, list_crops
from app.crop_verification import (
    STATUS_MISMATCH,
    STATUS_OK as CROP_STATUS_OK,
    STATUS_SKIPPED,
    STATUS_UNCERTAIN,
    verify_crop,
)
from app.guard import (
    QUALITY_OK,
    REASON_MESSAGES,
    compute_quality_metrics,
    evaluate_quality,
    evaluate_uncertainty,
    load_guard_thresholds,
)
from src.dataset import get_eval_transforms
from src.gradcam import denormalize, generate_overlay
from src.inference import predict_image
from src.model import load_checkpoint

logger = logging.getLogger("agrimind.backend")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# FastAPI App & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AgriMind AI API",
    description="Backend API serving crop disease classification, Grad-CAM explainability, and trilingual farmer advisory.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model Cache
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, Any] = {}


def _get_model(crop_name: str) -> Any:
    """Retrieve or load cached PyTorch model for crop."""
    crop_key = crop_name.strip().lower()
    if crop_key in _MODEL_CACHE:
        return _MODEL_CACHE[crop_key]

    cfg = get_crop_config(crop_name)
    model_path = PROJECT_ROOT / cfg["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    logger.info("Loading PyTorch model for %s from %s", crop_name, model_path)
    model = load_checkpoint(str(model_path), device="cpu")
    model.eval()
    _MODEL_CACHE[crop_key] = model
    return model


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    app: str
    available_crops: list[str]


class CropInfo(BaseModel):
    id: str
    name: str
    emoji: str
    tagline: str
    classes: list[str]
    descriptions: dict[str, str] = Field(default_factory=dict)


class CropsResponse(BaseModel):
    crops: list[CropInfo]


class AssistantRequest(BaseModel):
    crop: str
    predicted_disease: Optional[str] = None
    confidence: Optional[float] = None
    question: str = ""
    language: str = "en"


class AssistantResponse(BaseModel):
    status: str
    language: str
    response: str
    disease: Optional[str] = None
    confidence: Optional[float] = None
    sections: dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    status: str
    crop: str
    crop_label: str
    predicted_disease: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    gradcam_overlay_base64: Optional[str] = None
    status_label: Optional[str] = None
    description: Optional[str] = None
    recommended_next_step: Optional[str] = None
    cautions: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    message: Optional[str] = None
    detected_crop: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check service health and list supported crops."""
    crops = list_crops()
    return HealthResponse(
        status="ok",
        app="AgriMind AI API",
        available_crops=crops,
    )


@app.get("/api/v1/crops", response_model=CropsResponse)
def get_crops() -> CropsResponse:
    """Return list of supported crops and their metadata."""
    crop_names = list_crops()
    emoji_map = {"tomato": "🍅", "potato": "🥔", "apple": "🍎"}
    tagline_map = {
        "tomato": "Leaf disease screening for tomato plants",
        "potato": "Leaf disease screening for potato plants",
        "apple": "Leaf disease screening for apple trees",
    }

    result: list[CropInfo] = []
    for name in crop_names:
        cfg = get_crop_config(name)
        cid = name.lower()
        result.append(
            CropInfo(
                id=cid,
                name=name,
                emoji=emoji_map.get(cid, "🌿"),
                tagline=tagline_map.get(cid, f"Leaf disease screening for {name}"),
                classes=cfg.get("class_names", []),
                descriptions=cfg.get("class_descriptions", {}),
            )
        )
    return CropsResponse(crops=result)


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_leaf(
    image: UploadFile = File(...),
    crop: str = Form(...),
) -> AnalysisResponse:
    """Run the guarded leaf analysis pipeline on an uploaded image."""
    # 1. Resolve crop name
    available_crops = list_crops()
    crop_match = next((c for c in available_crops if c.lower() == crop.strip().lower()), None)
    if not crop_match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported crop '{crop}'. Available crops: {available_crops}",
        )
    selected_crop = crop_match

    # 2. Read image
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        logger.warning("Unreadable image upload: %s", exc)
        return AnalysisResponse(
            status="unreadable_image",
            crop=selected_crop.lower(),
            crop_label=selected_crop,
            message="The uploaded file could not be read as an image. Please upload a valid JPG, PNG, or BMP.",
        )

    crop_cfg = get_crop_config(selected_crop)
    thresholds = load_guard_thresholds(crop_cfg)

    # 3. Robustness Guard — Quality Check
    metrics = compute_quality_metrics(pil_image)
    quality = evaluate_quality(metrics, thresholds)
    if quality["status"] == "reject":
        reason_msgs = [REASON_MESSAGES.get(r, r) for r in quality["reasons"]]
        return AnalysisResponse(
            status="quality_rejected",
            crop=selected_crop.lower(),
            crop_label=selected_crop,
            reasons=quality["reasons"],
            message="\n".join(reason_msgs),
            recommended_next_step="Please upload a clear, well-lit, close-up color photo of a single leaf.",
        )

    # 4. Crop Verification
    crop_check = verify_crop(pil_image, selected_crop)
    if crop_check.status == STATUS_MISMATCH:
        return AnalysisResponse(
            status="crop_mismatch",
            crop=selected_crop.lower(),
            crop_label=selected_crop,
            detected_crop=crop_check.detected_crop,
            confidence=crop_check.confidence,
            message=crop_check.message,
            recommended_next_step=(
                f"Please switch to {crop_check.detected_crop} or upload a {selected_crop} leaf."
                if crop_check.detected_crop
                else "Please upload a photo matching the selected crop."
            ),
        )
    if crop_check.status == STATUS_UNCERTAIN:
        return AnalysisResponse(
            status="crop_uncertain",
            crop=selected_crop.lower(),
            crop_label=selected_crop,
            confidence=crop_check.confidence,
            message=crop_check.message,
            recommended_next_step="Please upload a clearer close-up of the leaf to verify the crop.",
        )

    # 5. Disease Model Inference + Grad-CAM
    model = _get_model(selected_crop)
    tmp_path: str | None = None
    try:
        suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = predict_image(
            model=model,
            image_path=tmp_path,
            device="cpu",
            image_size=crop_cfg["image_size"],
            class_names=crop_cfg["class_names"],
            gradcam=True,
            save_gradcam=False,
            gradcam_alpha=crop_cfg.get("gradcam_alpha", 0.4),
        )

        # Build Grad-CAM overlay image
        heatmap = result["gradcam"]
        transform = get_eval_transforms(crop_cfg["image_size"])
        img_tensor = transform(pil_image).unsqueeze(0)
        original_np = denormalize(img_tensor.squeeze(0))
        overlay_np = generate_overlay(
            original_np,
            heatmap,
            alpha=crop_cfg.get("gradcam_alpha", 0.4),
        )

        overlay_img = Image.fromarray(overlay_np)
        buf = io.BytesIO()
        overlay_img.save(buf, format="JPEG", quality=85)
        overlay_base64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # 6. Uncertainty evaluation
    uncertainty = evaluate_uncertainty(result["probabilities"], thresholds)
    cautions = [REASON_MESSAGES.get(c, c) for c in uncertainty["cautions"]]
    if quality["status"] != QUALITY_OK:
        cautions.extend([REASON_MESSAGES.get(r, r) for r in quality["reasons"]])

    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    descriptions = crop_cfg.get("class_descriptions", {})
    desc = descriptions.get(pred_class, "")

    is_healthy = pred_class.lower() == "healthy"
    status_label = "Healthy" if is_healthy else "Action Recommended"
    next_step = (
        "No treatment required. Continue standard monitoring and cultivation practices."
        if is_healthy
        else "Isolate affected plants and consult the Farmer Assistant for organic and cultural management practices."
    )

    return AnalysisResponse(
        status="success",
        crop=selected_crop.lower(),
        crop_label=selected_crop,
        predicted_disease=pred_class,
        confidence=confidence,
        probabilities=result["probabilities"],
        gradcam_overlay_base64=overlay_base64,
        status_label=status_label,
        description=desc,
        recommended_next_step=next_step,
        cautions=cautions,
    )


@app.post("/api/v1/assistant", response_model=AssistantResponse)
@app.post("/api/v1/ask", response_model=AssistantResponse)
def ask_assistant(req: AssistantRequest) -> AssistantResponse:
    """Query the deterministic trilingual Farmer Assistant."""
    fq = FarmerQuestion(
        crop=req.crop,
        predicted_disease=req.predicted_disease,
        confidence=req.confidence,
        question=req.question,
        language=req.language,
    )
    answer = answer_farmer_question(fq)
    return AssistantResponse(
        status=answer.status,
        language=answer.language,
        response=answer.response,
        disease=answer.disease,
        confidence=answer.confidence,
        sections=answer.sections,
    )
