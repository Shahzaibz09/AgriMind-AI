"""
AgriMind AI — Streamlit Web Application.

Farmer-first web interface for crop disease classification.  The
primary experience is the AI leaf analysis: upload a photo, receive a
diagnosis with an explainability visualization (Grad-CAM), see honest
uncertainty messaging, and get multilingual guidance from the Farmer
Assistant.  Technical model information lives in expandable
"Technical Details" sections, out of the farmer's way.

The production model loads once via ``st.cache_resource`` and is
shared across all re-runs.

Launch::

    streamlit run app/app.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap — ensure project root is on sys.path so that ``src.*``
# and ``app.*`` imports work regardless of how Streamlit was launched.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.assistant import FarmerQuestion, answer_farmer_question  # noqa: E402
from app.crop_config import get_crop_config, list_crops  # noqa: E402
from app.crop_verification import verify_crop  # noqa: E402
from app.guard import (  # noqa: E402
    QUALITY_OK,
    REASON_MESSAGES,
    compute_quality_metrics,
    evaluate_quality,
    evaluate_uncertainty,
    load_guard_thresholds,
)
from src.dataset import get_eval_transforms  # noqa: E402
from src.gradcam import denormalize, generate_overlay  # noqa: E402
from src.inference import predict_image  # noqa: E402
from src.model import load_checkpoint  # noqa: E402

_CROPS_DIR = _PROJECT_ROOT / "configs" / "crops"

# ---------------------------------------------------------------------------
# Page config & branding
#
# Local development serves on localhost:8501 (``streamlit run app/app.py``).
# The public deployment URL (for example, https://agrimindai.streamlit.app
# on Streamlit Community Cloud) is configured separately at deploy time —
# no code change is needed here.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AgriMind AI — Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — farmer-first design system (green branding, step badges,
# severity badges, highlighted probabilities).  Presentation only.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Hero banner */
    .agri-hero {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 55%, #43A047 100%);
        color: #FFFFFF;
        padding: 1.5rem 2rem 1.4rem 2rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    .agri-hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.25rem 0;
        letter-spacing: 0.5px;
    }
    .agri-hero h2 {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0 0 0.3rem 0;
        color: #E8F5E9;
    }
    .agri-hero p {
        font-size: 1.05rem;
        margin: 0;
        color: #DCEDC8;
    }
    .agri-hero .agri-hero-crop {
        font-size: 0.95rem;
        margin-top: 0.45rem;
        color: #A5D6A7;
    }
    /* Numbered step headers */
    .agri-step-header {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 1.4rem 0 0.8rem 0;
    }
    .agri-step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.9rem;
        height: 1.9rem;
        border-radius: 50%;
        background-color: #2E7D32;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1rem;
        flex: 0 0 auto;
    }
    .agri-step-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B5E20;
    }
    .agri-step-subtitle {
        font-size: 0.95rem;
        font-weight: 500;
        color: #607D8B;
    }
    /* Diagnosis card */
    /* Diagnosis result card — key/value rows */
    .agri-result-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem 1.2rem;
        align-items: baseline;
        padding: 0.42rem 0;
        border-bottom: 1px solid #E8F5E9;
    }
    .agri-result-row:last-child { border-bottom: none; }
    .agri-result-key {
        flex: 0 0 11rem;
        font-weight: 700;
        font-size: 0.98rem;
        color: #37474F;
    }
    .agri-result-val {
        flex: 1 1 12rem;
        font-size: 1.02rem;
        color: #1B5E20;
    }
    .agri-disease-name {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.25;
        color: #1B5E20;
    }
    /* Severity badges (presentation-only mapping) */
    .agri-badge {
        display: inline-block;
        padding: 0.22rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.4px;
    }
    .agri-badge-healthy { background-color: #C8E6C9; color: #1B5E20; }
    .agri-badge-disease { background-color: #FFE0B2; color: #E65100; }
    .agri-badge-severe  { background-color: #FFCDD2; color: #B71C1C; }
    .agri-badge-neutral { background-color: #ECEFF1; color: #37474F; }
    /* Highlighted predicted-class probability row */
    .agri-prob-predicted {
        font-weight: 700;
        color: #2E7D32;
        font-size: 1.02rem;
        margin: 0.45rem 0 0.1rem 0;
    }
    /* Progress bars stay green */
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    /* Safety workflow strip (under the hero) */
    .agri-workflow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.3rem 0.15rem;
        margin: 0.9rem 0 0.2rem 0;
    }
    .agri-workflow-step {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background-color: #F1F8E9;
        color: #1B5E20;
        border: 1px solid #C8E6C9;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        white-space: nowrap;
    }
    .agri-workflow-arrow { color: #81C784; font-size: 0.9rem; }
    /* Confidence meter (result card) */
    .agri-conf-meter {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-top: 0.7rem;
    }
    .agri-conf-track {
        flex: 1 1 auto;
        height: 0.9rem;
        border-radius: 999px;
        background-color: #ECEFF1;
        overflow: hidden;
    }
    .agri-conf-fill { height: 100%; border-radius: 999px; }
    .agri-conf-high { background-color: #2E7D32; }
    .agri-conf-medium { background-color: #F57C00; }
    .agri-conf-low { background-color: #C62828; }
    .agri-conf-value {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1B5E20;
        flex: 0 0 auto;
    }
    .agri-conf-note {
        font-size: 0.8rem;
        color: #607D8B;
        margin: 0.3rem 0 0 0;
    }
    /* Grad-CAM color legend */
    .agri-legend {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 0.55rem;
    }
    .agri-legend-bar {
        width: 170px;
        height: 0.7rem;
        border-radius: 999px;
        background: linear-gradient(
            90deg, #0000FF 0%, #00FFFF 25%, #00FF00 50%, #FFFF00 75%, #FF0000 100%
        );
    }
    .agri-legend-label { font-size: 0.8rem; color: #607D8B; }
    /* Demo sample chip (clearly labeled demo example) */
    .agri-demo-chip {
        display: inline-block;
        background-color: #E3F2FD;
        color: #1565C0;
        border: 1px solid #BBDEFB;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 0.18rem 0.7rem;
        margin-bottom: 0.35rem;
    }
    /* Button polish (presentation only) */
    .stButton > button { border-radius: 10px; font-weight: 600; }
    /* Mobile-friendly sizing */
    @media (max-width: 640px) {
        .agri-hero { padding: 1.15rem 1.25rem 1.05rem 1.25rem; }
        .agri-hero h1 { font-size: 1.7rem; }
        .agri-hero h2 { font-size: 1.05rem; }
        .agri-step-title { font-size: 1.12rem; }
        .agri-step-subtitle { font-size: 0.85rem; }
        .agri-result-key { flex-basis: 100%; }
        .agri-conf-track { height: 0.8rem; }
        .agri-legend { flex-direction: column; align-items: flex-start; gap: 0.3rem; }
        .agri-workflow-arrow { display: none; }
    }
    /* Presentation polish: hide developer chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Presentation helpers (display only — no effect on pipeline logic)
# ---------------------------------------------------------------------------


def _step_header(number: int, title: str, subtitle: str | None = None) -> None:
    """Render a numbered step header used to guide the farmer's flow."""
    _subtitle_html = (
        f'<span class="agri-step-subtitle">{subtitle}</span>' if subtitle else ""
    )
    st.markdown(
        f'<div class="agri-step-header">'
        f'<span class="agri-step-badge">{number}</span>'
        f'<span class="agri-step-title">{title}</span>'
        f'{_subtitle_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# Display-only severity mapping; unknown classes degrade to a neutral badge.
_SEVERITY_BADGES = {
    "healthy": ("agri-badge-healthy", "Healthy"),
    "early blight": ("agri-badge-disease", "Disease detected"),
    "late blight": ("agri-badge-severe", "Serious disease"),
    "apple scab": ("agri-badge-disease", "Disease detected"),
    "black rot": ("agri-badge-severe", "Serious disease"),
}


def _severity_for(predicted_class: str) -> tuple[str, str]:
    """Return (badge css class, status label) for the predicted class."""
    return _SEVERITY_BADGES.get(
        predicted_class.strip().lower(), ("agri-badge-neutral", predicted_class)
    )


# Crop emojis for the selector and result card (display only; the
# underlying crop values — and therefore the pipeline — are unchanged).
_CROP_EMOJI = {"Tomato": "🍅", "Potato": "🥔", "Apple": "🍎"}


def _crop_label(crop: str) -> str:
    """Emoji-prefixed crop label for display (e.g. '🍅 Tomato')."""
    return f"{_CROP_EMOJI.get(crop, '🌱')} {crop}"


# Farmer-friendly recommended next step per prediction (display only —
# general, non-prescriptive guidance consistent with the knowledge base).
_NEXT_STEPS = {
    "healthy": (
        "No disease detected. Keep monitoring your crop weekly and "
        "continue good field hygiene."
    ),
    "early blight": (
        "Remove badly affected leaves and review the care steps in "
        "Ask AgriMind below."
    ),
    "late blight": (
        "Act quickly — remove affected leaves, review the guidance "
        "below, and consult your local agricultural office."
    ),
    "apple scab": (
        "Remove fallen leaves around the tree and review the care "
        "steps in Ask AgriMind below."
    ),
    "black rot": (
        "Prune affected parts, review the guidance below, and consult "
        "your local agricultural office."
    ),
}
_NEXT_STEP_DEFAULT = (
    "Review the guidance in Ask AgriMind below and monitor your plant "
    "closely."
)


def _next_step_for(predicted_class: str) -> str:
    """Recommended next step shown on the result card (display only)."""
    return _NEXT_STEPS.get(predicted_class.strip().lower(), _NEXT_STEP_DEFAULT)


# Safety workflow shown under the hero (communicates the guarded pipeline:
# crop selection -> upload -> quality guard -> crop verification ->
# prediction -> confidence -> Grad-CAM -> farmer assistant).
_WORKFLOW_STEPS = (
    ("🌱", "Select crop"),
    ("📷", "Upload leaf"),
    ("✅", "Quality check"),
    ("🔎", "Crop check"),
    ("🧠", "Diagnosis"),
    ("🎯", "Grad-CAM"),
    ("🤝", "Guidance"),
)


def _workflow_strip() -> None:
    """Render the compact pipeline/safety workflow strip (display only)."""
    parts: list[str] = ['<div class="agri-workflow">']
    for _i, (_icon, _label) in enumerate(_WORKFLOW_STEPS):
        if _i:
            parts.append('<span class="agri-workflow-arrow">›</span>')
        parts.append(
            f'<span class="agri-workflow-step">{_icon} {_label}</span>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# Demo sample images (clearly labeled demo examples for live demos; single
# unmodified copies of existing repository images — see demo_images/README).
_DEMO_DIR = _PROJECT_ROOT / "app" / "demo_images"
_DEMO_SAMPLES = {
    "Tomato": ("tomato_demo.JPG", "Tomato · Early Blight example"),
    "Potato": ("potato_demo.JPG", "Potato · Late Blight example"),
    "Apple": ("apple_demo.JPG", "Apple · Apple Scab example"),
}


# Assistant response languages (labels shown in the UI).
_LANGUAGE_LABELS = {"en": "English", "ur": "اردو (Urdu)", "roman_ur": "Roman Urdu"}


# Quick-pick example questions (existing supported intents only —
# symptoms / management / prevention; no new advisory content).
_QUICK_QUESTIONS = (
    "What are the symptoms?",
    "How can I manage it?",
    "How can I prevent it?",
)


def _confidence_meter(confidence: float) -> None:
    """Render a lightweight confidence meter (display only).

    The underlying confidence value is unchanged; the color band mirrors
    the existing uncertainty-gate threshold (warn below 0.75).
    """
    if confidence >= 0.90:
        band = "agri-conf-high"
    elif confidence >= 0.75:
        band = "agri-conf-medium"
    else:
        band = "agri-conf-low"
    st.markdown(
        f'<div class="agri-conf-meter">'
        f'<div class="agri-conf-track">'
        f'<div class="agri-conf-fill {band}" '
        f'style="width:{confidence * 100:.1f}%"></div>'
        f"</div>"
        f'<span class="agri-conf-value">{confidence:.1%}</span>'
        f"</div>"
        f'<p class="agri-conf-note">Model confidence — how sure the AI is, '
        f"not a guarantee of accuracy.</p>",
        unsafe_allow_html=True,
    )


def _gradcam_legend() -> None:
    """Render the Grad-CAM color legend (display only)."""
    st.markdown(
        '<div class="agri-legend">'
        '<span class="agri-legend-label">Less influence</span>'
        '<div class="agri-legend-bar"></div>'
        '<span class="agri-legend-label">More influence</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# Known assistant section headers (imported read-only from the existing
# assistant module) so they can be emphasized when rendered as Markdown.
try:  # pragma: no cover — defensive: assistant internals are stable
    from app.assistant import _SECTION_HEADERS as _ASSISTANT_SECTION_HEADERS
except ImportError:  # noqa: BLE001
    _ASSISTANT_SECTION_HEADERS = {}

_ASSISTANT_HEADERS = {
    _header
    for _per_language in _ASSISTANT_SECTION_HEADERS.values()
    for _header in _per_language.values()
}


def _format_assistant_response(response: str) -> str:
    """Bold known section headers for readability (display only)."""
    return "\n".join(
        f"**{_line}**" if _line.strip() in _ASSISTANT_HEADERS else _line
        for _line in response.split("\n")
    )


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_model(checkpoint_path: str):
    """Load the production model once; cached across all re-runs."""
    model = load_checkpoint(checkpoint_path, device="cpu")
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def _load_crop_config(crop_name: str) -> dict:
    """Load crop configuration; cached by crop name."""
    return get_crop_config(crop_name, crops_dir=str(_CROPS_DIR))


# ---------------------------------------------------------------------------
# Sidebar — crop selection, plain-language guidance, technical details
# ---------------------------------------------------------------------------

available_crops = list_crops(str(_CROPS_DIR))
if not available_crops:
    st.error("No crop configurations found in configs/crops/.")
    st.stop()

# ---------------------------------------------------------------------------
# Shared UI state — crop & language selection, pending reset
#
# The crop (and the assistant language) can be changed from the sidebar or
# from the main page; ``_selected_crop`` / ``_selected_language`` are the
# single source of truth, kept in sync via widget callbacks.  Tomato is the
# default crop (the flagship demo crop).
# ---------------------------------------------------------------------------

_DEFAULT_CROP = "Tomato" if "Tomato" in available_crops else available_crops[0]
if "_selected_crop" not in st.session_state:
    st.session_state["_selected_crop"] = _DEFAULT_CROP
if st.session_state["_selected_crop"] not in available_crops:
    st.session_state["_selected_crop"] = _DEFAULT_CROP
if "_selected_language" not in st.session_state:
    st.session_state["_selected_language"] = "en"
if st.session_state["_selected_language"] not in _LANGUAGE_LABELS:
    st.session_state["_selected_language"] = "en"


def _on_sidebar_crop_change() -> None:
    st.session_state["_selected_crop"] = st.session_state["sidebar_crop"]


def _on_main_crop_change() -> None:
    st.session_state["_selected_crop"] = st.session_state["main_crop"]


def _on_sidebar_language_change() -> None:
    st.session_state["_selected_language"] = st.session_state["sidebar_language"]


def _on_main_language_change() -> None:
    st.session_state["_selected_language"] = st.session_state["main_language"]


# "Analyze Another Leaf" reset — applied at the top of the next run so the
# uploader widget state can be cleared before the widget is instantiated.
# Only analysis/session state is reset; models and configuration are not
# touched.
if st.session_state.pop("_pending_reset", False):
    for _key in (
        "result",
        "overlay",
        "uncertainty",
        "assistant_answer",
        "assistant_question",
        "_analysis_error",
        "_file_key",
        "_upload_id",
        "_demo_sample",
    ):
        st.session_state.pop(_key, None)
    # A file_uploader value cannot be assigned via session_state; popping
    # the key is the supported way to clear the widget.
    st.session_state.pop("leaf_uploader", None)

with st.sidebar:
    st.markdown("### 🌱 AgriMind AI")
    st.caption("AI-powered crop health intelligence")

    st.markdown("#### Crop & Language")
    st.session_state["sidebar_crop"] = st.session_state["_selected_crop"]
    st.selectbox(
        "Select crop",
        available_crops,
        format_func=_crop_label,
        key="sidebar_crop",
        on_change=_on_sidebar_crop_change,
    )
    st.session_state["sidebar_language"] = st.session_state["_selected_language"]
    st.selectbox(
        "Assistant language",
        list(_LANGUAGE_LABELS),
        format_func=lambda code: _LANGUAGE_LABELS.get(code, code),
        key="sidebar_language",
        on_change=_on_sidebar_language_change,
    )

    selected_crop = st.session_state["_selected_crop"]
    crop_cfg = _load_crop_config(selected_crop)

    st.divider()
    st.markdown("#### Supported Conditions")
    _class_descs = crop_cfg.get("class_descriptions", {})
    for _cls in crop_cfg.get("class_names", []):
        _desc = _class_descs.get(_cls)
        if _desc:
            st.markdown(f"- **{_cls}** — {_desc}")
        else:
            st.markdown(f"- {_cls}")

    st.divider()
    st.markdown("#### How It Works")
    st.markdown(
        "1. Select your crop and upload a single leaf photo\n"
        "2. Tap **Analyze Leaf** — AgriMind checks photo quality and crop\n"
        "3. Read the diagnosis and ask for guidance in your language"
    )

    st.divider()
    with st.expander("Technical Details"):
        display = crop_cfg.get("display", {})
        st.markdown(
            f"**Architecture:** {display.get('model_architecture', 'N/A')}"
        )
        test_acc = display.get("test_accuracy")
        if test_acc is not None:
            st.markdown(f"**Test Accuracy:** {test_acc}%")
        test_samples = display.get("test_samples")
        if test_samples is not None:
            st.markdown(f"**Test Samples:** {test_samples}")
        dataset = display.get("dataset")
        if dataset:
            st.markdown(f"**Dataset:** {dataset}")
        st.caption(
            "Automatic image quality checks are active for every upload."
        )

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="agri-hero">
        <h1>AgriMind AI</h1>
        <h2>AI-powered Crop Health Intelligence</h2>
        <p>Smart agriculture with multilingual farmer support —
        English · اردو · Roman Urdu</p>
        <p class="agri-hero-crop">Now checking: {_crop_label(selected_crop)} leaf health in seconds</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Safety workflow strip — communicates the guarded pipeline end to end.
_workflow_strip()

# ---------------------------------------------------------------------------
# Step 1 — Crop selection (main page; stays in sync with the sidebar)
# ---------------------------------------------------------------------------

_step_header(1, "Select your crop", "Choose the crop you want to check")
st.session_state["main_crop"] = st.session_state["_selected_crop"]
st.radio(
    "Select crop",
    available_crops,
    format_func=_crop_label,
    horizontal=True,
    key="main_crop",
    on_change=_on_main_crop_change,
    label_visibility="collapsed",
)
selected_crop = st.session_state["_selected_crop"]
crop_cfg = _load_crop_config(selected_crop)

# ---------------------------------------------------------------------------
# Step 2 — Upload
# ---------------------------------------------------------------------------

_step_header(2, "Upload a leaf photo", "A clear, close-up photo gives the best results")

uploaded_file = st.file_uploader(
    "Upload a clear leaf image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, JPEG, PNG, BMP",
    key="leaf_uploader",
)
st.caption("Use good lighting and keep one leaf visible.")

# -- Demo sample (clearly labeled demo example for the live demo) ----------
_demo_entry = _DEMO_SAMPLES.get(selected_crop)
if _demo_entry is not None and (_DEMO_DIR / _demo_entry[0]).exists():
    if st.button(
        f"🌱 Try a demo image — {_demo_entry[1]}",
        help="Loads a labeled sample image from this project (demo example).",
    ):
        st.session_state["_demo_sample"] = {
            "crop": selected_crop,
            "name": _demo_entry[0],
            "label": _demo_entry[1],
            "bytes": (_DEMO_DIR / _demo_entry[0]).read_bytes(),
        }

# ---------------------------------------------------------------------------
# Active image — a new upload supersedes the demo sample; the demo sample
# supersedes an unchanged upload.  Whenever the active image (or the crop)
# changes, stale results are discarded so a previous analysis is never
# shown alongside a new selection.
# ---------------------------------------------------------------------------

_demo_sample = st.session_state.get("_demo_sample")
if _demo_sample is not None and _demo_sample.get("crop") != selected_crop:
    # Demo sample belongs to a different crop — drop it (the crop
    # verification gate would reject it anyway).
    st.session_state.pop("_demo_sample", None)
    _demo_sample = None

_upload_id = (
    (uploaded_file.name, uploaded_file.size) if uploaded_file is not None else None
)
if _upload_id != st.session_state.get("_upload_id"):
    st.session_state["_upload_id"] = _upload_id
    if _upload_id is not None:
        # A new upload always supersedes the demo sample.
        st.session_state.pop("_demo_sample", None)
        _demo_sample = None

if _demo_sample is not None:
    _img_name = _demo_sample["name"]
    _img_bytes = _demo_sample["bytes"]
    _img_label = _demo_sample["label"]
    _file_key = f"{selected_crop}|demo:{_img_name}"
elif uploaded_file is not None:
    _img_name = uploaded_file.name
    _img_bytes = uploaded_file.getvalue()
    _img_label = None
    _file_key = f"{selected_crop}|{uploaded_file.name}|{uploaded_file.size}"
else:
    _img_name = None
    _img_bytes = None
    _img_label = None
    _file_key = None

if _file_key is not None and _file_key != st.session_state.get("_file_key"):
    st.session_state["_file_key"] = _file_key
    for _key in (
        "result",
        "overlay",
        "assistant_answer",
        "uncertainty",
        "_analysis_error",
    ):
        st.session_state.pop(_key, None)

# ---------------------------------------------------------------------------
# Robustness guard — quality gate (pre-inference).  Runs before the
# preview so an unreadable or rejected photo is reported with the
# standard guard message instead of crashing the image preview.
# ---------------------------------------------------------------------------

if _img_bytes is None:
    st.info("Upload a clear leaf image to begin analysis.")
    st.stop()

_guard_thresholds = load_guard_thresholds(crop_cfg)

try:
    _pil_image = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    _quality_metrics = compute_quality_metrics(_pil_image)
    _quality_verdict = evaluate_quality(_quality_metrics, _guard_thresholds)
except Exception:
    st.error(
        f"{REASON_MESSAGES['unreadable_image']} "
        "Please upload a valid JPG, PNG, or BMP image."
    )
    st.stop()

if _quality_verdict["status"] == "reject":
    _reasons = "\n".join(
        f"- {REASON_MESSAGES[r]}" for r in _quality_verdict["reasons"]
    )
    st.error(
        "This image cannot be analyzed:\n\n"
        f"{_reasons}\n\n"
        "Please upload a clear, well-lit color photo of a plant leaf."
    )
    st.stop()

if _quality_verdict["status"] != QUALITY_OK:
    for r in _quality_verdict["reasons"]:
        st.caption(f"Note: {REASON_MESSAGES[r]}")

# ---------------------------------------------------------------------------
# Image preview
# ---------------------------------------------------------------------------

with st.container(border=True):
    _preview_col, _checklist_col = st.columns([3, 2])
    with _preview_col:
        if _img_label is not None:
            st.markdown(
                '<span class="agri-demo-chip">🌱 Demo example</span>',
                unsafe_allow_html=True,
            )
            st.caption(_img_label)
        st.image(
            io.BytesIO(_img_bytes),
            caption=_img_name,
            width="stretch",
        )
    with _checklist_col:
        st.markdown("**Photo checklist**")
        st.markdown(
            "- One leaf fills the frame\n"
            "- Natural daylight\n"
            "- Leaf in sharp focus\n"
            "- Plain color photo"
        )

# ---------------------------------------------------------------------------
# Crop verification — is this image the selected crop?
# A wrong-crop image must never reach the disease model: stop the
# diagnosis with a clear, non-absolute message instead.
# ---------------------------------------------------------------------------

_crop_check = verify_crop(_pil_image, selected_crop)
if _crop_check.status == "mismatch":
    st.error(_crop_check.message)
    st.stop()
if _crop_check.status == "uncertain":
    st.warning(_crop_check.message)
    st.stop()
if _crop_check.status == "ok":
    st.caption(
        f"Crop check: {selected_crop} leaf confirmed "
        f"({_crop_check.confidence:.0%} confidence)."
    )

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

model_path = str(_PROJECT_ROOT / crop_cfg["model_path"])
if not Path(model_path).exists():
    st.error(f"Model not found: {model_path}")
    st.stop()

model = _load_model(model_path)

# ---------------------------------------------------------------------------
# Analyze button + pipeline
# ---------------------------------------------------------------------------

def _run_analysis() -> None:
    """Run the existing inference pipeline on the active image.

    The pipeline (quality guard -> crop verification -> prediction ->
    Grad-CAM) is unchanged; this wrapper only lets the Analyze button and
    the retry button share one code path.
    """
    tmp_path: str | None = None
    try:
        # Save the active image bytes to a temp file (inference expects a path)
        suffix = Path(_img_name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(_img_bytes)
            tmp_path = tmp.name

        with st.spinner("Analyzing..."):
            # --- Inference + Grad-CAM via existing pipeline ---
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

            # --- Build overlay for display ---
            heatmap = result["gradcam"]
            transform = get_eval_transforms(crop_cfg["image_size"])
            img_tensor = (
                transform(Image.open(tmp_path).convert("RGB"))
                .unsqueeze(0)
            )
            original_np = denormalize(img_tensor.squeeze(0))
            overlay_np = generate_overlay(
                original_np,
                heatmap,
                alpha=crop_cfg.get("gradcam_alpha", 0.4),
            )

            st.session_state["result"] = result
            st.session_state["overlay"] = overlay_np
            st.session_state["uncertainty"] = evaluate_uncertainty(
                result["probabilities"], _guard_thresholds
            )
            st.session_state.pop("_analysis_error", None)

    except Exception as exc:
        # Keep the full error visible; never suppress real failures.
        st.session_state["_analysis_error"] = str(exc)
    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)


if st.button("Analyze Leaf", type="primary", width="stretch"):
    _run_analysis()

# -- Error recovery: retry the same image, or start over -----------------
if st.session_state.get("_analysis_error"):
    st.error(f"Analysis failed: {st.session_state['_analysis_error']}")
    st.caption(
        "You can retry the analysis with the same image, or start over "
        "with a different photo."
    )
    _retry_col, _start_over_col = st.columns(2)
    with _retry_col:
        if st.button("↻ Retry analysis", width="stretch"):
            _run_analysis()
    with _start_over_col:
        if st.button("Start over", width="stretch"):
            st.session_state["_pending_reset"] = True
            st.rerun()

# ---------------------------------------------------------------------------
# Step 3 — Diagnosis (results display)
# ---------------------------------------------------------------------------

if "result" in st.session_state:
    result = st.session_state["result"]
    overlay_np = st.session_state["overlay"]
    class_descs = crop_cfg.get("class_descriptions", {})
    _uncertainty = st.session_state.get("uncertainty")

    _step_header(3, "Diagnosis", "AI result with confidence and explanation")

    # -- Result card: crop, disease, confidence, status, next step --
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    description = class_descs.get(pred_class, "")
    _sev_style, _sev_label = _severity_for(pred_class)

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="agri-result-row">
                <span class="agri-result-key">Crop</span>
                <span class="agri-result-val">{_crop_label(selected_crop)}</span>
            </div>
            <div class="agri-result-row">
                <span class="agri-result-key">Disease</span>
                <span class="agri-result-val">
                    <span class="agri-disease-name">{pred_class}</span>
                </span>
            </div>
            <div class="agri-result-row">
                <span class="agri-result-key">Confidence</span>
                <span class="agri-result-val">{confidence:.1%}</span>
            </div>
            <div class="agri-result-row">
                <span class="agri-result-key">Status</span>
                <span class="agri-result-val">
                    <span class="agri-badge {_sev_style}">{_sev_label}</span>
                </span>
            </div>
            <div class="agri-result-row">
                <span class="agri-result-key">Recommended next step</span>
                <span class="agri-result-val">{_next_step_for(pred_class)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _confidence_meter(confidence)

        # -- Uncertainty caution banner (robustness guard) --
        if _uncertainty and _uncertainty["cautions"]:
            _caution_msgs = " ".join(
                REASON_MESSAGES[c] for c in _uncertainty["cautions"]
            )
            st.warning(
                f"{_caution_msgs} Please double-check the image or consult "
                "an agricultural expert before acting on this prediction."
            )

        if description:
            st.info(description)

    # -- Explainability: why did AgriMind make this prediction? --
    st.markdown("#### Why did AgriMind make this prediction?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Your leaf**")
        st.image(io.BytesIO(_img_bytes), width="stretch")
    with col2:
        st.markdown("**Where the AI looked (Grad-CAM)**")
        st.image(overlay_np, width="stretch")

    st.caption(
        "The heatmap shows where the AI model focused its attention "
        "when making the prediction. Warmer colors (red/yellow) "
        "indicate regions that most influenced the decision."
    )
    _gradcam_legend()

    # -- Class probabilities (sorted, predicted class highlighted) --
    st.markdown("#### Confidence by condition")
    probs = result["probabilities"]
    for cls_name, prob in sorted(
        probs.items(), key=lambda kv: kv[1], reverse=True
    ):
        if cls_name == pred_class:
            st.markdown(
                f'<p class="agri-prob-predicted">{cls_name} — {prob:.1%}</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{cls_name}** — {prob:.1%}")
        st.progress(prob)

    # -- AI disclaimer --
    st.warning(
        "This is an AI-assisted screening tool, not a replacement for "
        "agricultural experts."
    )

    # -- Reset: analyze another leaf --------------------------------------
    if st.button("🔄 Analyze Another Leaf", width="stretch"):
        st.session_state["_pending_reset"] = True
        st.rerun()

# ---------------------------------------------------------------------------
# Step 4 — Farmer Assistant
# ---------------------------------------------------------------------------

st.divider()
_step_header(4, "🤝 Ask AgriMind", "Trilingual guidance based on your diagnosis")

with st.container(border=True):
    _assistant_context = st.session_state.get("result")
    if _assistant_context:
        st.caption(
            f"AgriMind answers using your latest result: "
            f"{_crop_label(selected_crop)} · "
            f"{_assistant_context['predicted_class']} "
            f"({_assistant_context['confidence']:.0%} confidence) — "
            "in your language."
        )
    else:
        st.caption(
            f"Ask AgriMind about your {_crop_label(selected_crop)} crop — "
            "answers in your language."
        )

    # Language radio stays in sync with the sidebar language selector.
    st.session_state["main_language"] = st.session_state["_selected_language"]
    st.radio(
        "Response language",
        list(_LANGUAGE_LABELS),
        format_func=lambda code: _LANGUAGE_LABELS.get(code, code),
        horizontal=True,
        key="main_language",
        on_change=_on_main_language_change,
    )
    _language = st.session_state["_selected_language"]

    # Quick-pick example questions (existing supported intents only — no
    # new advisory content).  A clicked chip fills the question box and
    # asks it in the same run.
    _quick_clicked = None
    _quick_cols = st.columns(len(_QUICK_QUESTIONS))
    for _i, _quick_q in enumerate(_QUICK_QUESTIONS):
        with _quick_cols[_i]:
            if st.button(_quick_q, key=f"quick_question_{_i}", width="stretch"):
                _quick_clicked = _quick_q

    if _quick_clicked is not None:
        # Setting the widget key before the text input is instantiated is
        # allowed, so the clicked question appears in the box right away.
        st.session_state["assistant_question"] = _quick_clicked

    _farmer_question_text = st.text_input(
        "Your question (optional)",
        placeholder="e.g. What are the symptoms? / Iska ilaj kya hai?",
        key="assistant_question",
    )

    if st.button("Ask AgriMind", width="stretch") or _quick_clicked is not None:
        _result = st.session_state.get("result")
        _assistant_question = FarmerQuestion(
            crop=selected_crop,
            predicted_disease=_result.get("predicted_class") if _result else None,
            confidence=_result.get("confidence") if _result else None,
            question=_farmer_question_text,
            language=_language,
        )
        st.session_state["assistant_answer"] = answer_farmer_question(
            _assistant_question,
            knowledge_dir=str(_PROJECT_ROOT / "configs" / "knowledge"),
        )

    if "assistant_answer" in st.session_state:
        _answer = st.session_state["assistant_answer"]
        if _answer.status == "ok":
            # Render the existing response as Markdown so headings,
            # bullets and emphasis stay readable.
            with st.container(border=True):
                st.markdown(_format_assistant_response(_answer.response))
            st.caption(
                "This guidance is general advice, not a prescription. Always "
                "confirm with your local agricultural extension office."
            )
        else:
            st.warning(_answer.response)

# ---------------------------------------------------------------------------
# Technical details (main area) — quality & uncertainty metrics
# ---------------------------------------------------------------------------

if "result" in st.session_state:
    _uncertainty = st.session_state.get("uncertainty")
    with st.expander("Technical Details"):
        st.markdown("**Image Quality**")
        st.markdown(
            f"- Sharpness (Laplacian variance): "
            f"{_quality_metrics.laplacian_variance:.0f} — higher means sharper"
        )
        st.markdown(
            f"- Plant-like green content: "
            f"{_quality_metrics.green_fraction:.1%} of the image"
        )
        st.markdown(
            f"- Image coherence (adjacent-pixel correlation): "
            f"{_quality_metrics.adjacent_correlation:.2f} — natural photos "
            "are high; noise is near zero"
        )
        if _uncertainty:
            st.markdown("**Prediction Uncertainty**")
            st.markdown(
                f"- Confidence (top-1 probability): {_uncertainty['msp']:.1%}"
            )
            st.markdown(
                f"- Margin (top-1 minus top-2): {_uncertainty['margin']:.1%}"
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "AgriMind AI — Smart Agriculture · AI-powered crop health "
    "intelligence for farmers"
)
st.caption(
    "This is an AI-assisted screening tool, not a replacement for "
    "agricultural experts."
)
