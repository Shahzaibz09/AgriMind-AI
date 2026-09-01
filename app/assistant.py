"""
AgriMind AI — Farmer Assistant Module.

Deterministic, rule-based advisory layer over a curated trilingual
(English / Urdu / Roman Urdu) knowledge base.  Receives the current
crop, predicted disease, prediction confidence, the farmer's question,
and the preferred language, and returns a structured advisory answer.

Design notes
------------
- Knowledge content lives in ``configs/knowledge/<crop>.yaml`` and is
  loaded at call time, so advisory text can be edited without code
  changes.
- This module has no Streamlit (or any other UI) dependency and makes
  no external LLM/API/network calls: when no external system is
  configured, the curated knowledge base is the only answer source.
- ``answer_farmer_question`` is the single public entry point.  A
  future RAG/LLM implementation can replace the internal lookup while
  keeping this interface — and the UI — unchanged.

Safety
------
Responses never include pesticide dosages or chemical prescriptions.
Management guidance is limited to cultural and preventive practices;
chemical treatment is always deferred to the local agricultural
extension office and product label instructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ("en", "ur", "roman_ur")
DEFAULT_LANGUAGE = "en"

DEFAULT_KNOWLEDGE_DIR = "configs/knowledge"

# Answer status codes
STATUS_OK = "ok"
STATUS_MISSING_PREDICTION = "missing_prediction"
STATUS_UNKNOWN_DISEASE = "unknown_disease"
STATUS_KNOWLEDGE_UNAVAILABLE = "knowledge_unavailable"

# Predictions below this confidence get an uncertainty note.
LOW_CONFIDENCE_THRESHOLD = 0.75

# Language aliases accepted by normalize_language (keys are lowercased).
_LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "ur": "ur",
    "urdu": "ur",
    "اردو": "ur",
    "roman_ur": "roman_ur",
    "roman urdu": "roman_ur",
}

# Section rendering order per detected question intent.
_INTENT_ORDER = {
    "explanation": ("explanation", "symptoms", "management", "expert_advice"),
    "symptoms": ("symptoms", "explanation", "management", "expert_advice"),
    "management": ("management", "symptoms", "explanation", "expert_advice"),
}

# Human-readable section headers per language.
_SECTION_HEADERS = {
    "en": {
        "explanation": "Explanation",
        "symptoms": "Common Symptoms",
        "management": "Prevention and Management",
        "expert_advice": "When to Seek Expert Help",
    },
    "ur": {
        "explanation": "وضاحت",
        "symptoms": "عام علامات",
        "management": "بچاؤ اور انتظام",
        "expert_advice": "ماہر سے رابطہ کب کریں",
    },
    "roman_ur": {
        "explanation": "Wazahat",
        "symptoms": "Aam Alamaat",
        "management": "Bachao aur Intezaam",
        "expert_advice": "Maahir se Rabta Kab Karein",
    },
}

# Intent keyword sets (substring match, case-insensitive; covers English,
# Roman Urdu, and Urdu script phrasing).  Checked in order: symptoms,
# management, what-is.
_SYMPTOM_KEYWORDS = (
    "symptom",
    "signs of",
    "look like",
    "how do i know",
    "identify",
    "nishaan",
    "alamat",
    "علامت",
    "علامات",
    "دھبے",
    "dhabay",
    "dhabbe",
    "peelapan",
    "پیلاہٹ",
)
_MANAGEMENT_KEYWORDS = (
    "treat",
    "manage",
    "control",
    "prevent",
    "cure",
    "fix",
    "solution",
    "save my crop",
    "ilaj",
    "ilaaj",
    "علاج",
    "bachao",
    "بچاؤ",
    "pani",
    "پانی",
)
_WHAT_IS_KEYWORDS = (
    "what is",
    "what are",
    "kya hai",
    "کیا ہے",
    "kyun",
    "kion",
    "why",
    "cause",
    "wajah",
    "سبب",
    "kya hota",
    "kya hoti",
)

# Hardcoded safe fallbacks used when the knowledge file is missing,
# corrupt, or incomplete — the assistant must always answer safely.
_BUILTIN_FALLBACKS = {
    "missing_prediction": {
        "en": (
            "Please analyze a leaf image first. Disease-specific guidance "
            "is available after a prediction."
        ),
    },
    "unknown_disease": {
        "en": (
            "I do not have specific guidance for this condition. Please "
            "contact your local agricultural extension office for expert "
            "confirmation."
        ),
    },
    "knowledge_unavailable": {
        "en": (
            "The knowledge base is currently unavailable. Please contact "
            "your local agricultural extension office for guidance."
        ),
    },
    "low_confidence_note": {
        "en": (
            "Note: The model's confidence in this prediction is low. "
            "Please confirm with an agricultural expert."
        ),
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FarmerQuestion:
    """A farmer's request to the assistant.

    Parameters
    ----------
    crop : str
        Crop name (must match ``crop_name`` in a knowledge YAML file).
    predicted_disease : str, optional
        Predicted disease from the classifier (``None`` if no analysis
        has been run yet).
    confidence : float, optional
        Prediction confidence in [0, 1] (``None`` if unknown).
    question : str
        Free-text question (may be empty).
    language : str
        Preferred language code or alias ("en", "ur", "roman_ur",
        "Urdu", ...).  Unknown values fall back to English.
    """

    crop: str
    predicted_disease: str | None = None
    confidence: float | None = None
    question: str = ""
    language: str = DEFAULT_LANGUAGE


@dataclass
class AssistantAnswer:
    """Structured assistant response.

    Attributes
    ----------
    status : str
        One of ``ok``, ``missing_prediction``, ``unknown_disease``,
        ``knowledge_unavailable``.
    language : str
        Language actually used for the response (after any fallback).
    response : str
        Full display text.
    disease : str or None
        Disease the response is about (``None`` if no prediction).
    confidence : float or None
        Prediction confidence passed through from the question.
    sections : dict
        Structured content used to build the response (intent,
        explanation, symptoms, management, expert_advice).
    """

    status: str
    language: str
    response: str
    disease: str | None = None
    confidence: float | None = None
    sections: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Language handling
# ---------------------------------------------------------------------------


def normalize_language(language: str | None) -> str:
    """Normalise a language name/alias to a supported language code.

    Accepts codes ("en", "ur", "roman_ur") and display names
    ("English", "Urdu", "اردو", "Roman Urdu").  Unknown or empty
    values fall back to English.
    """
    if not language:
        return DEFAULT_LANGUAGE
    key = str(language).strip().lower()
    return _LANGUAGE_ALIASES.get(key, DEFAULT_LANGUAGE)


# ---------------------------------------------------------------------------
# Knowledge loading
# ---------------------------------------------------------------------------


def load_knowledge(
    crop_name: str,
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
) -> dict[str, Any]:
    """Load the knowledge base for a crop.

    Scans *knowledge_dir* for ``*.yaml`` files and returns the first
    whose ``crop_name`` matches (case-insensitive).  Unreadable files
    are skipped so one corrupt file cannot break other crops.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or no matching crop is found.
    """
    knowledge_dir = Path(knowledge_dir)
    if not knowledge_dir.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    for yaml_file in sorted(knowledge_dir.glob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Skipping unreadable knowledge file %s: %s", yaml_file, exc)
            continue
        if (
            isinstance(data, dict)
            and str(data.get("crop_name", "")).lower() == str(crop_name).lower()
        ):
            return data

    raise FileNotFoundError(
        f"Knowledge for crop '{crop_name}' not found. "
        f"Available: {_available_crops(knowledge_dir)}"
    )


def _available_crops(knowledge_dir: Path) -> list[str]:
    """List crop names declared in the knowledge YAML files."""
    names: list[str] = []
    for yaml_file in sorted(knowledge_dir.glob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and data.get("crop_name"):
                names.append(str(data["crop_name"]))
        except Exception:  # noqa: BLE001 — best-effort listing
            continue
    return names


def get_disease_knowledge(
    knowledge: dict[str, Any],
    disease: str,
    language: str = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """Return the per-language knowledge entry for a disease.

    Parameters
    ----------
    knowledge : dict
        Knowledge base as returned by :func:`load_knowledge`.
    disease : str
        Disease name (must match a key under ``diseases``).
    language : str
        Language code or alias.  Note that unknown *aliases* normalise
        to English; only a supported code that is absent from the
        entry raises.

    Raises
    ------
    KeyError
        If the disease is unknown, or the (normalised) language is not
        present in the disease entry.
    """
    diseases = knowledge.get("diseases", {})
    if disease not in diseases:
        raise KeyError(
            f"Unknown disease: '{disease}'. Known: {sorted(diseases.keys())}"
        )
    entry = diseases[disease]
    lang = normalize_language(language)
    if lang not in entry:
        raise KeyError(
            f"Language '{lang}' not available for disease '{disease}'. "
            f"Available: {sorted(entry.keys())}"
        )
    return entry[lang]


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def _fallback_text(
    knowledge: dict[str, Any] | None,
    kind: str,
    language: str,
) -> str:
    """Resolve a fallback message, preferring the knowledge file.

    Falls back to the language default, then to the hardcoded builtin
    English text, so a safe response is always available.
    """
    if knowledge:
        entry = (knowledge.get("fallbacks") or {}).get(kind) or {}
        text = entry.get(language)
        if text:
            return str(text)
        text = entry.get(DEFAULT_LANGUAGE)
        if text:
            return str(text)
    builtin = _BUILTIN_FALLBACKS.get(kind, {})
    return str(builtin.get(language) or builtin.get(DEFAULT_LANGUAGE) or "")


# ---------------------------------------------------------------------------
# Question routing
# ---------------------------------------------------------------------------


def _detect_intent(question: str) -> str:
    """Detect the question intent: symptoms / management / explanation.

    Keywords are matched as substrings (case-insensitive) in the order
    symptoms, management, what-is; unmatched or empty questions use the
    default explanation-first order.
    """
    q = (question or "").strip().lower()
    if not q:
        return "explanation"
    if any(k in q for k in _SYMPTOM_KEYWORDS):
        return "symptoms"
    if any(k in q for k in _MANAGEMENT_KEYWORDS):
        return "management"
    if any(k in q for k in _WHAT_IS_KEYWORDS):
        return "explanation"
    return "explanation"


# ---------------------------------------------------------------------------
# Response rendering
# ---------------------------------------------------------------------------


def _render_card(entry: dict[str, Any], language: str, intent: str) -> str:
    """Render a disease knowledge entry as display text.

    The section order follows the detected intent; all available
    sections are always included.
    """
    headers = _SECTION_HEADERS.get(language) or _SECTION_HEADERS[DEFAULT_LANGUAGE]
    order = _INTENT_ORDER.get(intent, _INTENT_ORDER["explanation"])

    lines: list[str] = []
    title = str(entry.get("title", "")).strip()
    if title:
        lines.append(title)
        lines.append("")

    for section in order:
        if section in ("symptoms", "management"):
            items = entry.get(section) or []
            if items:
                lines.append(headers[section])
                lines.extend(f"- {item}" for item in items)
                lines.append("")
        else:  # explanation / expert_advice — plain text
            text = str(entry.get(section, "")).strip()
            if text:
                lines.append(headers[section])
                lines.append(text)
                lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def answer_farmer_question(
    question: FarmerQuestion,
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
) -> AssistantAnswer:
    """Answer a farmer's question using the curated knowledge base.

    This is the single public entry point of the assistant and the
    designated swap point for a future RAG/LLM backend: replacing the
    internal lookup while keeping this signature leaves the UI and the
    tests unchanged.

    Behaviour
    ---------
    - Knowledge file missing/corrupt -> ``knowledge_unavailable`` safe
      English fallback (never raises).
    - No prediction -> ``missing_prediction`` fallback in the
      requested language.
    - Prediction not in the knowledge base -> ``unknown_disease`` safe
      fallback in the requested language.
    - Known disease -> full knowledge card; the question intent
      reorders sections; low confidence (< 0.75) prepends an
      uncertainty note.  If the requested language is missing for the
      disease, the response falls back to English.
    """
    language = normalize_language(question.language)

    # -- 1. Load knowledge (graceful degradation) --------------------------
    knowledge: dict[str, Any] | None = None
    try:
        knowledge = load_knowledge(question.crop, knowledge_dir)
    except FileNotFoundError as exc:
        logger.warning("Knowledge not found for crop '%s': %s", question.crop, exc)
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Knowledge unreadable for crop '%s': %s", question.crop, exc)

    if knowledge is None:
        return AssistantAnswer(
            status=STATUS_KNOWLEDGE_UNAVAILABLE,
            language=DEFAULT_LANGUAGE,
            response=_fallback_text(None, "knowledge_unavailable", DEFAULT_LANGUAGE),
            disease=question.predicted_disease,
            confidence=question.confidence,
        )

    # -- 2. A prediction is required for disease-specific guidance ---------
    if not question.predicted_disease:
        return AssistantAnswer(
            status=STATUS_MISSING_PREDICTION,
            language=language,
            response=_fallback_text(knowledge, "missing_prediction", language),
            disease=question.predicted_disease,
            confidence=question.confidence,
        )

    # -- 3. The disease must be known --------------------------------------
    diseases = knowledge.get("diseases", {})
    if question.predicted_disease not in diseases:
        return AssistantAnswer(
            status=STATUS_UNKNOWN_DISEASE,
            language=language,
            response=_fallback_text(knowledge, "unknown_disease", language),
            disease=question.predicted_disease,
            confidence=question.confidence,
        )

    # -- 4. Build the knowledge card (language fallback) -------------------
    entry: dict[str, Any] | None = None
    for lang in (language, DEFAULT_LANGUAGE):
        try:
            entry = get_disease_knowledge(knowledge, question.predicted_disease, lang)
            language = lang
            break
        except KeyError:
            continue

    if entry is None:
        logger.warning(
            "No usable language entry for disease '%s'.",
            question.predicted_disease,
        )
        return AssistantAnswer(
            status=STATUS_UNKNOWN_DISEASE,
            language=language,
            response=_fallback_text(knowledge, "unknown_disease", language),
            disease=question.predicted_disease,
            confidence=question.confidence,
        )

    intent = _detect_intent(question.question)
    card = _render_card(entry, language, intent)

    response = card
    if (
        question.confidence is not None
        and question.confidence < LOW_CONFIDENCE_THRESHOLD
    ):
        note = _fallback_text(knowledge, "low_confidence_note", language)
        response = f"{note}\n\n{card}"

    return AssistantAnswer(
        status=STATUS_OK,
        language=language,
        response=response,
        disease=question.predicted_disease,
        confidence=question.confidence,
        sections={
            "intent": intent,
            "explanation": entry.get("explanation", ""),
            "symptoms": list(entry.get("symptoms") or []),
            "management": list(entry.get("management") or []),
            "expert_advice": entry.get("expert_advice", ""),
        },
    )
