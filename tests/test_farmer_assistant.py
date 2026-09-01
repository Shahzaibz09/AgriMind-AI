"""
Tests for app.assistant — Farmer Assistant Module.

Covers knowledge loading, content safety, language selection,
disease lookup, missing prediction, unknown disease, question intent
handling, confidence notes, and safe fallback responses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from app.assistant import (
    STATUS_KNOWLEDGE_UNAVAILABLE,
    STATUS_MISSING_PREDICTION,
    STATUS_OK,
    STATUS_UNKNOWN_DISEASE,
    FarmerQuestion,
    answer_farmer_question,
    get_disease_knowledge,
    load_knowledge,
    normalize_language,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "configs" / "knowledge"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_urdu_script(text: str) -> bool:
    """Return True if the text contains Arabic-script (Urdu) characters."""
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _ask_dir(knowledge_dir, **kwargs):
    defaults = dict(
        crop="Tomato",
        predicted_disease="Early Blight",
        question="",
        language="en",
    )
    defaults.update(kwargs)
    return answer_farmer_question(
        FarmerQuestion(**defaults), knowledge_dir=knowledge_dir
    )


def _ask(**kwargs):
    return _ask_dir(KNOWLEDGE_DIR, **kwargs)


def _write_kb(tmp_path, crop="Tomato", langs=("en",), disease="Late Blight"):
    """Write a minimal valid knowledge base for robustness tests."""
    entry = {
        "title": f"{disease} Title",
        "explanation": "A serious disease. Contact your extension office.",
        "symptoms": ["Dark patches on leaves"],
        "management": ["Remove infected plants"],
        "expert_advice": "Contact your local agricultural extension office.",
    }
    kb = {
        "crop_name": crop,
        "diseases": {disease: {lang: dict(entry) for lang in langs}},
        "fallbacks": {
            "missing_prediction": {"en": "Analyze a leaf image first."},
            "unknown_disease": {
                "en": "No specific guidance. Please contact an expert."
            },
        },
    }
    path = tmp_path / f"{crop.lower()}.yaml"
    path.write_text(yaml.safe_dump(kb, allow_unicode=True), encoding="utf-8")
    return path


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
# Knowledge loading
# ---------------------------------------------------------------------------


class TestKnowledgeLoading:
    def test_tomato_knowledge_loads(self):
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        assert knowledge["crop_name"] == "Tomato"
        assert "diseases" in knowledge

    def test_case_insensitive_crop_match(self):
        knowledge = load_knowledge("tomato", KNOWLEDGE_DIR)
        assert knowledge["crop_name"] == "Tomato"

    def test_all_three_diseases_present(self):
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        assert set(knowledge["diseases"]) == {
            "Healthy",
            "Early Blight",
            "Late Blight",
        }

    def test_all_languages_available_per_disease(self):
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            assert set(entry) == {"en", "ur", "roman_ur"}, disease

    def test_required_sections_per_language(self):
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        required = {"title", "explanation", "symptoms", "management", "expert_advice"}
        for disease, entry in knowledge["diseases"].items():
            for lang, content in entry.items():
                assert required.issubset(content), (disease, lang)
                assert content["symptoms"], (disease, lang)
                assert content["management"], (disease, lang)

    def test_unknown_crop_raises(self):
        with pytest.raises(FileNotFoundError):
            load_knowledge("Wheat", KNOWLEDGE_DIR)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_knowledge("Tomato", tmp_path / "does_not_exist")

    def test_fallbacks_declared_in_knowledge(self):
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        fallbacks = knowledge.get("fallbacks", {})
        for kind in ("missing_prediction", "unknown_disease", "low_confidence_note"):
            assert kind in fallbacks
            assert "en" in fallbacks[kind]


# ---------------------------------------------------------------------------
# Content safety — no pesticide dosages or chemical prescriptions
# ---------------------------------------------------------------------------


class TestContentSafety:
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
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        for text in _iter_strings(knowledge):
            assert re.search(pattern, text, re.IGNORECASE) is None, (
                f"Pattern {pattern!r} matched in: {text!r}"
            )

    def test_management_lists_are_cultural_only(self):
        """Management guidance must not prescribe chemical products."""
        banned = ("pesticide", "fungicide", "insecticide", "spray", "chemical")
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        for disease, entry in knowledge["diseases"].items():
            for lang, content in entry.items():
                for item in content["management"]:
                    lower = item.lower()
                    for word in banned:
                        assert word not in lower, (disease, lang, item)


# ---------------------------------------------------------------------------
# Language selection
# ---------------------------------------------------------------------------


class TestLanguageSelection:
    def test_normalize_english_aliases(self):
        assert normalize_language("en") == "en"
        assert normalize_language("English") == "en"
        assert normalize_language("EN") == "en"

    def test_normalize_urdu_aliases(self):
        assert normalize_language("ur") == "ur"
        assert normalize_language("Urdu") == "ur"
        assert normalize_language("اردو") == "ur"

    def test_normalize_roman_urdu_aliases(self):
        assert normalize_language("roman_ur") == "roman_ur"
        assert normalize_language("Roman Urdu") == "roman_ur"

    def test_unknown_language_defaults_to_english(self):
        assert normalize_language("fr") == "en"
        assert normalize_language("Punjabi") == "en"

    def test_none_and_empty_default_to_english(self):
        assert normalize_language(None) == "en"
        assert normalize_language("") == "en"

    def test_urdu_response_contains_urdu_script(self):
        answer = _ask(language="ur")
        assert answer.status == STATUS_OK
        assert answer.language == "ur"
        assert _has_urdu_script(answer.response)

    def test_roman_urdu_response_is_ascii(self):
        answer = _ask(language="roman_ur")
        assert answer.status == STATUS_OK
        assert answer.language == "roman_ur"
        assert answer.response.isascii()


# ---------------------------------------------------------------------------
# Disease lookup
# ---------------------------------------------------------------------------


class TestDiseaseLookup:
    def test_known_disease_returns_ok(self):
        answer = _ask(predicted_disease="Late Blight")
        assert answer.status == STATUS_OK
        assert answer.disease == "Late Blight"

    def test_full_card_contains_all_sections(self):
        answer = _ask(predicted_disease="Early Blight")
        resp = answer.response
        assert "Early Blight" in resp  # title
        assert "Explanation" in resp
        assert "Common Symptoms" in resp
        assert "Prevention and Management" in resp
        assert "When to Seek Expert Help" in resp

    def test_get_disease_knowledge_returns_language_entry(self):
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        entry = get_disease_knowledge(knowledge, "Early Blight", "ur")
        assert _has_urdu_script(entry["explanation"])

    def test_get_disease_knowledge_unknown_disease_raises(self):
        knowledge = load_knowledge("Tomato", KNOWLEDGE_DIR)
        with pytest.raises(KeyError):
            get_disease_knowledge(knowledge, "Bacterial Spot")

    def test_get_disease_knowledge_missing_language_raises(self, tmp_path):
        path = _write_kb(tmp_path, langs=("en",))
        knowledge = load_knowledge("Tomato", tmp_path)
        with pytest.raises(KeyError):
            get_disease_knowledge(knowledge, "Late Blight", "ur")
        assert path.exists()

    def test_healthy_guidance_is_positive(self):
        answer = _ask(predicted_disease="Healthy")
        assert answer.status == STATUS_OK
        assert "healthy" in answer.response.lower()

    def test_early_blight_mentions_pathogen(self):
        answer = _ask(predicted_disease="Early Blight")
        assert "Alternaria" in answer.response

    def test_late_blight_mentions_pathogen(self):
        answer = _ask(predicted_disease="Late Blight")
        assert "Phytophthora" in answer.response

    def test_unknown_disease_returns_safe_fallback(self):
        answer = _ask(predicted_disease="Bacterial Spot")
        assert answer.status == STATUS_UNKNOWN_DISEASE
        resp = answer.response.lower()
        assert "expert" in resp or "extension" in resp

    def test_unknown_disease_fallback_in_urdu(self):
        answer = _ask(predicted_disease="Bacterial Spot", language="ur")
        assert answer.status == STATUS_UNKNOWN_DISEASE
        assert _has_urdu_script(answer.response)


# ---------------------------------------------------------------------------
# Missing prediction
# ---------------------------------------------------------------------------


class TestMissingPrediction:
    def test_missing_prediction_status(self):
        answer = _ask(predicted_disease=None)
        assert answer.status == STATUS_MISSING_PREDICTION

    def test_empty_disease_is_missing_prediction(self):
        answer = _ask(predicted_disease="")
        assert answer.status == STATUS_MISSING_PREDICTION

    def test_fallback_in_urdu(self):
        answer = _ask(predicted_disease=None, language="ur")
        assert answer.language == "ur"
        assert _has_urdu_script(answer.response)

    def test_fallback_in_roman_urdu(self):
        answer = _ask(predicted_disease=None, language="roman_ur")
        assert answer.language == "roman_ur"
        assert answer.response.isascii()

    def test_fallback_mentions_analyze(self):
        answer = _ask(predicted_disease=None)
        assert "analyze" in answer.response.lower()


# ---------------------------------------------------------------------------
# Question handling
# ---------------------------------------------------------------------------


class TestQuestionHandling:
    def test_symptom_question_intent(self):
        answer = _ask(question="What are the symptoms of this disease?")
        assert answer.sections["intent"] == "symptoms"

    def test_management_question_intent(self):
        answer = _ask(question="How do I treat this?")
        assert answer.sections["intent"] == "management"

    def test_what_is_question_intent(self):
        answer = _ask(question="What is early blight?")
        assert answer.sections["intent"] == "explanation"

    def test_empty_question_default_intent(self):
        answer = _ask(question="")
        assert answer.sections["intent"] == "explanation"

    def test_unmatched_question_default_intent(self):
        answer = _ask(question="Hello assistant")
        assert answer.sections["intent"] == "explanation"

    def test_urdu_keyword_symptom_intent(self):
        answer = _ask(question="اس بیماری کی علامات کیا ہیں؟", language="ur")
        assert answer.sections["intent"] == "symptoms"

    def test_roman_urdu_keyword_management_intent(self):
        answer = _ask(question="Iska ilaj kya hai?", language="roman_ur")
        assert answer.sections["intent"] == "management"

    def test_symptom_intent_reorders_response(self):
        answer = _ask(question="What are the symptoms?")
        resp = answer.response
        assert resp.index("Common Symptoms") < resp.index("Explanation")
        assert resp.index("Common Symptoms") < resp.index(
            "Prevention and Management"
        )

    def test_default_order_starts_with_explanation(self):
        answer = _ask(question="")
        resp = answer.response
        assert resp.index("Explanation") < resp.index("Common Symptoms")
        assert resp.index("Explanation") < resp.index(
            "Prevention and Management"
        )

    def test_all_sections_present_regardless_of_intent(self):
        for question in ("", "What are the symptoms?", "How to treat?"):
            answer = _ask(question=question)
            assert set(answer.sections) >= {
                "intent",
                "explanation",
                "symptoms",
                "management",
                "expert_advice",
            }


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_low_confidence_adds_note(self):
        answer = _ask(confidence=0.5)
        assert "Note:" in answer.response

    def test_low_confidence_note_in_urdu(self):
        answer = _ask(confidence=0.4, language="ur")
        assert "نوٹ" in answer.response

    def test_high_confidence_no_note(self):
        answer = _ask(confidence=0.95)
        assert "Note:" not in answer.response

    def test_none_confidence_no_note(self):
        answer = _ask(confidence=None)
        assert "Note:" not in answer.response

    def test_boundary_confidence_no_note(self):
        answer = _ask(confidence=0.75)
        assert "Note:" not in answer.response

    def test_confidence_passthrough(self):
        answer = _ask(confidence=0.42)
        assert answer.confidence == 0.42


# ---------------------------------------------------------------------------
# Robustness — safe fallbacks, never crash
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_missing_knowledge_dir_no_crash(self, tmp_path):
        answer = _ask_dir(tmp_path / "nope")
        assert answer.status == STATUS_KNOWLEDGE_UNAVAILABLE
        assert answer.response

    def test_empty_knowledge_dir_no_crash(self, tmp_path):
        answer = _ask_dir(tmp_path)
        assert answer.status == STATUS_KNOWLEDGE_UNAVAILABLE
        assert answer.response

    def test_corrupt_yaml_no_crash(self, tmp_path):
        (tmp_path / "broken.yaml").write_text(
            "key: [unclosed", encoding="utf-8"
        )
        answer = _ask_dir(tmp_path)
        assert answer.status == STATUS_KNOWLEDGE_UNAVAILABLE
        assert answer.response

    def test_crop_mismatch_no_crash(self, tmp_path):
        _write_kb(tmp_path, crop="Potato")
        answer = _ask_dir(tmp_path)  # asks for Tomato
        assert answer.status == STATUS_KNOWLEDGE_UNAVAILABLE

    def test_knowledge_unavailable_is_english_and_safe(self, tmp_path):
        answer = _ask_dir(tmp_path / "nope", language="ur")
        assert answer.language == "en"
        assert "extension office" in answer.response

    def test_corrupt_sibling_does_not_break_valid_file(self, tmp_path):
        _write_kb(tmp_path)  # tomato.yaml (valid)
        (tmp_path / "aaa_broken.yaml").write_text(
            "key: [unclosed", encoding="utf-8"
        )
        answer = _ask_dir(tmp_path, predicted_disease="Late Blight")
        assert answer.status == STATUS_OK

    def test_missing_language_falls_back_to_english(self, tmp_path):
        _write_kb(tmp_path, langs=("en",))
        answer = _ask_dir(tmp_path, predicted_disease="Late Blight", language="ur")
        assert answer.status == STATUS_OK
        assert answer.language == "en"

    def test_entry_without_english_degrades_safely(self, tmp_path):
        _write_kb(tmp_path, langs=("ur",))
        answer = _ask_dir(tmp_path, predicted_disease="Late Blight", language="en")
        assert answer.status == STATUS_UNKNOWN_DISEASE
        assert answer.response
