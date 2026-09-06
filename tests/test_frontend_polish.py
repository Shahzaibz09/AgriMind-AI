"""
Tests for the Streamlit frontend polish — app/app.py.

Drives the real application script with ``streamlit.testing.v1.AppTest``
(no browser, no new dependencies):

- Default crop is Tomato; main-page crop radio and sidebar crop
  selectbox stay synchronized in both directions.
- Demo sample images load and analyze correctly for all three crops.
- Real upload flow works for all three crops (prediction + Grad-CAM).
- Crop mismatch still blocks diagnosis (demo safety).
- "Analyze Another Leaf" resets only analysis/session state.
- A new upload supersedes the demo sample.
- The quality guard still rejects unreadable images.
- Farmer Assistant quick-pick questions work (symptoms / management
  intents — existing supported intents only).
- Assistant answers in English, Urdu, and Roman Urdu; the sidebar and
  main-page language selectors stay synchronized.

These tests require the trained checkpoints and the demo images; they
skip when artifacts are missing.
"""

from __future__ import annotations

from pathlib import Path

# Import the real ``app`` package BEFORE AppTest executes the app script.
# Streamlit's script runner prepends the script directory (``<root>/app``) to
# ``sys.path`` while running ``app/app.py``.  Under pytest the project root
# is already on ``sys.path``, so the app's own bootstrap does not move it to
# the front — and ``app/app.py`` would then shadow the ``app`` *package*,
# breaking every ``from app.* import ...`` inside the script with
# ``ModuleNotFoundError: No module named 'app.assistant'``.  Importing the
# package here pins ``sys.modules["app"]`` so the script imports resolve
# against the package regardless of ``sys.path`` ordering.
import app  # noqa: F401  (imported for its sys.modules side effect)
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app" / "app.py"
DEMO_DIR = PROJECT_ROOT / "app" / "demo_images"

# crop -> (demo file name, expected predicted class)
DEMO_FILES = {
    "Tomato": ("tomato_demo.JPG", "Early Blight"),
    "Potato": ("potato_demo.JPG", "Late Blight"),
    "Apple": ("apple_demo.JPG", "Apple Scab"),
}

REQUIRED_ARTIFACTS = (
    PROJECT_ROOT / "models" / "best_model.pth",
    PROJECT_ROOT / "models" / "potato_best_model.pth",
    PROJECT_ROOT / "models" / "apple_best_model.pth",
    PROJECT_ROOT / "models" / "crop_classifier.pth",
)


def _artifacts_available() -> bool:
    return all(p.exists() for p in REQUIRED_ARTIFACTS) and all(
        (DEMO_DIR / name).exists() for name, _ in DEMO_FILES.values()
    )


pytestmark = pytest.mark.skipif(
    not _artifacts_available(), reason="checkpoints/demo images not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app() -> AppTest:
    """Fresh AppTest with one completed run (default state)."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=600)
    at.run()
    assert not at.exception
    return at


def _button(at: AppTest, label: str):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"Button not found: {label!r}")


def _click(at: AppTest, label: str) -> None:
    """Click the button with *label* and re-run the app."""
    _button(at, label).click()
    at.run()
    assert not at.exception


def _upload(at: AppTest, name: str, data: bytes) -> None:
    """Upload *data* as *name* and re-run the app."""
    at.file_uploader[0].set_value((name, data, "image/jpeg"))
    at.run()
    assert not at.exception


def _select_crop(at: AppTest, crop: str) -> None:
    """Switch crop via the main-page radio and re-run."""
    at.radio[0].set_value(crop)
    at.run()
    assert not at.exception


def _use_demo(at: AppTest, crop: str) -> None:
    """Click the demo-sample button for *crop* and re-run."""
    _select_crop(at, crop)
    demo_buttons = [
        b for b in at.button if b.label.startswith("🌱 Try a demo image")
    ]
    assert demo_buttons, f"demo button not rendered for {crop}"
    demo_buttons[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["_demo_sample"]["crop"] == crop


def _analyze(at: AppTest) -> dict:
    """Click Analyze and return the stored result."""
    _click(at, "Analyze Leaf")
    assert "result" in at.session_state, "no result after Analyze"
    assert "overlay" in at.session_state, "no Grad-CAM overlay after Analyze"
    return at.session_state["result"]


def _demo_bytes(crop: str) -> bytes:
    return (DEMO_DIR / DEMO_FILES[crop][0]).read_bytes()


# ---------------------------------------------------------------------------
# Default crop & selector synchronization
# ---------------------------------------------------------------------------


class TestCropSelection:
    def test_default_crop_is_tomato(self):
        at = _app()
        assert at.session_state["_selected_crop"] == "Tomato"
        assert at.radio[0].value == "Tomato"
        assert at.sidebar.selectbox[0].value == "Tomato"

    def test_main_radio_updates_sidebar_selectbox(self):
        at = _app()
        _select_crop(at, "Potato")
        assert at.session_state["_selected_crop"] == "Potato"
        assert at.sidebar.selectbox[0].value == "Potato"
        assert at.radio[0].value == "Potato"

    def test_sidebar_selectbox_updates_main_radio(self):
        at = _app()
        at.sidebar.selectbox[0].set_value("Apple")
        at.run()
        assert not at.exception
        assert at.session_state["_selected_crop"] == "Apple"
        assert at.radio[0].value == "Apple"
        assert at.sidebar.selectbox[0].value == "Apple"

    def test_all_three_crops_available_in_both_selectors(self):
        at = _app()
        expected = {"Tomato", "Potato", "Apple"}
        # The raw widget values are crop names (format_func is display-only).
        assert at.radio[0].value in expected
        assert at.sidebar.selectbox[0].value in expected
        # Cycling through every crop via the sidebar keeps the main-page
        # radio synchronized.
        for crop in ("Potato", "Apple", "Tomato"):
            at.sidebar.selectbox[0].set_value(crop)
            at.run()
            assert not at.exception
            assert at.radio[0].value == crop
            assert at.session_state["_selected_crop"] == crop


# ---------------------------------------------------------------------------
# Demo sample images
# ---------------------------------------------------------------------------


class TestDemoSamples:
    def test_demo_tomato_analyzes_correctly(self):
        at = _app()
        _use_demo(at, "Tomato")
        result = _analyze(at)
        assert result["predicted_class"] == "Early Blight"
        assert result["confidence"] > 0.9

    def test_demo_potato_analyzes_correctly(self):
        at = _app()
        _use_demo(at, "Potato")
        result = _analyze(at)
        assert result["predicted_class"] == "Late Blight"
        assert result["confidence"] > 0.9

    def test_demo_apple_analyzes_correctly(self):
        at = _app()
        _use_demo(at, "Apple")
        result = _analyze(at)
        assert result["predicted_class"] == "Apple Scab"
        assert result["confidence"] > 0.9

    def test_demo_sample_is_cleared_when_crop_changes(self):
        at = _app()
        _use_demo(at, "Tomato")
        _select_crop(at, "Potato")
        assert "_demo_sample" not in at.session_state


# ---------------------------------------------------------------------------
# Real upload flow (all three crops)
# ---------------------------------------------------------------------------


class TestUploadFlow:
    def test_upload_tomato_leaf(self):
        at = _app()
        _upload(at, "my_tomato.jpg", _demo_bytes("Tomato"))
        result = _analyze(at)
        assert result["predicted_class"] == "Early Blight"

    def test_upload_potato_leaf(self):
        at = _app()
        _select_crop(at, "Potato")
        _upload(at, "my_potato.jpg", _demo_bytes("Potato"))
        result = _analyze(at)
        assert result["predicted_class"] == "Late Blight"

    def test_upload_apple_leaf(self):
        at = _app()
        _select_crop(at, "Apple")
        _upload(at, "my_apple.jpg", _demo_bytes("Apple"))
        result = _analyze(at)
        assert result["predicted_class"] == "Apple Scab"

    def test_gradcam_overlay_has_image_shape(self):
        at = _app()
        _use_demo(at, "Tomato")
        _analyze(at)
        overlay = at.session_state["overlay"]
        assert overlay is not None
        assert overlay.shape[0] > 0 and overlay.shape[1] > 0


# ---------------------------------------------------------------------------
# Crop verification gate (demo safety)
# ---------------------------------------------------------------------------


class TestCropMismatch:
    def test_tomato_image_under_potato_is_blocked(self):
        at = _app()
        _select_crop(at, "Potato")
        _upload(at, "wrong_crop.jpg", _demo_bytes("Tomato"))
        assert at.error, "crop mismatch must raise an error"
        assert "Crop mismatch" in at.error[0].value
        assert "Tomato" in at.error[0].value
        assert "result" not in at.session_state, (
            "diagnosis must not run on a wrong-crop image"
        )

    def test_new_upload_supersedes_demo_sample(self):
        at = _app()
        _use_demo(at, "Tomato")
        _upload(at, "my_own.jpg", _demo_bytes("Tomato"))
        assert "_demo_sample" not in at.session_state
        result = _analyze(at)
        assert result["predicted_class"] == "Early Blight"


# ---------------------------------------------------------------------------
# Quality guard
# ---------------------------------------------------------------------------


class TestQualityGuard:
    def test_unreadable_image_is_rejected(self):
        at = _app()
        _upload(at, "broken.jpg", b"this is not an image")
        assert at.error, "unreadable upload must raise an error"
        assert "could not be read as an image" in at.error[0].value
        assert "result" not in at.session_state


# ---------------------------------------------------------------------------
# Reset ("Analyze Another Leaf")
# ---------------------------------------------------------------------------


class TestResetFlow:
    def test_reset_clears_analysis_state_but_keeps_crop(self):
        at = _app()
        _use_demo(at, "Tomato")
        _analyze(at)
        _click(at, "What are the symptoms?")  # populate assistant_answer
        assert "assistant_answer" in at.session_state

        _click(at, "🔄 Analyze Another Leaf")

        for key in ("result", "overlay", "uncertainty", "assistant_answer",
                    "_demo_sample", "_analysis_error"):
            assert key not in at.session_state, f"{key} must be reset"
        # Crop selection and model state are untouched by the reset.
        assert at.session_state["_selected_crop"] == "Tomato"
        assert at.file_uploader[0].value is None

    def test_fresh_analysis_after_reset(self):
        at = _app()
        _use_demo(at, "Tomato")
        _analyze(at)
        _click(at, "🔄 Analyze Another Leaf")
        # Demo button works again right after the reset.
        _use_demo(at, "Potato")
        result = _analyze(at)
        assert result["predicted_class"] == "Late Blight"


# ---------------------------------------------------------------------------
# Farmer Assistant — quick questions and languages
# ---------------------------------------------------------------------------


class TestFarmerAssistant:
    def _prepare(self) -> AppTest:
        at = _app()
        _use_demo(at, "Tomato")
        _analyze(at)
        return at

    def test_quick_question_symptoms(self):
        at = self._prepare()
        _click(at, "What are the symptoms?")
        answer = at.session_state["assistant_answer"]
        assert answer.status == "ok"
        assert answer.sections["intent"] == "symptoms"
        assert "Common Symptoms" in answer.response

    def test_quick_question_management(self):
        at = self._prepare()
        _click(at, "How can I manage it?")
        answer = at.session_state["assistant_answer"]
        assert answer.status == "ok"
        assert answer.sections["intent"] == "management"
        assert "Prevention and Management" in answer.response

    def test_quick_question_prevent_maps_to_management(self):
        at = self._prepare()
        _click(at, "How can I prevent it?")
        answer = at.session_state["assistant_answer"]
        assert answer.status == "ok"
        assert answer.sections["intent"] == "management"

    def test_answer_in_english(self):
        at = self._prepare()
        _click(at, "What are the symptoms?")
        answer = at.session_state["assistant_answer"]
        assert answer.language == "en"
        assert "Common Symptoms" in answer.response

    def test_answer_in_urdu(self):
        at = self._prepare()
        at.radio[1].set_value("ur")
        at.run()
        assert not at.exception
        _click(at, "What are the symptoms?")
        answer = at.session_state["assistant_answer"]
        assert answer.language == "ur"
        assert "عام علامات" in answer.response

    def test_answer_in_roman_urdu(self):
        at = self._prepare()
        at.radio[1].set_value("roman_ur")
        at.run()
        assert not at.exception
        _click(at, "What are the symptoms?")
        answer = at.session_state["assistant_answer"]
        assert answer.language == "roman_ur"
        assert "Aam Alamaat" in answer.response

    def test_language_selectors_stay_synchronized(self):
        at = self._prepare()
        # Sidebar -> main radio
        at.sidebar.selectbox[1].set_value("ur")
        at.run()
        assert not at.exception
        assert at.session_state["_selected_language"] == "ur"
        assert at.radio[1].value == "ur"
        # Main radio -> sidebar
        at.radio[1].set_value("roman_ur")
        at.run()
        assert not at.exception
        assert at.session_state["_selected_language"] == "roman_ur"
        assert at.sidebar.selectbox[1].value == "roman_ur"

    def test_assistant_without_prediction_prompts_for_analysis(self):
        at = _app()
        _use_demo(at, "Tomato")
        # Ask before analyzing (no result yet).
        _click(at, "What are the symptoms?")
        answer = at.session_state["assistant_answer"]
        assert answer.status == "missing_prediction"
