"""
Tests for AgriMind AI FastAPI Backend Endpoints.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

client = TestClient(app)
DEMO_DIR = Path(__file__).resolve().parent.parent / "app" / "demo_images"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Tomato" in data["available_crops"]
    assert "Potato" in data["available_crops"]
    assert "Apple" in data["available_crops"]


def test_get_crops():
    response = client.get("/api/v1/crops")
    assert response.status_code == 200
    data = response.json()
    assert "crops" in data
    crop_ids = [c["id"] for c in data["crops"]]
    assert "tomato" in crop_ids
    assert "potato" in crop_ids
    assert "apple" in crop_ids


def test_analyze_invalid_crop():
    dummy_img = io.BytesIO(b"fake image data")
    response = client.post(
        "/api/v1/analyze",
        data={"crop": "banana"},
        files={"image": ("test.jpg", dummy_img, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unsupported crop" in response.json()["detail"]


def test_analyze_unreadable_image():
    dummy_img = io.BytesIO(b"not a valid image")
    response = client.post(
        "/api/v1/analyze",
        data={"crop": "tomato"},
        files={"image": ("test.jpg", dummy_img, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unreadable_image"


def test_analyze_quality_rejected_grayscale():
    # Create flat grayscale image
    img = Image.new("L", (100, 100), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/v1/analyze",
        data={"crop": "tomato"},
        files={"image": ("gray.png", buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "quality_rejected"
    assert "grayscale_image" in data["reasons"] or "no_plant_material" in data["reasons"]


def test_analyze_real_tomato_demo_image():
    demo_img_path = DEMO_DIR / "tomato_demo.JPG"
    if not demo_img_path.exists():
        pytest.skip("tomato_demo.JPG fixture not found")

    with open(demo_img_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/analyze",
        data={"crop": "tomato"},
        files={"image": ("tomato_demo.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["crop"] == "tomato"
    assert data["predicted_disease"] in ["Healthy", "Early Blight", "Late Blight"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "gradcam_overlay_base64" in data
    assert data["gradcam_overlay_base64"].startswith("data:image/jpeg;base64,")


def test_analyze_crop_mismatch():
    # Apple image uploaded while Potato is selected
    apple_demo = DEMO_DIR / "apple_demo.JPG"
    if not apple_demo.exists():
        pytest.skip("apple_demo.JPG fixture not found")

    with open(apple_demo, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/analyze",
        data={"crop": "potato"},
        files={"image": ("apple.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    # It should either detect crop mismatch or uncertain
    assert data["status"] in ["crop_mismatch", "crop_uncertain"]


def test_assistant_english():
    response = client.post(
        "/api/v1/assistant",
        json={
            "crop": "tomato",
            "predicted_disease": "Early Blight",
            "confidence": 0.95,
            "question": "What are the symptoms?",
            "language": "en",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["language"] == "en"
    assert "Common Symptoms" in data["response"]


def test_assistant_urdu():
    response = client.post(
        "/api/v1/assistant",
        json={
            "crop": "tomato",
            "predicted_disease": "Early Blight",
            "confidence": 0.95,
            "question": "علامات کیا ہیں؟",
            "language": "ur",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["language"] == "ur"
    assert "علامات" in data["response"]


def test_assistant_roman_urdu():
    response = client.post(
        "/api/v1/assistant",
        json={
            "crop": "tomato",
            "predicted_disease": "Early Blight",
            "confidence": 0.95,
            "question": "Alamaat kya hain?",
            "language": "roman_ur",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["language"] == "roman_ur"
    assert "Aam Alamaat" in data["response"]


def test_assistant_missing_prediction():
    response = client.post(
        "/api/v1/assistant",
        json={
            "crop": "tomato",
            "predicted_disease": None,
            "question": "How to treat blight?",
            "language": "en",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "missing_prediction"
