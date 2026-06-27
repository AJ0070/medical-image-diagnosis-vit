"""Integration tests for the FastAPI endpoints using TestClient."""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_predictor():
    """Create a mock Predictor that returns a known result."""
    from inference.predictor import PredictionResult

    mock = MagicMock()
    mock.device = MagicMock(type="cpu")
    mock.cfg = {
        "model": {"name": "efficientnet_b0"},
        "project": {"name": "test"},
        "dataset": {"num_classes": 2, "classes": ["Normal", "Pneumonia"]},
    }
    result = PredictionResult(
        predicted_class=0,
        predicted_label="Normal",
        confidence=0.92,
        probabilities={"Normal": 0.92, "Pneumonia": 0.08},
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    mock.predict.return_value = result
    return mock


def _make_image_bytes() -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    from api import dependencies
    from api.main import app

    dependencies._predictor = _fake_predictor()
    dependencies._cfg = dependencies._predictor.cfg

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_predict_png(self, client):
        image_bytes = _make_image_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_label" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_unsupported_format(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.pdf", b"fake-pdf-content", "application/pdf")},
        )
        assert resp.status_code == 415


class TestModelInfoEndpoint:
    def test_model_info(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "num_classes" in data
