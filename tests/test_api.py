"""
Integration tests for the FastAPI app using TestClient.

These tests assume `python run_training.py` has been run at least once
so models/ contains the registry and at least one trained state.
If no models are present, the entire module is skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.dependencies import AppState
from main import app

REGISTRY = Path(__file__).resolve().parent.parent / "models" / "model_registry.joblib"
pytestmark = pytest.mark.skipif(
    not REGISTRY.exists(),
    reason="No trained models found — run `python run_training.py` first.",
)


@pytest.fixture(scope="module")
def client():
    AppState.load()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def known_state(client):
    """Pick the first available trained state for tests."""
    r = client.get("/models")
    return r.json()["models"][0]["state"]


class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["states_loaded"] > 0
        assert body["models_loaded"] > 0


class TestModels:
    def test_list_all_models(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        body = r.json()
        assert body["total_states"] > 0
        for m in body["models"]:
            assert m["best_model"] in {"arima", "sarima", "xgboost", "lstm", "prophet"}
            assert "rmse" in m["validation_metrics"][m["best_model"]]

    def test_filter_by_state_case_insensitive(self, client, known_state):
        r = client.get(f"/models?state={known_state.lower()}")
        assert r.status_code == 200
        assert r.json()["total_states"] == 1

    def test_unknown_state_returns_404(self, client):
        r = client.get("/models?state=Atlantis")
        assert r.status_code == 404

    def test_invalid_state_format_returns_422(self, client):
        r = client.get("/models?state=../../etc/passwd")
        assert r.status_code == 422


class TestPredict:
    def test_predict_returns_8_weeks(self, client, known_state):
        r = client.get(f"/predict?state={known_state}")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == known_state
        assert body["forecast_horizon_weeks"] == 8
        assert len(body["forecast"]) == 8

    def test_forecast_dates_are_consecutive_weeks(self, client, known_state):
        from datetime import date, timedelta
        r = client.get(f"/predict?state={known_state}")
        forecast = r.json()["forecast"]
        dates = [date.fromisoformat(p["date"]) for p in forecast]
        for i in range(1, len(dates)):
            assert (dates[i] - dates[i - 1]) == timedelta(days=7)

    def test_forecast_values_are_non_negative(self, client, known_state):
        r = client.get(f"/predict?state={known_state}")
        for p in r.json()["forecast"]:
            assert p["forecast_sales"] >= 0

    def test_unknown_state_returns_404(self, client):
        r = client.get("/predict?state=Atlantis")
        assert r.status_code == 404

    def test_path_traversal_rejected_with_422(self, client):
        r = client.get("/predict?state=../../etc/passwd")
        assert r.status_code == 422

    def test_injection_payload_rejected(self, client):
        r = client.get("/predict?state=California; DROP TABLE")
        assert r.status_code == 422

    def test_missing_state_returns_422(self, client):
        r = client.get("/predict")
        assert r.status_code == 422
