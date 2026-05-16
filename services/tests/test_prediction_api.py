"""Tests for the authenticated prediction API."""

from __future__ import annotations

import datetime as dt
import hmac

import numpy as np
import pytest
from fastapi import HTTPException

from services.prediction_api.app import (
    PredictionAPISettings,
    build_prediction_response,
    create_app,
    validate_api_key,
    validate_user_session,
)
from services.prediction_api.bundle import ServingBundle


class FakeRedis:
    def __init__(self, state: dict[str, dict[str, str]] | None = None):
        self.state = state or {}

    def hgetall(self, key):
        return self.state.get(key, {}).copy()

    def scard(self, key):
        if key.endswith(":products"):
            return 2
        if key.endswith(":categories"):
            return 1
        return 0


class RecordingModel:
    def __init__(self, probabilities: np.ndarray):
        self.probabilities = probabilities
        self.inputs = []

    def predict_proba(self, row):
        self.inputs.append(row.copy())
        return self.probabilities


def _settings() -> PredictionAPISettings:
    return PredictionAPISettings(
        api_key="secret-key",
        redis_url="redis://unused",
        mlflow_bundle_uri="runs:/winner-test-run",
    )


def _redis_state() -> dict[str, dict[str, str]]:
    return {
        "session:session-1": {
            "first_event_time": "2019-11-01T00:00:00+00:00",
            "last_event_time": "2019-11-01T00:03:00+00:00",
            "count_view": "3",
            "count_cart": "2",
            "count_remove_from_cart": "1",
            "latest_price": "12.5",
            "latest_category_id": "post-cat-id",
            "latest_category_code": "post.category.code",
            "latest_brand": "post-brand",
            "latest_event_type": "cart",
            "serving_total_views": "2",
            "serving_total_carts": "1",
            "serving_total_removes": "0",
            "serving_net_cart_count": "1",
            "serving_cart_to_view_ratio": "0.5",
            "serving_unique_categories": "2",
            "serving_unique_products": "3",
            "serving_session_duration_sec": "120.0",
            "serving_price": "9.99",
            "serving_category_id": "cat-id",
            "serving_category_code": "",
            "serving_brand": "",
            "serving_event_type": "view",
        }
    }


def _bundle(model) -> ServingBundle:
    return ServingBundle(
        model=model,
        model_uri="runs:/winner-test-run/serving/model/model.joblib",
        model_version="run-1",
        feature_column_order=[
            "total_views",
            "total_carts",
            "net_cart_count",
            "cart_to_view_ratio",
            "unique_categories",
            "unique_products",
            "session_duration_sec",
            "price",
            "category_id",
            "category_code",
            "brand",
            "event_type",
        ],
        category_maps={
            "category_id": {"__MISSING__": 0, "__UNK__": 1, "cat-id": 2},
            "category_code": {"__MISSING__": 0, "__UNK__": 1},
            "brand": {"__MISSING__": 0, "__UNK__": 1},
            "event_type": {"__MISSING__": 0, "__UNK__": 1, "view": 2, "cart": 3},
        },
        missing_token="__MISSING__",
        unknown_token="__UNK__",
        threshold=0.42,
        prediction_horizon_minutes=10,
        response_contract_version="v1",
    )


def test_settings_from_env_reads_runtime_values():
    settings = PredictionAPISettings.from_env(
        {
            "API_KEY": "env-secret",
            "REDIS_URL": "redis://example:6379/0",
            "MLFLOW_SERVING_BUNDLE_URI": "runs:/env-run",
        }
    )

    assert settings == PredictionAPISettings(
        api_key="env-secret",
        redis_url="redis://example:6379/0",
        mlflow_bundle_uri="runs:/env-run",
    )


def test_settings_from_env_requires_bundle_uri():
    with pytest.raises(ValueError, match="MLFLOW.*bundle"):
        PredictionAPISettings.from_env({"API_KEY": "env-secret"})


def test_create_app_exposes_health_and_predict_routes():
    app = create_app(_settings())

    routes = {route.path for route in app.routes}

    assert "/health" in routes
    assert "/api/v1/predict/{user_session}" in routes


def test_validate_api_key_rejects_missing_or_invalid_key():
    with pytest.raises(HTTPException, match="Unauthorized") as excinfo:
        validate_api_key("secret-key", None)
    assert excinfo.value.status_code == 401

    with pytest.raises(HTTPException, match="Unauthorized") as excinfo:
        validate_api_key("secret-key", "wrong")
    assert excinfo.value.status_code == 401


def test_validate_api_key_uses_constant_time_compare(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_compare_digest(left: str, right: str) -> bool:
        calls.append((left, right))
        return True

    monkeypatch.setattr(hmac, "compare_digest", fake_compare_digest)

    validate_api_key("secret-key", "secret-key")

    assert calls == [("secret-key", "secret-key")]


def test_validate_user_session_rejects_invalid_values():
    with pytest.raises(HTTPException, match="Invalid user_session") as excinfo:
        validate_user_session("bad session!")
    assert excinfo.value.status_code == 422


def test_build_prediction_response_returns_redis_miss_fallback_without_loading_bundle():
    loaded = []

    def bundle_loader():
        loaded.append("called")
        raise AssertionError("bundle should not load for redis miss")

    response = build_prediction_response(
        user_session="missing-session",
        redis_client=FakeRedis(),
        bundle_loader=bundle_loader,
    )

    assert response.purchase_probability == 0.5
    assert response.fallback_reason == "redis_miss"
    assert response.prediction_mode == "fallback"
    assert response.cached is False
    assert response.model_uri is None
    assert response.model_version is None
    assert response.prediction_horizon_minutes is None
    assert loaded == []


def test_build_prediction_response_returns_model_unavailable_when_bundle_load_fails():
    def bundle_loader():
        raise RuntimeError("bundle unavailable")

    response = build_prediction_response(
        user_session="session-1",
        redis_client=FakeRedis(_redis_state()),
        bundle_loader=bundle_loader,
    )

    assert response.purchase_probability == 0.5
    assert response.fallback_reason == "model_unavailable"
    assert response.prediction_mode == "fallback"
    assert response.cached is False


def test_build_prediction_response_returns_model_unavailable_when_prediction_fails():
    class BrokenModel:
        def predict_proba(self, row):
            raise RuntimeError("prediction failed")

    response = build_prediction_response(
        user_session="session-1",
        redis_client=FakeRedis(_redis_state()),
        bundle_loader=lambda: _bundle(BrokenModel()),
    )

    assert response.purchase_probability == 0.5
    assert response.fallback_reason == "model_unavailable"
    assert response.prediction_mode == "fallback"
    assert response.cached is False


def test_build_prediction_response_returns_model_backed_response_from_class_one_probability():
    model = RecordingModel(np.array([[0.25, 0.75]]))

    response = build_prediction_response(
        user_session="session-1",
        redis_client=FakeRedis(_redis_state()),
        bundle_loader=lambda: _bundle(model),
    )

    assert response.purchase_probability == 0.75
    assert response.prediction_mode == "model"
    assert response.fallback_reason is None
    assert response.cached is False
    assert response.model_uri == "runs:/winner-test-run/serving/model/model.joblib"
    assert response.model_version == "run-1"
    assert response.prediction_horizon_minutes == 10
    dt.datetime.fromisoformat(response.prediction_time)
    assert len(model.inputs) == 1
    assert model.inputs[0].columns.tolist() == [
        "total_views",
        "total_carts",
        "net_cart_count",
        "cart_to_view_ratio",
        "unique_categories",
        "unique_products",
        "session_duration_sec",
        "price",
        "category_id",
        "category_code",
        "brand",
        "event_type",
    ]
    assert model.inputs[0].iloc[0]["total_views"] == 2
    assert model.inputs[0].iloc[0]["total_carts"] == 1
    assert model.inputs[0].iloc[0]["net_cart_count"] == 1
    assert model.inputs[0].iloc[0]["unique_categories"] == 2
    assert model.inputs[0].iloc[0]["unique_products"] == 3
    assert model.inputs[0].iloc[0]["session_duration_sec"] == 120.0
    assert model.inputs[0].iloc[0]["price"] == 9.99
    assert model.inputs[0].iloc[0]["event_type"] == "view"
    assert model.inputs[0].iloc[0]["category_code"] == "__MISSING__"
    assert model.inputs[0].iloc[0]["brand"] == "__MISSING__"
