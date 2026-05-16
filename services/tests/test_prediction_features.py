"""Tests for prediction API bundle loading and feature assembly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.prediction_api.bundle import (
    ServingBundle,
    load_serving_bundle,
    validate_prediction_contract,
)
from services.prediction_api.features import build_feature_row


class FakeRedis:
    def __init__(self):
        self.hashes = {
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
        self.sets = {
            "session:session-1:products": {"p1", "p2"},
            "session:session-1:categories": {"cat-id"},
        }

    def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    def scard(self, key):
        if key.endswith(":products"):
            return 99
        if key.endswith(":categories"):
            return 88
        return len(self.sets.get(key, set()))


def _bundle() -> ServingBundle:
    return ServingBundle(
        model=SimpleNamespace(name="model"),
        model_uri="runs:/run-1/serving/model",
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
        threshold=0.5,
        prediction_horizon_minutes=10,
        response_contract_version="v1",
    )


def test_validate_prediction_contract_rejects_non_v1():
    with pytest.raises(ValueError, match="Unsupported response contract version"):
        validate_prediction_contract({"response_contract_version": "v2"})


def test_load_serving_bundle_reads_bundle_artifacts(monkeypatch, tmp_path):
    loaded = {}

    def fake_load_dict(uri):
        loaded.setdefault("load_dict", []).append(uri)
        if uri.endswith("model_metadata.json"):
            return {
                "model_uri": "runs:/run-1/serving/model/model.joblib",
                "run_id": "run-1",
            }
        if uri.endswith("feature_column_order.json"):
            return {"columns": ["total_views", "category_id"]}
        if uri.endswith("categorical_encoding.json"):
            return {
                "category_maps": {"category_id": {"__MISSING__": 0, "__UNK__": 1}},
                "missing_token": "__MISSING__",
                "unknown_token": "__UNK__",
            }
        if uri.endswith("threshold.json"):
            return {"optimal_threshold": 0.42}
        if uri.endswith("prediction_contract.json"):
            return {
                "prediction_horizon_minutes": 10,
                "response_contract_version": "v1",
            }
        raise AssertionError(uri)

    def fake_download_artifacts(uri):
        loaded["download_artifacts"] = uri
        model_path = tmp_path / "model.joblib"
        model_path.write_text("unused")
        return str(model_path)

    def fake_load(path):
        loaded["joblib.load"] = path
        return SimpleNamespace(name="model")

    monkeypatch.setattr(
        "services.prediction_api.bundle.mlflow.artifacts.load_dict",
        fake_load_dict,
    )
    monkeypatch.setattr(
        "services.prediction_api.bundle.mlflow.artifacts.download_artifacts",
        fake_download_artifacts,
    )
    monkeypatch.setattr("services.prediction_api.bundle.joblib.load", fake_load)

    bundle = load_serving_bundle("runs:/run-1")

    assert bundle.model.name == "model"
    assert bundle.model_uri == "runs:/run-1/serving/model/model.joblib"
    assert bundle.model_version == "run-1"
    assert bundle.feature_column_order == ["total_views", "category_id"]
    assert bundle.category_maps == {"category_id": {"__MISSING__": 0, "__UNK__": 1}}
    assert bundle.missing_token == "__MISSING__"
    assert bundle.unknown_token == "__UNK__"
    assert bundle.threshold == 0.42
    assert bundle.prediction_horizon_minutes == 10
    assert bundle.response_contract_version == "v1"


def test_build_feature_row_matches_online_state_semantics():
    row = build_feature_row(FakeRedis(), "session-1", _bundle())

    assert row is not None
    assert row.iloc[0].to_dict() == {
        "total_views": 2,
        "total_carts": 1,
        "net_cart_count": 1,
        "cart_to_view_ratio": 0.5,
        "unique_categories": 2,
        "unique_products": 3,
        "session_duration_sec": 120.0,
        "price": 9.99,
        "category_id": "cat-id",
        "category_code": "__MISSING__",
        "brand": "__MISSING__",
        "event_type": "view",
    }
    assert str(row["category_id"].dtype) == "category"
    assert list(row["category_id"].cat.categories) == ["__MISSING__", "__UNK__", "cat-id"]


def test_build_feature_row_returns_none_when_serving_snapshot_is_incomplete():
    redis = FakeRedis()
    redis.hashes["session:session-1"].pop("serving_unique_products")

    row = build_feature_row(redis, "session-1", _bundle())

    assert row is None


def test_build_feature_row_returns_none_on_redis_miss():
    redis = FakeRedis()

    row = build_feature_row(redis, "missing-session", _bundle())

    assert row is None
