"""Tests for the training orchestration module."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import optuna
import pytest

from training.src.train import (
    CANDIDATE_MODEL_NAMES,
    build_training_data,
    find_best_model_by_validation_pr_auc,
    main,
    _catboost_params,
    _lightgbm_params,
    _xgboost_params,
    train_catboost_candidate,
    train_lightgbm_candidate,
    train_xgboost_candidate,
)


@pytest.fixture
def gold_data(tmp_path):
    """Fixture: minimal gold data with categorical columns."""
    n_samples = 60
    rng = np.random.default_rng(42)
    total_views = rng.integers(1, 6, size=n_samples)
    total_carts = rng.integers(0, 3, size=n_samples)
    data = {
        "total_views": total_views,
        "total_carts": total_carts,
        "net_cart_count": total_carts - rng.integers(0, 2, size=n_samples),
        "cart_to_view_ratio": np.divide(
            total_carts,
            total_views,
            out=np.zeros_like(total_carts, dtype=float),
            where=total_views != 0,
        ),
        "unique_categories": rng.integers(1, 4, size=n_samples),
        "unique_products": rng.integers(1, 8, size=n_samples),
        "session_duration_sec": rng.uniform(5.0, 120.0, size=n_samples),
        "price": rng.uniform(10.0, 200.0, size=n_samples),
        "category_id": [f"cat-{i % 7}" for i in range(n_samples)],
        "category_code": [
            "appliances.environment.vacuum",
            "electronics.smartphone",
            "furniture.living_room.sofa",
        ]
        * (n_samples // 3)
        + ["appliances.environment.vacuum"] * (n_samples % 3),
        "brand": ["brand-a", "brand-b", "brand-c", "brand-d"] * (n_samples // 4)
        + ["brand-a"] * (n_samples % 4),
        "event_type": rng.choice(["view", "cart", "purchase"], size=n_samples).tolist(),
        "label": (total_views + total_carts > 4).astype(int),
    }

    df = pd.DataFrame(data)

    train_file = tmp_path / "train.parquet"
    val_file = tmp_path / "val.parquet"
    test_file = tmp_path / "test.parquet"
    split_map_file = tmp_path / "session_split_map.parquet"

    df.to_parquet(train_file)
    df.to_parquet(val_file)
    df.to_parquet(test_file)

    pd.DataFrame(
        {
            "user_session": ["session-1", "session-2"],
            "session_start_time": pd.to_datetime(
                ["2019-10-01 00:00:00", "2019-10-01 00:05:00"]
            ),
            "session_end_time": pd.to_datetime(
                ["2019-10-01 00:01:00", "2019-10-01 00:07:00"]
            ),
            "split": ["train", "val"],
        }
    ).to_parquet(split_map_file)

    return {
        "train_path": str(train_file),
        "val_path": str(val_file),
        "test_path": str(test_file),
        "split_map_path": str(split_map_file),
    }


class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMlflow:
    def __init__(self):
        self.sklearn = SimpleNamespace(log_model=lambda *args, **kwargs: None)

    def set_tracking_uri(self, *args, **kwargs):
        return None

    def set_experiment(self, *args, **kwargs):
        return None

    def start_run(self, *args, **kwargs):
        return _DummyRun()

    def log_dict(self, *args, **kwargs):
        return None

    def log_metrics(self, *args, **kwargs):
        return None

    def log_text(self, *args, **kwargs):
        return None


class _FakeModel:
    def predict_proba(self, *args, **kwargs):
        return np.array([[0.1, 0.9]])


def test_candidate_model_names():
    assert CANDIDATE_MODEL_NAMES == ("catboost", "lightgbm", "xgboost")


def test_build_training_data(gold_data):
    data = build_training_data(
        gold_data["train_path"],
        gold_data["val_path"],
        gold_data["test_path"],
    )

    assert data.train_features.shape[0] > 0
    assert data.val_features.shape[0] > 0
    assert data.train_target.shape[0] == data.train_features.shape[0]
    assert data.val_target.shape[0] == data.val_features.shape[0]
    assert data.categorical_columns == [
        "category_id", "category_code", "brand", "event_type",
    ]
    assert data.numeric_columns == [
        "total_views",
        "total_carts",
        "net_cart_count",
        "cart_to_view_ratio",
        "unique_categories",
        "unique_products",
        "session_duration_sec",
        "price",
    ]
    assert str(data.train_features["category_code"].dtype) == "category"
    assert str(data.train_features["brand"].dtype) == "category"
    assert str(data.train_features["event_type"].dtype) == "category"


def test_gpu_params_are_set_for_all_candidates():
    trial = optuna.trial.FixedTrial(
        {
            "iterations": 100,
            "depth": 4,
            "learning_rate": 0.1,
            "l2_leaf_reg": 1.0,
            "scale_pos_weight": 1.0,
            "is_unbalance": False,
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 50,
        }
    )

    cat_params = _catboost_params(trial, device="gpu", gpu_device_id="0")
    lgb_params = _lightgbm_params(trial, device="gpu", gpu_device_id="0")
    xgb_params = _xgboost_params(trial, device="gpu", gpu_device_id="0")

    assert cat_params["task_type"] == "GPU"
    assert cat_params["devices"] == "0"
    assert lgb_params["device_type"] == "gpu"
    assert lgb_params["gpu_device_id"] == 0
    assert xgb_params["device"] == "cuda:0"
    assert xgb_params["tree_method"] == "hist"


def test_train_three_models(gold_data):
    data = build_training_data(
        gold_data["train_path"],
        gold_data["val_path"],
        gold_data["test_path"],
    )

    cat_model, cat_metrics = train_catboost_candidate(
        data.train_features,
        data.train_target,
        data.val_features,
        data.val_target,
        data.categorical_columns,
        n_trials=1,
        device="cpu",
        gpu_device_id="0",
    )
    assert cat_model is not None
    assert "pr_auc" in cat_metrics

    lgb_model, lgb_metrics = train_lightgbm_candidate(
        data.train_features,
        data.train_target,
        data.val_features,
        data.val_target,
        data.categorical_columns,
        n_trials=1,
        device="cpu",
        gpu_device_id="0",
    )
    assert lgb_model is not None
    assert "pr_auc" in lgb_metrics

    xgb_model, xgb_metrics = train_xgboost_candidate(
        data.train_features,
        data.train_target,
        data.val_features,
        data.val_target,
        n_trials=1,
        device="cpu",
        gpu_device_id="0",
    )
    assert xgb_model is not None
    assert "pr_auc" in xgb_metrics


def test_find_best_model_by_validation_pr_auc():
    results = {
        "catboost": {"model": "cat", "metrics": {"pr_auc": 0.76}},
        "lightgbm": {"model": "lgb", "metrics": {"pr_auc": 0.75}},
        "xgboost": {"model": "xgb", "metrics": {"pr_auc": 0.74}},
    }

    winner_name, winner_data = find_best_model_by_validation_pr_auc(results)
    assert winner_name == "catboost"
    assert winner_data["metrics"]["pr_auc"] == 0.76


def test_main_trains_three_categorical_models(gold_data, monkeypatch):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("training.src.train.mlflow", fake_mlflow)
    monkeypatch.setattr("training.src.train.OPTUNA_SMOKE_TRIALS", 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.src.train",
            "--train",
            gold_data["train_path"],
            "--val",
            gold_data["val_path"],
            "--test",
            gold_data["test_path"],
            "--session-split-map",
            gold_data["split_map_path"],
            "--smoke-mode",
            "--device",
            "cpu",
            "--gpu-device-id",
            "0",
        ],
    )

    exit_code = main()
    assert exit_code == 0


def test_smoke_and_full_runs_pass_same_device_policy(gold_data, monkeypatch):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("training.src.train.mlflow", fake_mlflow)
    monkeypatch.setattr("training.src.train.OPTUNA_SMOKE_TRIALS", 1)
    monkeypatch.setattr("training.src.train.OPTUNA_TARGET_TRIALS", 1)

    recorded_devices: list[str] = []

    def fake_train(*args, **kwargs):
        recorded_devices.append(kwargs["device"])
        return _FakeModel(), {"pr_auc": 0.8, "confusion_matrix": np.array([[1, 0], [0, 1]])}

    monkeypatch.setattr("training.src.train.train_catboost_candidate", fake_train)
    monkeypatch.setattr("training.src.train.train_lightgbm_candidate", fake_train)
    monkeypatch.setattr("training.src.train.train_xgboost_candidate", fake_train)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.src.train",
            "--train",
            gold_data["train_path"],
            "--val",
            gold_data["val_path"],
            "--test",
            gold_data["test_path"],
            "--session-split-map",
            gold_data["split_map_path"],
            "--smoke-mode",
            "--device",
            "gpu",
            "--gpu-device-id",
            "0",
        ],
    )

    assert main() == 0
    assert recorded_devices == ["gpu", "gpu", "gpu"]


def test_auto_device_falls_back_to_cpu_when_gpu_fails(gold_data, monkeypatch):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("training.src.train.mlflow", fake_mlflow)
    monkeypatch.setattr("training.src.train.OPTUNA_SMOKE_TRIALS", 1)

    calls = {"cat": 0}

    def cat_train(*args, **kwargs):
        calls["cat"] += 1
        if calls["cat"] == 1:
            raise RuntimeError("GPU unavailable")
        return _FakeModel(), {"pr_auc": 0.8, "confusion_matrix": np.array([[1, 0], [0, 1]])}

    def ok_train(*args, **kwargs):
        return _FakeModel(), {"pr_auc": 0.8, "confusion_matrix": np.array([[1, 0], [0, 1]])}

    monkeypatch.setattr("training.src.train.train_catboost_candidate", cat_train)
    monkeypatch.setattr("training.src.train.train_lightgbm_candidate", ok_train)
    monkeypatch.setattr("training.src.train.train_xgboost_candidate", ok_train)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.src.train",
            "--train",
            gold_data["train_path"],
            "--val",
            gold_data["val_path"],
            "--test",
            gold_data["test_path"],
            "--session-split-map",
            gold_data["split_map_path"],
            "--smoke-mode",
            "--device",
            "auto",
            "--gpu-device-id",
            "0",
        ],
    )

    assert main() == 0
    assert calls["cat"] == 2
