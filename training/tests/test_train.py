"""Tests for the training orchestration module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from training.src.train import (
    build_train_matrix,
    train_xgboost_candidate,
    train_lightgbm_candidate,
    train_random_forest_candidate,
    find_best_model_by_validation_pr_auc,
)


@pytest.fixture
def gold_data(tmp_path):
    """Fixture: minimal gold data (train, val, test)"""
    n_samples = 100
    data = {
        "user_id": ["user_" + str(i % 10) for i in range(n_samples)],
        "user_session": ["session_" + str(i % 5) for i in range(n_samples)],
        "source_event_time": pd.date_range("2019-10-01", periods=n_samples),
        "event_type": ["view", "click", "purchase", "add_to_cart"] * (n_samples // 4),
        "product_id": ["prod_" + str(i % 20) for i in range(n_samples)],
        "price": np.random.rand(n_samples) * 100,
        "label": np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)

    train_file = tmp_path / "train.parquet"
    val_file = tmp_path / "val.parquet"
    test_file = tmp_path / "test.parquet"

    df.to_parquet(train_file)
    df.to_parquet(val_file)
    df.to_parquet(test_file)

    return {
        "train_path": str(train_file),
        "val_path": str(val_file),
        "test_path": str(test_file),
    }


def test_build_train_matrix(gold_data):
    """Test that training matrix is built correctly"""
    X_train, y_train, X_val, y_val = build_train_matrix(
        gold_data["train_path"],
        gold_data["val_path"],
    )

    assert X_train.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert X_val.shape[1] == X_train.shape[1]


def test_train_three_models(gold_data):
    """Test that all three models can be trained"""
    X_train, y_train, X_val, y_val = build_train_matrix(
        gold_data["train_path"],
        gold_data["val_path"],
    )

    with patch("training.src.train.mlflow"):
        xgb_model, xgb_metrics = train_xgboost_candidate(X_train, y_train, X_val, y_val, n_trials=2)
        assert xgb_model is not None
        assert "pr_auc" in xgb_metrics

        lgb_model, lgb_metrics = train_lightgbm_candidate(X_train, y_train, X_val, y_val, n_trials=2)
        assert lgb_model is not None
        assert "pr_auc" in lgb_metrics

        rf_model, rf_metrics = train_random_forest_candidate(X_train, y_train, X_val, y_val)
        assert rf_model is not None
        assert "pr_auc" in rf_metrics


def test_find_best_model_by_validation_pr_auc():
    """Test winner selection by validation PR-AUC"""
    results = {
        "xgboost": {"model": "xgb", "metrics": {"pr_auc": 0.72}},
        "lightgbm": {"model": "lgb", "metrics": {"pr_auc": 0.75}},
        "random_forest": {"model": "rf", "metrics": {"pr_auc": 0.68}},
    }

    winner_name, winner_data = find_best_model_by_validation_pr_auc(results)
    assert winner_name == "lightgbm"
    assert winner_data["metrics"]["pr_auc"] == 0.75
