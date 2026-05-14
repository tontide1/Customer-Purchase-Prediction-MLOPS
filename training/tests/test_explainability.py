import os

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from training.src.explainability import (
    deserialize_explainer,
    generate_shap_artifacts,
    serialize_explainer,
)


def _build_training_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "f1": rng.random(80),
            "f2": rng.random(80),
            "f3": rng.random(80),
            "f4": rng.random(80),
            "f5": rng.random(80),
        }
    )
    y = (X["f1"] + X["f2"] > 1).astype(int)
    return X, y


@pytest.fixture(params=["catboost", "lightgbm", "xgboost"])
def trained_model(request):
    X, y = _build_training_data()

    if request.param == "catboost":
        model = CatBoostClassifier(
            iterations=10,
            depth=4,
            learning_rate=0.2,
            loss_function="Logloss",
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(X, y)
    elif request.param == "lightgbm":
        model = LGBMClassifier(
            n_estimators=25,
            max_depth=4,
            learning_rate=0.2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
    else:
        model = XGBClassifier(
            n_estimators=25,
            max_depth=4,
            learning_rate=0.2,
            random_state=42,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="aucpr",
            n_jobs=-1,
        )
        model.fit(X, y, verbose=False)

    return request.param, model, X, y


def test_shap_artifacts_structure(trained_model):
    """Test that SHAP artifacts include summary plot and explainer."""
    model_name, model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)

    assert "explainer" in artifacts
    assert "summary_plot_path" in artifacts
    assert "summary_values" in artifacts
    assert os.path.exists(artifacts["summary_plot_path"])
    assert os.path.getsize(artifacts["summary_plot_path"]) > 0


def test_shap_explainer_can_predict(trained_model):
    """Test that SHAP explainer produces values for samples."""
    model_name, model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)

    explainer = artifacts["explainer"]
    shap_values = explainer.shap_values(X[:5])

    assert shap_values is not None


def test_summary_values_shape(trained_model):
    """Test that summary values have correct shape."""
    model_name, model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)

    summary = artifacts["summary_values"]
    assert isinstance(summary, np.ndarray)
    assert summary.ndim == 2


def test_shap_explainer_round_trip(trained_model, tmp_path):
    """Test SHAP explainer serialization round-trip."""
    model_name, model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)

    explainer = artifacts["explainer"]
    X_eval = X[:10]
    shap_values_before = explainer.shap_values(X_eval)

    explainer_path = tmp_path / "explainer.pkl"
    serialize_explainer(explainer, explainer_path)
    restored_explainer = deserialize_explainer(explainer_path)

    shap_values_after = restored_explainer.shap_values(X_eval)
    assert np.array(shap_values_after).shape == np.array(shap_values_before).shape
