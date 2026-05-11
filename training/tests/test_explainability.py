import os

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from training.src.explainability import (
    deserialize_explainer,
    generate_shap_artifacts,
    serialize_explainer,
)


@pytest.fixture
def trained_model():
    """Fixture: simple trained random forest"""
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_shap_artifacts_structure(trained_model):
    """Test that SHAP artifacts include summary plot and explainer"""
    model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)
    
    assert "explainer" in artifacts
    assert "summary_plot_path" in artifacts
    assert "summary_values" in artifacts
    assert os.path.exists(artifacts["summary_plot_path"])
    assert os.path.getsize(artifacts["summary_plot_path"]) > 0


def test_shap_explainer_can_predict(trained_model):
    """Test that SHAP explainer produces values for samples"""
    model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)
    
    explainer = artifacts["explainer"]
    shap_values = explainer.shap_values(X[:5])
    
    # For RandomForest binary classification, shap_values is a 3D array
    # (n_samples, n_features, n_classes)
    assert isinstance(shap_values, np.ndarray)
    assert shap_values.shape == (5, 5, 2)  # 5 samples, 5 features, 2 classes


def test_summary_values_shape(trained_model):
    """Test that summary values have correct shape"""
    model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)
    
    summary = artifacts["summary_values"]
    assert isinstance(summary, np.ndarray)
    assert summary.ndim == 2


def test_shap_explainer_round_trip(trained_model, tmp_path):
    """Test SHAP explainer serialization round-trip."""
    model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)

    explainer = artifacts["explainer"]
    X_eval = X[:10]
    shap_values_before = explainer.shap_values(X_eval)

    explainer_path = tmp_path / "explainer.pkl"
    serialize_explainer(explainer, explainer_path)
    restored_explainer = deserialize_explainer(explainer_path)

    shap_values_after = restored_explainer.shap_values(X_eval)
    assert np.array(shap_values_after).shape == np.array(shap_values_before).shape
