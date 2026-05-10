import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from training.src.explainability import generate_shap_artifacts


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
