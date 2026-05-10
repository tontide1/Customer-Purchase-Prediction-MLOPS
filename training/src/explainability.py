"""SHAP explainability artifacts for best model."""

import os
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import shap


def generate_shap_artifacts(
    model: Any,
    X_background: np.ndarray,
    X_test: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Generate SHAP artifacts for a tree model.
    
    Args:
        model: Trained tree model (XGBoost, LightGBM, RandomForest)
        X_background: Background data for SHAP explainer
        X_test: Optional test data for summary plot
    
    Returns:
        Dictionary with:
        - 'explainer': TreeExplainer object (can be pickled)
        - 'summary_values': SHAP values for background data
        - 'summary_plot_path': Path to matplotlib summary plot
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)
    
    # Handle different SHAP value formats:
    # - List of 2 arrays for binary classification
    # - 3D array (n_samples, n_features, 2) for RandomForest binary
    # Use class 1 (positive class)
    if isinstance(shap_values, list):
        summary_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D array: take class 1 (positive class)
        summary_values = shap_values[:, :, 1]
    else:
        summary_values = shap_values
    
    # Create summary plot
    temp_dir = tempfile.gettempdir()
    summary_plot_path = os.path.join(temp_dir, f"shap_summary_plot_{uuid.uuid4().hex}.png")
    
    # Generate and save the plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(summary_values, X_background, show=False)
    plt.savefig(summary_plot_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    artifacts = {
        "explainer": explainer,
        "summary_values": summary_values,
        "summary_plot_path": summary_plot_path,
        "model": model,
    }
    
    return artifacts


def serialize_explainer(explainer: shap.TreeExplainer, path: str | Path) -> None:
    """Pickle the SHAP explainer to disk for MLflow artifact storage."""
    with open(path, "wb") as f:
        pickle.dump(explainer, f)


def deserialize_explainer(path: str | Path) -> shap.TreeExplainer:
    """Load a pickled SHAP explainer from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
