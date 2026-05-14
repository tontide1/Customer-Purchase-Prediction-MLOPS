"""SHAP explainability artifacts for the winner model."""

from __future__ import annotations

import os
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _extract_positive_class_shap_values(shap_values: Any) -> np.ndarray:
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        summary_values = np.asarray(shap_values[1])
    else:
        summary_values = np.asarray(shap_values)

    if summary_values.ndim == 3:
        summary_values = summary_values[:, :, 1]
    if summary_values.ndim == 1:
        summary_values = summary_values.reshape(-1, 1)

    return summary_values


def _prepare_plot_frame(frame: Any) -> Any:
    if not isinstance(frame, pd.DataFrame):
        return frame

    plot_frame = frame.copy()
    for column in plot_frame.columns:
        if not pd.api.types.is_numeric_dtype(plot_frame[column]):
            plot_frame[column] = plot_frame[column].astype("category")
    return plot_frame


def generate_shap_artifacts(
    model: Any,
    X_background: Any,
    X_test: Any | None = None,
) -> dict[str, Any]:
    """Generate SHAP artifacts for CatBoost, LightGBM, and XGBoost models."""

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)
    summary_values = _extract_positive_class_shap_values(shap_values)

    plot_frame = _prepare_plot_frame(X_test if X_test is not None else X_background)

    temp_dir = tempfile.gettempdir()
    summary_plot_path = os.path.join(
        temp_dir, f"shap_summary_plot_{uuid.uuid4().hex}.png"
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(summary_values, plot_frame, show=False)
    plt.savefig(summary_plot_path, bbox_inches="tight", dpi=100)
    plt.close()

    return {
        "explainer": explainer,
        "summary_values": summary_values,
        "summary_plot_path": summary_plot_path,
        "model": model,
    }


def serialize_explainer(explainer: shap.TreeExplainer, path: str | Path) -> None:
    """Pickle the SHAP explainer to disk for artifact storage."""

    with open(path, "wb") as f:
        pickle.dump(explainer, f)


def deserialize_explainer(path: str | Path) -> shap.TreeExplainer:
    """Load a pickled SHAP explainer from disk."""

    with open(path, "rb") as f:
        return pickle.load(f)
