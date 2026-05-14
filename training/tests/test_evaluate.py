"""Tests for the model evaluation metrics module."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from training.src.evaluate import compute_metrics, compute_optimal_threshold


@pytest.fixture
def binary_predictions():
    """Fixture: y_true and y_pred for binary classification."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.75, 0.9, 0.25, 0.85])
    return y_true, y_pred


def test_compute_metrics_structure(binary_predictions):
    """Test that compute_metrics returns tuple: (metrics_dict, threshold)."""
    y_true, y_pred = binary_predictions
    metrics, threshold = compute_metrics(y_true, y_pred)

    required_keys = [
        "pr_auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "confusion_matrix",
        "optimal_threshold",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"

    assert isinstance(threshold, (float, np.floating)), (
        "Returned threshold should be float"
    )
    assert metrics["optimal_threshold"] == threshold, (
        "optimal_threshold in dict should match returned threshold"
    )


def test_pr_auc_in_valid_range(binary_predictions):
    """Test that PR-AUC is between 0 and 1."""
    y_true, y_pred = binary_predictions
    metrics, _ = compute_metrics(y_true, y_pred)

    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_compute_optimal_threshold(binary_predictions):
    """Test threshold selection from precision-recall curve."""
    y_true, y_pred = binary_predictions
    threshold = compute_optimal_threshold(y_true, y_pred)

    assert isinstance(threshold, (float, np.floating))
    assert 0.0 <= threshold <= 1.0


def test_confusion_matrix_format(binary_predictions):
    """Test that confusion matrix has correct structure."""
    y_true, y_pred = binary_predictions
    metrics, _ = compute_metrics(y_true, y_pred)
    cm = metrics["confusion_matrix"]

    assert cm.shape == (2, 2), "Confusion matrix should be 2x2"
    assert cm.sum() == len(y_true), "Confusion matrix sum should equal sample count"


def test_compute_metrics_includes_average_precision(binary_predictions):
    y_true, y_pred = binary_predictions
    metrics, threshold = compute_metrics(y_true, y_pred)

    assert "average_precision" in metrics
    assert metrics["average_precision"] == pytest.approx(
        average_precision_score(y_true, y_pred)
    )
    assert metrics["optimal_threshold"] == threshold
