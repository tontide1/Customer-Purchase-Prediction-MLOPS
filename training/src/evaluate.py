"""Model evaluation metrics: PR-AUC, F1, Precision, Recall, threshold selection."""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple[dict, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true: Binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for class 1

    Returns:
        Tuple of (metrics_dict, optimal_threshold) where metrics_dict contains:
        pr_auc, f1, precision, recall, confusion_matrix, optimal_threshold
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

    threshold = compute_optimal_threshold(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "pr_auc": float(pr_auc),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "optimal_threshold": float(threshold),
    }

    return metrics, threshold


def compute_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Select threshold that maximizes F1 score on precision-recall curve.

    Args:
        y_true: Binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        Optimal threshold value
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(
        y_true, y_pred_proba
    )

    # Compute F1 for each threshold (skip when precision+recall=0 to avoid /0)
    f1_scores = (
        2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    )

    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    else:
        return 0.5  # Fallback to 0.5 if threshold not found
