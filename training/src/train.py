"""Training orchestration: three models, Optuna search, MLflow tracking."""

import argparse
import json
import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import mlflow
import optuna
from optuna.samplers import TPESampler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from training.src.evaluate import compute_metrics
from training.src.model_validation import validate_model_gate
from training.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = Config.MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME = Config.MLFLOW_EXPERIMENT_NAME
OPTUNA_SMOKE_TRIALS = Config.OPTUNA_SMOKE_TRIALS
OPTUNA_TARGET_TRIALS = Config.OPTUNA_TARGET_TRIALS
MIN_VALIDATION_PR_AUC_THRESHOLD = Config.MIN_VALIDATION_PR_AUC_THRESHOLD


def _log_metrics(metrics: dict, confusion_matrix_name: str) -> None:
    scalar_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.floating))
    }
    mlflow.log_metrics(scalar_metrics)
    mlflow.log_text(
        json.dumps(metrics["confusion_matrix"].tolist()),
        confusion_matrix_name,
    )


def load_gold_data(train_path: str, val_path: str, test_path: str):
    """Load gold parquet files."""
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    assert not train_df.empty, "Train data is empty"
    assert not val_df.empty, "Validation data is empty"
    assert not test_df.empty, "Test data is empty"

    return train_df, val_df, test_df


def build_train_matrix(
    train_path: str,
    val_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare training matrices.

    Returns:
        (X_train, y_train, X_val, y_val)
    """
    train_df, val_df, _ = load_gold_data(train_path, val_path, train_path)

    target_col = "target_purchase"
    feature_cols = [col for col in train_df.columns if col != target_col]

    feature_cols = [col for col in feature_cols if col not in
                   ["user_id", "user_session", "event_type", "product_id", "source_event_time"]]

    X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)

    X_val = val_df[feature_cols].fillna(0).values.astype(np.float32)
    y_val = val_df[target_col].values.astype(int)

    return X_train, y_train, X_val, y_val


def train_xgboost_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = OPTUNA_TARGET_TRIALS,
) -> Tuple[XGBClassifier, Dict[str, float]]:
    """Train XGBoost with Optuna hyperparameter search."""

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
            "random_state": 42,
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)

        return metrics["pr_auc"]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = 42

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)

    return model, metrics


def train_lightgbm_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = OPTUNA_TARGET_TRIALS,
) -> Tuple[LGBMClassifier, Dict[str, float]]:
    """Train LightGBM with Optuna hyperparameter search."""

    def objective(trial):
        use_unbalance = trial.suggest_categorical("is_unbalance", [True, False])
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "is_unbalance": use_unbalance,
            "random_state": 42,
        }
        if not use_unbalance:
            params["scale_pos_weight"] = trial.suggest_float(
                "scale_pos_weight", 0.5, 5.0
            )

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)

        return metrics["pr_auc"]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = 42

    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)

    return model, metrics


def train_random_forest_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train Random Forest (no hyperparameter search)."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)

    return model, metrics


def find_best_model_by_validation_pr_auc(results: Dict[str, Dict]) -> Tuple[str, Dict]:
    """Find winner by highest validation PR-AUC."""
    best_name = max(results, key=lambda k: results[k]["metrics"]["pr_auc"])
    return best_name, results[best_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.parquet")
    parser.add_argument("--val", required=True, help="Path to val.parquet")
    parser.add_argument("--test", required=True, help="Path to test.parquet")
    parser.add_argument("--session-split-map", required=True, help="Path to session_split_map.parquet")
    parser.add_argument("--smoke-mode", action="store_true", help="Use smoke budgets")

    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("Loading gold data...")
    X_train, y_train, X_val, y_val = build_train_matrix(args.train, args.val)

    n_trials = OPTUNA_SMOKE_TRIALS if args.smoke_mode else OPTUNA_TARGET_TRIALS

    logger.info(f"Training 3 candidates (smoke={args.smoke_mode}, trials={n_trials})...")

    results = {}

    # Train XGBoost
    with mlflow.start_run(run_name="xgboost"):
        xgb_model, xgb_metrics = train_xgboost_candidate(X_train, y_train, X_val, y_val, n_trials)
        _log_metrics(xgb_metrics, "xgboost_confusion_matrix.json")
        mlflow.sklearn.log_model(xgb_model, "model")
        results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics}

    # Train LightGBM
    with mlflow.start_run(run_name="lightgbm"):
        lgb_model, lgb_metrics = train_lightgbm_candidate(X_train, y_train, X_val, y_val, n_trials)
        _log_metrics(lgb_metrics, "lightgbm_confusion_matrix.json")
        mlflow.sklearn.log_model(lgb_model, "model")
        results["lightgbm"] = {"model": lgb_model, "metrics": lgb_metrics}

    # Train Random Forest
    with mlflow.start_run(run_name="random_forest"):
        rf_model, rf_metrics = train_random_forest_candidate(X_train, y_train, X_val, y_val)
        _log_metrics(rf_metrics, "random_forest_confusion_matrix.json")
        mlflow.sklearn.log_model(rf_model, "model")
        results["random_forest"] = {"model": rf_model, "metrics": rf_metrics}

    # Find winner
    winner_name, winner_data = find_best_model_by_validation_pr_auc(results)
    logger.info(f"Winner: {winner_name} (PR-AUC: {winner_data['metrics']['pr_auc']:.4f})")

    # Validation gate
    gate_pass = validate_model_gate(
        new_model_pr_auc=winner_data["metrics"]["pr_auc"],
        production_model_pr_auc=None,
        min_threshold=MIN_VALIDATION_PR_AUC_THRESHOLD,
    )

    if not gate_pass:
        logger.error("Model failed validation gate")
        return 1

    logger.info("Model passed validation gate")
    return 0


if __name__ == "__main__":
    exit(main())
