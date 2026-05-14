"""Training orchestration: CatBoost, LightGBM, XGBoost, Optuna, MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from xgboost import XGBClassifier

from training.src.categorical_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    CategoricalEncodingArtifacts,
    fit_categorical_encoders,
    prepare_training_frame,
    transform_with_categorical_contract,
)
from training.src.config import Config
from training.src.data_lineage import gather_lineage_metadata
from training.src.evaluate import compute_metrics
from training.src.model_validation import validate_model_gate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANDIDATE_MODEL_NAMES = ("catboost", "lightgbm", "xgboost")

MLFLOW_TRACKING_URI = Config.MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME = Config.MLFLOW_EXPERIMENT_NAME
OPTUNA_SMOKE_TRIALS = Config.OPTUNA_SMOKE_TRIALS
OPTUNA_TARGET_TRIALS = Config.OPTUNA_TARGET_TRIALS
MIN_VALIDATION_PR_AUC_THRESHOLD = Config.MIN_VALIDATION_PR_AUC_THRESHOLD


@dataclass(frozen=True)
class PreparedTrainingData:
    """Model-ready train/validation bundle."""

    train_features: pd.DataFrame
    train_target: pd.Series
    val_features: pd.DataFrame
    val_target: pd.Series
    categorical_columns: list[str]
    numeric_columns: list[str]
    categorical_artifacts: CategoricalEncodingArtifacts


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


def build_training_data(
    train_path: str,
    val_path: str,
    test_path: str,
) -> PreparedTrainingData:
    """Prepare categorical-aware train/validation frames from gold parquet files."""
    train_df, val_df, _ = load_gold_data(train_path, val_path, test_path)

    train_frame = prepare_training_frame(train_df)
    val_frame = prepare_training_frame(val_df)

    categorical_artifacts = fit_categorical_encoders(train_frame.features)
    train_features = transform_with_categorical_contract(
        train_frame.features, categorical_artifacts
    )
    val_features = transform_with_categorical_contract(
        val_frame.features, categorical_artifacts
    )

    return PreparedTrainingData(
        train_features=train_features,
        train_target=train_frame.target,
        val_features=val_features,
        val_target=val_frame.target,
        categorical_columns=list(CATEGORICAL_FEATURE_COLUMNS),
        numeric_columns=list(NUMERIC_FEATURE_COLUMNS),
        categorical_artifacts=categorical_artifacts,
    )


def _catboost_params(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 300),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        "loss_function": "Logloss",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
    }


def _lightgbm_params(trial: optuna.Trial) -> dict:
    use_unbalance = trial.suggest_categorical("is_unbalance", [True, False])
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "random_state": 42,
        "n_jobs": -1,
    }
    if use_unbalance:
        params["is_unbalance"] = True
    else:
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.5, 5.0)
    return params


def _xgboost_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        "random_state": 42,
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "aucpr",
        "n_jobs": -1,
    }


def train_catboost_candidate(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    val_features: pd.DataFrame,
    y_val: pd.Series,
    categorical_columns: list[str],
    n_trials: int = OPTUNA_TARGET_TRIALS,
) -> tuple[CatBoostClassifier, Dict[str, float]]:
    """Train CatBoost with Optuna hyperparameter search."""

    train_pool = Pool(train_features, label=y_train, cat_features=categorical_columns)
    val_pool = Pool(val_features, label=y_val, cat_features=categorical_columns)

    def objective(trial: optuna.Trial):
        model = CatBoostClassifier(**_catboost_params(trial))
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)
        return metrics["pr_auc"]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = _catboost_params(optuna.trial.FixedTrial(study.best_params))
    model = CatBoostClassifier(**best_params)
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    y_pred_proba = model.predict_proba(val_pool)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)
    return model, metrics


def train_lightgbm_candidate(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    val_features: pd.DataFrame,
    y_val: pd.Series,
    categorical_columns: list[str],
    n_trials: int = OPTUNA_TARGET_TRIALS,
) -> tuple[LGBMClassifier, Dict[str, float]]:
    """Train LightGBM with Optuna hyperparameter search."""

    def objective(trial: optuna.Trial):
        params = _lightgbm_params(trial)
        model = LGBMClassifier(**params)
        model.fit(
            train_features,
            y_train,
            categorical_feature=categorical_columns,
        )
        y_pred_proba = model.predict_proba(val_features)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)
        return metrics["pr_auc"]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = _lightgbm_params(optuna.trial.FixedTrial(study.best_params))
    model = LGBMClassifier(**best_params)
    model.fit(
        train_features,
        y_train,
        categorical_feature=categorical_columns,
    )

    y_pred_proba = model.predict_proba(val_features)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)
    return model, metrics


def train_xgboost_candidate(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    val_features: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = OPTUNA_TARGET_TRIALS,
) -> tuple[XGBClassifier, Dict[str, float]]:
    """Train XGBoost with Optuna hyperparameter search."""

    def objective(trial: optuna.Trial):
        params = _xgboost_params(trial)
        model = XGBClassifier(**params)
        model.fit(train_features, y_train, verbose=False)
        y_pred_proba = model.predict_proba(val_features)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)
        return metrics["pr_auc"]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = _xgboost_params(optuna.trial.FixedTrial(study.best_params))
    model = XGBClassifier(**best_params)
    model.fit(train_features, y_train, verbose=False)

    y_pred_proba = model.predict_proba(val_features)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)
    return model, metrics


def find_best_model_by_validation_pr_auc(results: Dict[str, Dict]) -> tuple[str, Dict]:
    """Find winner by highest validation PR-AUC."""
    best_name = max(results, key=lambda key: results[key]["metrics"]["pr_auc"])
    return best_name, results[best_name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.parquet")
    parser.add_argument("--val", required=True, help="Path to val.parquet")
    parser.add_argument("--test", required=True, help="Path to test.parquet")
    parser.add_argument(
        "--session-split-map",
        required=True,
        help="Path to session_split_map.parquet",
    )
    parser.add_argument("--smoke-mode", action="store_true", help="Use smoke budgets")

    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("Loading gold data...")
    data = build_training_data(args.train, args.val, args.test)
    lineage_metadata = gather_lineage_metadata(
        args.train,
        args.val,
        args.test,
        args.session_split_map,
    )

    n_trials = OPTUNA_SMOKE_TRIALS if args.smoke_mode else OPTUNA_TARGET_TRIALS
    logger.info(
        "Training 3 candidates (smoke=%s, trials=%d)...", args.smoke_mode, n_trials
    )

    results = {}

    with mlflow.start_run(run_name="catboost"):
        mlflow.log_dict(lineage_metadata, "lineage/metadata.json")
        catboost_model, catboost_metrics = train_catboost_candidate(
            data.train_features,
            data.train_target,
            data.val_features,
            data.val_target,
            data.categorical_columns,
            n_trials,
        )
        _log_metrics(catboost_metrics, "catboost_confusion_matrix.json")
        mlflow.sklearn.log_model(catboost_model, "model")
        results["catboost"] = {
            "model": catboost_model,
            "metrics": catboost_metrics,
        }

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_dict(lineage_metadata, "lineage/metadata.json")
        lgb_model, lgb_metrics = train_lightgbm_candidate(
            data.train_features,
            data.train_target,
            data.val_features,
            data.val_target,
            data.categorical_columns,
            n_trials,
        )
        _log_metrics(lgb_metrics, "lightgbm_confusion_matrix.json")
        mlflow.sklearn.log_model(lgb_model, "model")
        results["lightgbm"] = {"model": lgb_model, "metrics": lgb_metrics}

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_dict(lineage_metadata, "lineage/metadata.json")
        xgb_model, xgb_metrics = train_xgboost_candidate(
            data.train_features,
            data.train_target,
            data.val_features,
            data.val_target,
            n_trials,
        )
        _log_metrics(xgb_metrics, "xgboost_confusion_matrix.json")
        mlflow.sklearn.log_model(xgb_model, "model")
        results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics}

    winner_name, winner_data = find_best_model_by_validation_pr_auc(results)
    logger.info(
        "Winner: %s (PR-AUC: %.4f)",
        winner_name,
        winner_data["metrics"]["pr_auc"],
    )

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
    raise SystemExit(main())
