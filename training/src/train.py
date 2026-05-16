"""Training orchestration: CatBoost, LightGBM, XGBoost, Optuna, MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import mlflow
import joblib
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
from training.src.explainability import generate_shap_artifacts, serialize_explainer
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
TRAIN_DEVICE = Config.TRAIN_DEVICE
GPU_DEVICE_ID = Config.GPU_DEVICE_ID
TEST_SAMPLE_SIZE = Config.TEST_SAMPLE_SIZE
RESPONSE_CONTRACT_VERSION = "v1"
SERVING_MODEL_ARTIFACT_PATH = "serving/model"
SERVING_MODEL_ARTIFACT_FILE = "model.joblib"
TRAIN_REPORT_PATH = Path("reports/train_metrics.json")


@dataclass(frozen=True)
class PreparedTrainingData:
    """Model-ready train/validation bundle."""

    train_features: pd.DataFrame
    train_target: pd.Series
    val_features: pd.DataFrame
    val_target: pd.Series
    test_features: pd.DataFrame
    test_target: pd.Series
    categorical_columns: list[str]
    numeric_columns: list[str]
    categorical_artifacts: CategoricalEncodingArtifacts


def _log_metrics(
    metrics: dict,
    confusion_matrix_name: str,
    metric_prefix: str = "",
) -> None:
    scalar_metrics = {
        f"{metric_prefix}{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.floating))
    }
    mlflow.log_metrics(scalar_metrics)
    mlflow.log_text(
        json.dumps(metrics["confusion_matrix"].tolist()),
        confusion_matrix_name,
    )


def _log_model_artifact(candidate_name: str, model) -> None:
    if candidate_name == "catboost":
        mlflow.catboost.log_model(model, "model")
        return

    mlflow.sklearn.log_model(model, "model")


def _log_serving_model_artifact(model) -> str:
    with tempfile.TemporaryDirectory(prefix="winner-serving-model-") as tmp_dir:
        model_path = Path(tmp_dir) / SERVING_MODEL_ARTIFACT_FILE
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path=SERVING_MODEL_ARTIFACT_PATH)

    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run is not None else ""
    return f"runs:/{run_id}/{SERVING_MODEL_ARTIFACT_PATH}/{SERVING_MODEL_ARTIFACT_FILE}"


def _prepare_shap_sample(test_features: pd.DataFrame) -> pd.DataFrame:
    if len(test_features) <= TEST_SAMPLE_SIZE:
        return test_features
    return test_features.sample(n=TEST_SAMPLE_SIZE, random_state=42)


def _log_winner_shap_artifacts(model, test_features: pd.DataFrame) -> None:
    shap_features = _prepare_shap_sample(test_features)
    shap_artifacts = generate_shap_artifacts(
        model,
        shap_features,
        X_test=shap_features,
    )
    mlflow.log_artifact(shap_artifacts["summary_plot_path"], artifact_path="shap")

    with tempfile.TemporaryDirectory(prefix="winner-shap-") as temp_dir:
        explainer_path = Path(temp_dir) / "explainer.pkl"
        serialize_explainer(shap_artifacts["explainer"], explainer_path)
        mlflow.log_artifact(str(explainer_path), artifact_path="shap")


def _log_serving_bundle(
    *,
    winner_name: str,
    winner_model,
    data: PreparedTrainingData,
    winner_threshold: float,
) -> None:
    model_uri = _log_serving_model_artifact(winner_model)
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run is not None else ""
    mlflow.log_dict(
        {
            "model_type": winner_name,
            "model_name": winner_name,
            "run_id": run_id,
            "model_uri": model_uri,
            "artifact_path": SERVING_MODEL_ARTIFACT_PATH,
            "artifact_file": SERVING_MODEL_ARTIFACT_FILE,
            "load_flavor": "joblib",
            "probability_method": "predict_proba",
        },
        "serving/model_metadata.json",
    )
    mlflow.log_dict(
        {"columns": data.numeric_columns + data.categorical_columns},
        "serving/feature_column_order.json",
    )
    mlflow.log_dict(
        {
            "category_maps": data.categorical_artifacts.category_maps,
            "missing_token": data.categorical_artifacts.missing_token,
            "unknown_token": data.categorical_artifacts.unknown_token,
        },
        "serving/categorical_encoding.json",
    )
    mlflow.log_dict(
        {"optimal_threshold": float(winner_threshold)},
        "serving/threshold.json",
    )
    mlflow.log_dict(
        {
            "prediction_horizon_minutes": Config.PREDICTION_HORIZON_MINUTES,
            "response_contract_version": RESPONSE_CONTRACT_VERSION,
        },
        "serving/prediction_contract.json",
    )


def _write_train_report(
    *,
    winner_name: str,
    winner_metrics: dict,
    test_metrics: dict,
    gate_pass: bool,
) -> None:
    report_path = TRAIN_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "winner_name": winner_name,
        "winner_validation_pr_auc": float(winner_metrics["pr_auc"]),
        "winner_validation_average_precision": float(
            winner_metrics["average_precision"]
        ),
        "winner_validation_threshold": float(
            winner_metrics.get("optimal_threshold", 0.5)
        ),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_average_precision": float(test_metrics["average_precision"]),
        "validation_gate_passed": bool(gate_pass),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def _resolve_training_device(
    device: str | None, gpu_device_id: str | None
) -> tuple[str, str]:
    """Resolve runtime device settings to concrete values."""

    return (
        TRAIN_DEVICE if device is None else device,
        GPU_DEVICE_ID if gpu_device_id is None else str(gpu_device_id),
    )


def _is_gpu_training_error(exc: Exception) -> bool:
    """Return True when an exception looks like a GPU/device failure."""

    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "cuda",
            "cudnn",
            "cublas",
            "device-side assert",
            "device ordinal",
            "no gpus are available",
            "gpu",
            "hip error",
            "out of memory",
        )
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
    """Prepare categorical-aware train/validation/test frames from gold parquet files."""
    train_df, val_df, test_df = load_gold_data(train_path, val_path, test_path)

    train_frame = prepare_training_frame(train_df)
    val_frame = prepare_training_frame(val_df)
    test_frame = prepare_training_frame(test_df)

    categorical_artifacts = fit_categorical_encoders(train_frame.features)
    train_features = transform_with_categorical_contract(
        train_frame.features, categorical_artifacts
    )
    val_features = transform_with_categorical_contract(
        val_frame.features, categorical_artifacts
    )
    test_features = transform_with_categorical_contract(
        test_frame.features, categorical_artifacts
    )

    return PreparedTrainingData(
        train_features=train_features,
        train_target=train_frame.target,
        val_features=val_features,
        val_target=val_frame.target,
        test_features=test_features,
        test_target=test_frame.target,
        categorical_columns=list(CATEGORICAL_FEATURE_COLUMNS),
        numeric_columns=list(NUMERIC_FEATURE_COLUMNS),
        categorical_artifacts=categorical_artifacts,
    )


def _catboost_params(trial: optuna.Trial, device: str, gpu_device_id: str) -> dict:
    params = {
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
    if device in {"auto", "gpu"}:
        params["task_type"] = "GPU"
        params["devices"] = gpu_device_id
    return params


def _lightgbm_params(trial: optuna.Trial, device: str, gpu_device_id: str) -> dict:
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
    if device in {"auto", "gpu"}:
        params["device_type"] = "gpu"
        params["gpu_device_id"] = int(gpu_device_id)
    else:
        params["device_type"] = "cpu"
    return params


def _xgboost_params(trial: optuna.Trial, device: str, gpu_device_id: str) -> dict:
    params = {
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
    if device in {"auto", "gpu"}:
        params["device"] = f"cuda:{gpu_device_id}"
    else:
        params["device"] = "cpu"
    return params


def _train_candidate_with_device_policy(
    candidate_name: str,
    train_fn,
    *,
    device: str,
    gpu_device_id: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    val_features: pd.DataFrame,
    val_target: pd.Series,
    n_trials: int,
    categorical_columns: list[str] | None = None,
):
    """Run a candidate with the selected device policy."""

    def run(resolved_device: str):
        call_kwargs = {
            "train_features": train_features,
            "y_train": train_target,
            "val_features": val_features,
            "y_val": val_target,
            "n_trials": n_trials,
            "device": resolved_device,
            "gpu_device_id": gpu_device_id,
        }
        if categorical_columns is not None:
            call_kwargs["categorical_columns"] = categorical_columns
        return train_fn(**call_kwargs)

    if device != "auto":
        return run(device)

    try:
        return run("gpu")
    except Exception as exc:
        if not _is_gpu_training_error(exc):
            raise
        logger.warning(
            "%s GPU training failed with %s: %s, retrying on CPU",
            candidate_name,
            type(exc).__name__,
            exc,
        )
        return run("cpu")


def _lightgbm_device_for_policy(device: str) -> str:
    """Keep LightGBM off GPU because high-cardinality categorical bins exceed GPU limits."""

    if device in {"auto", "gpu"}:
        return "cpu"
    return device


def train_catboost_candidate(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    val_features: pd.DataFrame,
    y_val: pd.Series,
    categorical_columns: list[str],
    n_trials: int = OPTUNA_TARGET_TRIALS,
    device: str | None = None,
    gpu_device_id: str | None = None,
) -> tuple[CatBoostClassifier, Dict[str, float]]:
    """Train CatBoost with Optuna hyperparameter search."""

    device, gpu_device_id = _resolve_training_device(device, gpu_device_id)
    train_pool = Pool(train_features, label=y_train, cat_features=categorical_columns)
    val_pool = Pool(val_features, label=y_val, cat_features=categorical_columns)

    def objective(trial: optuna.Trial):
        model = CatBoostClassifier(**_catboost_params(trial, device, gpu_device_id))
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)
        return metrics["pr_auc"]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = _catboost_params(
        optuna.trial.FixedTrial(study.best_params), device, gpu_device_id
    )
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
    device: str | None = None,
    gpu_device_id: str | None = None,
) -> tuple[LGBMClassifier, Dict[str, float]]:
    """Train LightGBM with Optuna hyperparameter search."""

    device, gpu_device_id = _resolve_training_device(device, gpu_device_id)

    def objective(trial: optuna.Trial):
        params = _lightgbm_params(trial, device, gpu_device_id)
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

    best_params = _lightgbm_params(
        optuna.trial.FixedTrial(study.best_params), device, gpu_device_id
    )
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
    device: str | None = None,
    gpu_device_id: str | None = None,
) -> tuple[XGBClassifier, Dict[str, float]]:
    """Train XGBoost with Optuna hyperparameter search."""

    device, gpu_device_id = _resolve_training_device(device, gpu_device_id)

    def objective(trial: optuna.Trial):
        params = _xgboost_params(trial, device, gpu_device_id)
        model = XGBClassifier(**params)
        model.fit(train_features, y_train, verbose=False)
        y_pred_proba = model.predict_proba(val_features)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)
        return metrics["pr_auc"]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = _xgboost_params(
        optuna.trial.FixedTrial(study.best_params), device, gpu_device_id
    )
    model = XGBClassifier(**best_params)
    model.fit(train_features, y_train, verbose=False)

    y_pred_proba = model.predict_proba(val_features)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)
    return model, metrics


def find_best_model_by_validation_pr_auc(results: Dict[str, Dict]) -> tuple[str, Dict]:
    """Find winner by highest validation PR-AUC."""
    best_name = max(results, key=lambda key: results[key]["metrics"]["pr_auc"])
    return best_name, results[best_name]


def evaluate_winner_on_test(
    model,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    threshold: float,
) -> dict:
    """Evaluate the selected winner model on the held-out test set."""
    y_pred_proba = model.predict_proba(test_features)[:, 1]
    metrics, _ = compute_metrics(test_target, y_pred_proba, threshold=threshold)
    return metrics


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
    parser.add_argument(
        "--device",
        default=Config.TRAIN_DEVICE,
        choices=["auto", "cpu", "gpu"],
        help="Training device policy",
    )
    parser.add_argument(
        "--gpu-device-id",
        default=Config.GPU_DEVICE_ID,
        help="GPU device ordinal to use when GPU is enabled",
    )

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
    logger.info("Device policy: %s (gpu_device_id=%s)", args.device, args.gpu_device_id)

    results = {}

    with mlflow.start_run(run_name="catboost"):
        mlflow.log_dict(lineage_metadata, "lineage/metadata.json")
        catboost_model, catboost_metrics = _train_candidate_with_device_policy(
            "catboost",
            train_catboost_candidate,
            device=args.device,
            gpu_device_id=args.gpu_device_id,
            train_features=data.train_features,
            train_target=data.train_target,
            val_features=data.val_features,
            val_target=data.val_target,
            n_trials=n_trials,
            categorical_columns=data.categorical_columns,
        )
        _log_metrics(catboost_metrics, "catboost_confusion_matrix.json")
        _log_model_artifact("catboost", catboost_model)
        results["catboost"] = {
            "model": catboost_model,
            "metrics": catboost_metrics,
        }

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_dict(lineage_metadata, "lineage/metadata.json")
        lightgbm_device = _lightgbm_device_for_policy(args.device)
        if lightgbm_device != args.device:
            logger.info(
                "LightGBM device policy resolved to %s to avoid GPU categorical-bin limits",
                lightgbm_device,
            )
        lgb_model, lgb_metrics = _train_candidate_with_device_policy(
            "lightgbm",
            train_lightgbm_candidate,
            device=lightgbm_device,
            gpu_device_id=args.gpu_device_id,
            train_features=data.train_features,
            train_target=data.train_target,
            val_features=data.val_features,
            val_target=data.val_target,
            n_trials=n_trials,
            categorical_columns=data.categorical_columns,
        )
        _log_metrics(lgb_metrics, "lightgbm_confusion_matrix.json")
        _log_model_artifact("lightgbm", lgb_model)
        results["lightgbm"] = {"model": lgb_model, "metrics": lgb_metrics}

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_dict(lineage_metadata, "lineage/metadata.json")
        xgb_model, xgb_metrics = _train_candidate_with_device_policy(
            "xgboost",
            train_xgboost_candidate,
            device=args.device,
            gpu_device_id=args.gpu_device_id,
            train_features=data.train_features,
            train_target=data.train_target,
            val_features=data.val_features,
            val_target=data.val_target,
            n_trials=n_trials,
        )
        _log_metrics(xgb_metrics, "xgboost_confusion_matrix.json")
        _log_model_artifact("xgboost", xgb_model)
        results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics}

    winner_name, winner_data = find_best_model_by_validation_pr_auc(results)
    winner_threshold = winner_data["metrics"].get("optimal_threshold", 0.5)
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

    with mlflow.start_run(run_name=f"{winner_name}_test_evaluation"):
        mlflow.log_dict(lineage_metadata, "lineage/metadata.json")
        mlflow.log_metrics({"validation_gate_passed": float(gate_pass)})
        test_metrics = evaluate_winner_on_test(
            winner_data["model"],
            data.test_features,
            data.test_target,
            threshold=winner_threshold,
        )
        _log_metrics(
            test_metrics,
            f"{winner_name}_test_confusion_matrix.json",
            metric_prefix="test_",
        )
        _log_winner_shap_artifacts(winner_data["model"], data.test_features)
        _log_serving_bundle(
            winner_name=winner_name,
            winner_model=winner_data["model"],
            data=data,
            winner_threshold=winner_threshold,
        )
    logger.info(
        "Test results for %s: PR-AUC=%.4f, average_precision=%.4f",
        winner_name,
        test_metrics["pr_auc"],
        test_metrics["average_precision"],
    )

    _write_train_report(
        winner_name=winner_name,
        winner_metrics=winner_data["metrics"],
        test_metrics=test_metrics,
        gate_pass=gate_pass,
    )
    logger.info("Model passed validation gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
