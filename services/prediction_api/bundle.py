"""Serving bundle loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import mlflow

SUPPORTED_RESPONSE_CONTRACT_VERSION = "v1"


@dataclass(frozen=True)
class ServingBundle:
    model: Any
    model_uri: str
    model_version: str
    feature_column_order: list[str]
    category_maps: dict[str, dict[str, int]]
    missing_token: str
    unknown_token: str
    threshold: float
    prediction_horizon_minutes: int
    response_contract_version: str


def validate_prediction_contract(contract: dict[str, Any]) -> None:
    version = contract.get("response_contract_version")
    if version != SUPPORTED_RESPONSE_CONTRACT_VERSION:
        raise ValueError(f"Unsupported response contract version: {version}")


def load_serving_bundle(run_uri: str) -> ServingBundle:
    model_metadata = mlflow.artifacts.load_dict(f"{run_uri}/serving/model_metadata.json")
    feature_column_order = mlflow.artifacts.load_dict(
        f"{run_uri}/serving/feature_column_order.json"
    )
    categorical_encoding = mlflow.artifacts.load_dict(
        f"{run_uri}/serving/categorical_encoding.json"
    )
    threshold = mlflow.artifacts.load_dict(f"{run_uri}/serving/threshold.json")
    prediction_contract = mlflow.artifacts.load_dict(
        f"{run_uri}/serving/prediction_contract.json"
    )

    validate_prediction_contract(prediction_contract)

    model_path = mlflow.artifacts.download_artifacts(model_metadata["model_uri"])
    model = joblib.load(model_path)

    return ServingBundle(
        model=model,
        model_uri=model_metadata["model_uri"],
        model_version=model_metadata["run_id"],
        feature_column_order=list(feature_column_order["columns"]),
        category_maps=categorical_encoding["category_maps"],
        missing_token=categorical_encoding["missing_token"],
        unknown_token=categorical_encoding["unknown_token"],
        threshold=float(threshold["optimal_threshold"]),
        prediction_horizon_minutes=int(prediction_contract["prediction_horizon_minutes"]),
        response_contract_version=prediction_contract["response_contract_version"],
    )

