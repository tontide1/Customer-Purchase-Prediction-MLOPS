"""
Centralized configuration for training pipeline.

Manages paths, DVC/MinIO settings, and runtime parameters.
All paths and credentials are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path


class Config:
    """Configuration singleton for training pipeline."""

    # ========================================================================
    # Paths (relative to repo root)
    # ========================================================================
    TRAIN_RAW_DATA_PATH = os.getenv("TRAIN_RAW_DATA_PATH", "data/train_raw")
    SIMULATION_RAW_DATA_PATH = os.getenv(
        "SIMULATION_RAW_DATA_PATH", "data/simulation_raw/2019-Nov.csv.gz"
    )
    RETRAIN_RAW_DATA_DIR = os.getenv("RETRAIN_RAW_DATA_DIR", "data/retrain_raw")
    RETRAIN_DATA_DIR = os.getenv("RETRAIN_DATA_DIR", "data/retrain")
    BRONZE_DATA_PATH = os.getenv("BRONZE_DATA_PATH", "data/bronze")
    SILVER_DATA_PATH = os.getenv("SILVER_DATA_PATH", "data/silver")
    GOLD_DATA_DIR = os.getenv("GOLD_DATA_DIR", "data/gold")
    RETRAIN_WINDOW_DAYS = int(os.getenv("RETRAIN_WINDOW_DAYS", "7"))

    # ========================================================================
    # Raw Data Window Selection
    # ========================================================================
    # Baseline training is intentionally scoped to 2019-Oct. Online replay uses
    # 2019-Nov and must be persisted/exported before it is used for retraining.
    DATA_WINDOW_PROFILE = os.getenv("DATA_WINDOW_PROFILE", "training")
    TRAINING_WINDOW_START = os.getenv("TRAINING_WINDOW_START", "2019-10")
    TRAINING_WINDOW_END = os.getenv("TRAINING_WINDOW_END", "2019-10")
    DEV_SMOKE_WINDOW_START = os.getenv("DEV_SMOKE_WINDOW_START", "2019-10")
    DEV_SMOKE_WINDOW_END = os.getenv("DEV_SMOKE_WINDOW_END", "2019-10")
    REPLAY_WINDOW_START = os.getenv("REPLAY_WINDOW_START", "2019-11")
    REPLAY_WINDOW_END = os.getenv("REPLAY_WINDOW_END", "2019-11")

    # ========================================================================
    # Prediction Contract
    # ========================================================================
    # Locked for Week 1: 10-minute window for next event prediction
    PREDICTION_HORIZON_MINUTES = int(os.getenv("PREDICTION_HORIZON_MINUTES", "10"))

    # ========================================================================
    # DVC Remote Configuration
    # ========================================================================
    DVC_REMOTE_NAME = os.getenv("DVC_REMOTE_NAME", "minio-local")
    DVC_REMOTE_URL = os.getenv("DVC_REMOTE_URL", "s3://mlops-artifacts/dvc")

    # ========================================================================
    # MinIO / S3 Credentials (for local demo)
    # ========================================================================
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "mlops-artifacts")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    # ========================================================================
    # DVC Configuration (read-only, for reference)
    # ========================================================================
    # These are typically set via `dvc remote add` and stored in .dvc/config
    # This class provides defaults for initialization only.

    # ========================================================================
    # MLflow Configuration (Sprint 2b)
    # ========================================================================
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sprint2b_training")
    MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "models")

    # ========================================================================
    # Optuna Configuration (Sprint 2b)
    # ========================================================================
    OPTUNA_SMOKE_TRIALS = int(os.getenv("OPTUNA_SMOKE_TRIALS", "3"))
    OPTUNA_TARGET_TRIALS = int(os.getenv("OPTUNA_TARGET_TRIALS", "50"))
    OPTUNA_TIMEOUT_SECONDS = int(os.getenv("OPTUNA_TIMEOUT_SECONDS", "3600"))

    # ========================================================================
    # Training Configuration (Sprint 2b)
    # ========================================================================
    MIN_VALIDATION_PR_AUC_THRESHOLD = float(os.getenv("MIN_VALIDATION_PR_AUC_THRESHOLD", "0.5"))
    TEST_SAMPLE_SIZE = int(os.getenv("TEST_SAMPLE_SIZE", "500"))
    SMOKE_MODE_ENABLED = os.getenv("SMOKE_MODE_ENABLED", "true").lower() == "true"

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required paths exist and configuration is sound.

        Returns:
            True if configuration is valid, False otherwise.
        """
        # Check that baseline training raw path exists
        if not Path(cls.TRAIN_RAW_DATA_PATH).exists():
            print(
                "Warning: TRAIN_RAW_DATA_PATH does not exist: "
                f"{cls.TRAIN_RAW_DATA_PATH}"
            )
            return False

        # Ensure output directories can be created
        Path(cls.GOLD_DATA_DIR).parent.mkdir(parents=True, exist_ok=True)

        return True

    @classmethod
    def get_all_settings(cls) -> dict:
        """Return all current settings as a dictionary (for debugging/logging)."""
        return {
            "train_raw_data_path": cls.TRAIN_RAW_DATA_PATH,
            "simulation_raw_data_path": cls.SIMULATION_RAW_DATA_PATH,
            "retrain_raw_data_dir": cls.RETRAIN_RAW_DATA_DIR,
            "retrain_data_dir": cls.RETRAIN_DATA_DIR,
            "bronze_data_path": cls.BRONZE_DATA_PATH,
            "silver_data_path": cls.SILVER_DATA_PATH,
            "gold_data_dir": cls.GOLD_DATA_DIR,
            "retrain_window_days": cls.RETRAIN_WINDOW_DAYS,
            "data_window_profile": cls.DATA_WINDOW_PROFILE,
            "training_window_start": cls.TRAINING_WINDOW_START,
            "training_window_end": cls.TRAINING_WINDOW_END,
            "dev_smoke_window_start": cls.DEV_SMOKE_WINDOW_START,
            "dev_smoke_window_end": cls.DEV_SMOKE_WINDOW_END,
            "replay_window_start": cls.REPLAY_WINDOW_START,
            "replay_window_end": cls.REPLAY_WINDOW_END,
            "prediction_horizon_minutes": cls.PREDICTION_HORIZON_MINUTES,
            "dvc_remote_name": cls.DVC_REMOTE_NAME,
            "dvc_remote_url": cls.DVC_REMOTE_URL,
            "minio_endpoint": cls.MINIO_ENDPOINT,
            "minio_bucket": cls.MINIO_BUCKET,
            "mlflow_tracking_uri": cls.MLFLOW_TRACKING_URI,
            "mlflow_experiment_name": cls.MLFLOW_EXPERIMENT_NAME,
            "mlflow_artifact_path": cls.MLFLOW_ARTIFACT_PATH,
            "optuna_smoke_trials": cls.OPTUNA_SMOKE_TRIALS,
            "optuna_target_trials": cls.OPTUNA_TARGET_TRIALS,
            "optuna_timeout_seconds": cls.OPTUNA_TIMEOUT_SECONDS,
            "min_validation_pr_auc_threshold": cls.MIN_VALIDATION_PR_AUC_THRESHOLD,
            "smoke_mode_enabled": cls.SMOKE_MODE_ENABLED,
        }


# Convenience singleton instance
config = Config()
