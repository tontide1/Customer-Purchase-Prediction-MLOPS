"""
Centralized configuration for training pipeline.

Manages paths, DVC/MinIO settings, and runtime parameters.
All paths and credentials are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration singleton for training pipeline."""

    # ========================================================================
    # Paths (relative to repo root)
    # ========================================================================
    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw")
    BRONZE_DATA_PATH = os.getenv("BRONZE_DATA_PATH", "data/bronze/events.parquet")
    SILVER_DATA_PATH = os.getenv("SILVER_DATA_PATH", "data/silver/events.parquet")
    GOLD_DATA_DIR = os.getenv("GOLD_DATA_DIR", "data/gold")

    # ========================================================================
    # Raw Window Selection
    # ========================================================================
    DATA_WINDOW_PROFILE = os.getenv("DATA_WINDOW_PROFILE", "dev_smoke")
    TRAINING_WINDOW_START = os.getenv("TRAINING_WINDOW_START", "2019-10")
    TRAINING_WINDOW_END = os.getenv("TRAINING_WINDOW_END", "2020-02")
    DEV_SMOKE_WINDOW_START = os.getenv("DEV_SMOKE_WINDOW_START", "2019-10")
    DEV_SMOKE_WINDOW_END = os.getenv("DEV_SMOKE_WINDOW_END", "2019-10")
    REPLAY_WINDOW_START = os.getenv("REPLAY_WINDOW_START", "2020-03")
    REPLAY_WINDOW_END = os.getenv("REPLAY_WINDOW_END", "2020-04")

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

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required paths exist and configuration is sound.

        Returns:
            True if configuration is valid, False otherwise.
        """
        # Check that raw data path exists
        if not Path(cls.RAW_DATA_PATH).exists():
            print(f"Warning: RAW_DATA_PATH does not exist: {cls.RAW_DATA_PATH}")
            return False

        # Ensure output directories can be created
        Path(cls.GOLD_DATA_DIR).parent.mkdir(parents=True, exist_ok=True)

        return True

    @classmethod
    def get_window_bounds(
        cls, profile: str | None = None
    ) -> tuple[str | None, str | None]:
        """Return inclusive raw window bounds for a profile."""
        selected_profile = (profile or cls.DATA_WINDOW_PROFILE).strip().lower()

        if selected_profile == "training":
            return cls.TRAINING_WINDOW_START, cls.TRAINING_WINDOW_END
        if selected_profile == "replay":
            return cls.REPLAY_WINDOW_START, cls.REPLAY_WINDOW_END
        if selected_profile == "dev_smoke":
            return cls.DEV_SMOKE_WINDOW_START, cls.DEV_SMOKE_WINDOW_END
        if selected_profile == "all":
            return None, None

        raise ValueError(
            f"Unknown DATA_WINDOW_PROFILE '{selected_profile}'. Expected training, replay, dev_smoke, or all."
        )

    @classmethod
    def get_all_settings(cls) -> dict:
        """Return all current settings as a dictionary (for debugging/logging)."""
        return {
            "raw_data_path": cls.RAW_DATA_PATH,
            "bronze_data_path": cls.BRONZE_DATA_PATH,
            "silver_data_path": cls.SILVER_DATA_PATH,
            "gold_data_dir": cls.GOLD_DATA_DIR,
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
        }


# Convenience singleton instance
config = Config()
