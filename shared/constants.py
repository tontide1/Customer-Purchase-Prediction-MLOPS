"""
Shared constants for data pipeline.

This module defines canonical values for data layers, field names, event types,
and artifact names used across the entire training pipeline.
"""

# ============================================================================
# Data Layer Names
# ============================================================================
LAYER_RAW = "raw"
LAYER_BRONZE = "bronze"
LAYER_SILVER = "silver"
LAYER_GOLD = "gold"

# ============================================================================
# Artifact Names
# ============================================================================
ARTIFACT_EVENTS = "events.parquet"
ARTIFACT_SESSION_SPLIT_MAP = "session_split_map.parquet"
ARTIFACT_TRAIN_SNAPSHOTS = "train_snapshots.parquet"
ARTIFACT_VAL_SNAPSHOTS = "val_snapshots.parquet"
ARTIFACT_TEST_SNAPSHOTS = "test_snapshots.parquet"

# ============================================================================
# Timestamp Field Names (Contract-locked)
# ============================================================================
# Raw layer: original field from source
FIELD_EVENT_TIME = "event_time"

# Bronze/Silver/Gold layers: standardized internal field
FIELD_SOURCE_EVENT_TIME = "source_event_time"

# Replay and prediction timestamps (for future use)
FIELD_REPLAY_TIME = "replay_time"
FIELD_PREDICTION_TIME = "prediction_time"

# ============================================================================
# Event Type Constraints
# ============================================================================
ALLOWED_EVENT_TYPES = {
    "view",
    "cart",
    "remove_from_cart",
    "purchase",
}

# ============================================================================
# Required Fields (Must be present in every record)
# ============================================================================
REQUIRED_FIELDS = {
    FIELD_EVENT_TIME,  # Or FIELD_SOURCE_EVENT_TIME in downstream layers
    "event_type",
    "product_id",
    "user_id",
    "user_session",
}

# ============================================================================
# Nullable Fields (May be missing, but if present must be valid)
# ============================================================================
NULLABLE_FIELDS = {
    "category_code",
    "brand",
    "price",
}

# ============================================================================
# Default Values
# ============================================================================
DEFAULT_PRICE_THRESHOLD = 0.0  # price <= 0 is invalid
