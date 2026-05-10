"""
Shared schema definitions for data pipeline layers.

Defines PyArrow schemas for each data layer:
- raw: Input CSV schema with original field names
- bronze: Internal standardized schema after ingestion
- silver: Cleaned and deduplicated schema
"""

import pyarrow as pa
from . import constants

# ============================================================================
# Raw Layer Schema
# ============================================================================
# This is the input CSV schema from the dataset.
# Field names match the source system exactly (e.g., event_time, not source_event_time).

RAW_SCHEMA = pa.schema(
    [
        (constants.FIELD_EVENT_TIME, pa.timestamp("us")),
        ("event_type", pa.string()),
        ("product_id", pa.string()),
        (constants.FIELD_CATEGORY_ID, pa.string()),
        ("user_id", pa.string()),
        ("user_session", pa.string()),
        ("category_code", pa.string()),  # nullable
        ("brand", pa.string()),  # nullable
        ("price", pa.float64()),  # nullable
    ]
)

# ============================================================================
# Bronze Layer Schema
# ============================================================================
# After ingestion: event_time renamed to source_event_time, all fields preserved.
# This is the immutable, validated internal representation.

BRONZE_SCHEMA = pa.schema(
    [
        (constants.FIELD_SOURCE_EVENT_TIME, pa.timestamp("us")),
        ("event_type", pa.string()),
        ("product_id", pa.string()),
        (constants.FIELD_CATEGORY_ID, pa.string()),
        ("user_id", pa.string()),
        ("user_session", pa.string()),
        ("category_code", pa.string()),  # nullable
        ("brand", pa.string()),  # nullable
        ("price", pa.float64()),  # nullable
    ]
)

# ============================================================================
# Silver Layer Schema
# ============================================================================
# After cleaning: records with invalid values removed, deterministically sorted.
# Same field set as bronze, but rows filtered.

SILVER_SCHEMA = pa.schema(
    [
        (constants.FIELD_SOURCE_EVENT_TIME, pa.timestamp("us")),
        ("event_type", pa.string()),
        ("product_id", pa.string()),
        (constants.FIELD_CATEGORY_ID, pa.string()),
        ("user_id", pa.string()),
        ("user_session", pa.string()),
        ("category_code", pa.string()),  # nullable
        ("brand", pa.string()),  # nullable
        ("price", pa.float64()),  # nullable
    ]
)

# ============================================================================
# Gold Layer Schema
# ============================================================================

GOLD_SCHEMA = pa.schema(
    [
        (constants.FIELD_SOURCE_EVENT_TIME, pa.timestamp("us")),
        (constants.FIELD_CATEGORY_ID, pa.string()),
        ("user_session", pa.string()),
        ("user_id", pa.string()),
        ("event_type", pa.string()),
        ("product_id", pa.string()),
        ("category_code", pa.string()),
        ("brand", pa.string()),
        ("price", pa.float64()),
        ("total_views", pa.int64()),
        ("total_carts", pa.int64()),
        ("net_cart_count", pa.int64()),
        ("cart_to_view_ratio", pa.float64()),
        ("unique_categories", pa.int64()),
        ("unique_products", pa.int64()),
        ("session_duration_sec", pa.float64()),
        ("label", pa.int8()),
    ]
)


def get_raw_fields() -> set:
    """Return set of field names in raw schema."""
    return {field.name for field in RAW_SCHEMA}


def get_bronze_fields() -> set:
    """Return set of field names in bronze schema."""
    return {field.name for field in BRONZE_SCHEMA}


def get_silver_fields() -> set:
    """Return set of field names in silver schema."""
    return {field.name for field in SILVER_SCHEMA}


def get_gold_fields() -> set:
    """Return set of field names in gold schema."""
    return {field.name for field in GOLD_SCHEMA}
