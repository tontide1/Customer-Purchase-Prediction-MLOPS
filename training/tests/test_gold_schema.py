"""Tests for the Sprint 2a gold schema contract."""

import pyarrow as pa

from shared import schemas


def test_gold_schema_matches_mvp_contract() -> None:
    expected_names = [
        "source_event_time",
        "category_id",
        "user_session",
        "user_id",
        "event_type",
        "product_id",
        "category_code",
        "brand",
        "price",
        "total_views",
        "total_carts",
        "net_cart_count",
        "cart_to_view_ratio",
        "unique_categories",
        "unique_products",
        "session_duration_sec",
        "label",
    ]
    expected_field_types = {
        "source_event_time": pa.timestamp("us"),
        "category_id": pa.string(),
        "user_session": pa.string(),
        "user_id": pa.string(),
        "event_type": pa.string(),
        "product_id": pa.string(),
        "category_code": pa.string(),
        "brand": pa.string(),
        "price": pa.float64(),
        "total_views": pa.int64(),
        "total_carts": pa.int64(),
        "net_cart_count": pa.int64(),
        "cart_to_view_ratio": pa.float64(),
        "unique_categories": pa.int64(),
        "unique_products": pa.int64(),
        "session_duration_sec": pa.float64(),
        "label": pa.int8(),
    }

    assert schemas.GOLD_SCHEMA.names == expected_names
    assert set(schemas.get_gold_fields()) == set(schemas.GOLD_SCHEMA.names)

    for field_name, expected_type in expected_field_types.items():
        assert schemas.GOLD_SCHEMA.field(field_name).type == expected_type
