"""Tests for the Sprint 2a gold schema contract."""

import pyarrow as pa

from shared import schemas


def test_gold_schema_matches_mvp_contract() -> None:
    assert schemas.GOLD_SCHEMA.names == [
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

    assert schemas.GOLD_SCHEMA.field("cart_to_view_ratio").type == pa.float64()
    assert schemas.GOLD_SCHEMA.field("session_duration_sec").type == pa.float64()
    assert schemas.GOLD_SCHEMA.field("label").type == pa.int8()
