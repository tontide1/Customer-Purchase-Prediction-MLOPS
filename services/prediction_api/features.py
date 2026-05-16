"""Build model-ready feature rows from Redis online state."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from services.prediction_api.bundle import ServingBundle


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _categorical_value(
    value: Any,
    mapping: dict[str, int],
    *,
    missing_token: str,
    unknown_token: str,
) -> str:
    normalized = missing_token if _text(value) == "" else _text(value)
    return normalized if normalized in mapping else unknown_token


def build_feature_row(
    redis_client,
    user_session: str,
    bundle: ServingBundle,
) -> pd.DataFrame | None:
    hash_key = f"session:{user_session}"
    state = redis_client.hgetall(hash_key)
    if not state:
        return None

    first_event_time = dt.datetime.fromisoformat(_text(state["first_event_time"]))
    last_event_time = dt.datetime.fromisoformat(_text(state["last_event_time"]))
    total_views = int(_text(state.get("count_view", 0)))
    total_carts = int(_text(state.get("count_cart", 0)))
    total_removes = int(_text(state.get("count_remove_from_cart", 0)))

    values = {
        "total_views": total_views,
        "total_carts": total_carts,
        "net_cart_count": total_carts - total_removes,
        "cart_to_view_ratio": 0.0 if total_views == 0 else total_carts / total_views,
        "unique_categories": redis_client.scard(f"{hash_key}:categories"),
        "unique_products": redis_client.scard(f"{hash_key}:products"),
        "session_duration_sec": (last_event_time - first_event_time).total_seconds(),
        "price": float(_text(state.get("latest_price", "0"))),
        "category_id": _categorical_value(
            state["latest_category_id"],
            bundle.category_maps["category_id"],
            missing_token=bundle.missing_token,
            unknown_token=bundle.unknown_token,
        ),
        "category_code": _categorical_value(
            state.get("latest_category_code", ""),
            bundle.category_maps["category_code"],
            missing_token=bundle.missing_token,
            unknown_token=bundle.unknown_token,
        ),
        "brand": _categorical_value(
            state.get("latest_brand", ""),
            bundle.category_maps["brand"],
            missing_token=bundle.missing_token,
            unknown_token=bundle.unknown_token,
        ),
        "event_type": _categorical_value(
            state["latest_event_type"],
            bundle.category_maps["event_type"],
            missing_token=bundle.missing_token,
            unknown_token=bundle.unknown_token,
        ),
    }

    frame = pd.DataFrame([{column: values[column] for column in bundle.feature_column_order}])
    for column, mapping in bundle.category_maps.items():
        frame[column] = pd.Categorical(
            frame[column],
            categories=list(mapping.keys()),
            ordered=False,
        )
    return frame

