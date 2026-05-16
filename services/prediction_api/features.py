"""Build model-ready feature rows from Redis online state."""

from __future__ import annotations

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


def _required_text(state: dict[str, Any], key: str) -> str:
    value = _text(state[key])
    if value == "":
        raise ValueError(f"Missing required serving field: {key}")
    return value


def _required_int(state: dict[str, Any], key: str) -> int:
    return int(_text(state[key]))


def _required_float(state: dict[str, Any], key: str) -> float:
    return float(_text(state[key]))


def build_feature_row(
    redis_client,
    user_session: str,
    bundle: ServingBundle,
) -> pd.DataFrame | None:
    hash_key = f"session:{user_session}"
    state = redis_client.hgetall(hash_key)
    if not state:
        return None

    try:
        values = {
            "total_views": _required_int(state, "serving_total_views"),
            "total_carts": _required_int(state, "serving_total_carts"),
            "net_cart_count": _required_int(state, "serving_net_cart_count"),
            "cart_to_view_ratio": _required_float(
                state, "serving_cart_to_view_ratio"
            ),
            "unique_categories": _required_int(state, "serving_unique_categories"),
            "unique_products": _required_int(state, "serving_unique_products"),
            "session_duration_sec": _required_float(
                state, "serving_session_duration_sec"
            ),
            "price": _required_float(state, "serving_price"),
            "category_id": _categorical_value(
                _required_text(state, "serving_category_id"),
                bundle.category_maps["category_id"],
                missing_token=bundle.missing_token,
                unknown_token=bundle.unknown_token,
            ),
            "category_code": _categorical_value(
                state["serving_category_code"],
                bundle.category_maps["category_code"],
                missing_token=bundle.missing_token,
                unknown_token=bundle.unknown_token,
            ),
            "brand": _categorical_value(
                state["serving_brand"],
                bundle.category_maps["brand"],
                missing_token=bundle.missing_token,
                unknown_token=bundle.unknown_token,
            ),
            "event_type": _categorical_value(
                _required_text(state, "serving_event_type"),
                bundle.category_maps["event_type"],
                missing_token=bundle.missing_token,
                unknown_token=bundle.unknown_token,
            ),
        }
    except (KeyError, TypeError, ValueError):
        return None

    frame = pd.DataFrame([{column: values[column] for column in bundle.feature_column_order}])
    for column, mapping in bundle.category_maps.items():
        frame[column] = pd.Categorical(
            frame[column],
            categories=list(mapping.keys()),
            ordered=False,
        )
    return frame
