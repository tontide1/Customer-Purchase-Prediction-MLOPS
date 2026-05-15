"""Redis-backed online session state updates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from training.src.features import normalize_category_value


def _redis_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    return bool(pd.isna(value))


def _state_value(state: dict, key: str, default: str) -> str:
    value = state.get(key, default)
    return _redis_text(value)


def _latest_nullable_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value)


def apply_event_to_session_state(redis_client, event: dict[str, Any], *, ttl_seconds: int) -> None:
    session = event["user_session"]
    hash_key = f"session:{session}"
    products_key = f"{hash_key}:products"
    categories_key = f"{hash_key}:categories"

    current = redis_client.hgetall(hash_key)
    count_view = int(_state_value(current, "count_view", "0"))
    count_cart = int(_state_value(current, "count_cart", "0"))
    count_remove = int(_state_value(current, "count_remove_from_cart", "0"))

    if event["event_type"] == "view":
        count_view += 1
    elif event["event_type"] == "cart":
        count_cart += 1
    elif event["event_type"] == "remove_from_cart":
        count_remove += 1

    first_event_time = _state_value(current, "first_event_time", event["source_event_time"])
    price = 0 if _is_missing(event.get("price")) else event["price"]

    redis_client.hset(
        hash_key,
        mapping={
            "first_event_time": first_event_time,
            "last_event_time": event["source_event_time"],
            "count_view": str(count_view),
            "count_cart": str(count_cart),
            "count_remove_from_cart": str(count_remove),
            "latest_price": str(price),
            "latest_category_id": event["category_id"],
            "latest_category_code": _latest_nullable_text(event.get("category_code")),
            "latest_brand": _latest_nullable_text(event.get("brand")),
            "latest_event_type": event["event_type"],
        },
    )
    redis_client.sadd(products_key, event["product_id"])
    redis_client.sadd(
        categories_key,
        normalize_category_value(
            None if _is_missing(event.get("category_code")) else event.get("category_code"),
            event["category_id"],
        ),
    )

    for key in (hash_key, products_key, categories_key):
        redis_client.expire(key, ttl_seconds)
    redis_client.delete(f"cache:predict:session:{session}")
