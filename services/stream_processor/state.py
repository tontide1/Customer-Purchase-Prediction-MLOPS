"""Redis-backed online session state updates."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

from training.src.features import normalize_category_value


def _redis_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if value == "":
            return True
    except Exception:
        pass
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _state_value(state: dict, key: str, default: str) -> str:
    value = state.get(key, default)
    return _redis_text(value)


def _latest_nullable_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value)


def _normalize_utc_timestamp_text(value: Any) -> str:
    text = _redis_text(value).replace("Z", "+00:00")
    timestamp = dt.datetime.fromisoformat(text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    else:
        timestamp = timestamp.astimezone(dt.timezone.utc)
    return timestamp.isoformat()


def _session_duration_seconds(first_event_time: str, current_event_time: str) -> float:
    return (
        dt.datetime.fromisoformat(current_event_time)
        - dt.datetime.fromisoformat(first_event_time)
    ).total_seconds()


def apply_event_to_session_state(redis_client, event: dict[str, Any], *, ttl_seconds: int) -> None:
    session = event["user_session"]
    hash_key = f"session:{session}"
    products_key = f"{hash_key}:products"
    categories_key = f"{hash_key}:categories"

    current = redis_client.hgetall(hash_key)
    count_view = int(_state_value(current, "count_view", "0"))
    count_cart = int(_state_value(current, "count_cart", "0"))
    count_remove = int(_state_value(current, "count_remove_from_cart", "0"))
    serving_total_views = count_view
    serving_total_carts = count_cart
    serving_total_removes = count_remove
    serving_net_cart_count = serving_total_carts - serving_total_removes
    serving_cart_to_view_ratio = (
        0.0 if serving_total_views == 0 else serving_total_carts / serving_total_views
    )
    serving_unique_products = redis_client.scard(products_key)
    serving_unique_categories = redis_client.scard(categories_key)

    if event["event_type"] == "view":
        count_view += 1
    elif event["event_type"] == "cart":
        count_cart += 1
    elif event["event_type"] == "remove_from_cart":
        count_remove += 1

    first_event_time = _normalize_utc_timestamp_text(
        _state_value(current, "first_event_time", event["source_event_time"])
    )
    last_event_time = _normalize_utc_timestamp_text(event["source_event_time"])
    price = 0 if _is_missing(event.get("price")) else event["price"]

    mapping = {
        "first_event_time": first_event_time,
        "last_event_time": last_event_time,
        "count_view": str(count_view),
        "count_cart": str(count_cart),
        "count_remove_from_cart": str(count_remove),
        "latest_price": str(price),
        "latest_category_id": event["category_id"],
        "latest_category_code": _latest_nullable_text(event.get("category_code")),
        "latest_brand": _latest_nullable_text(event.get("brand")),
        "latest_event_type": event["event_type"],
        "serving_total_views": str(serving_total_views),
        "serving_total_carts": str(serving_total_carts),
        "serving_total_removes": str(serving_total_removes),
        "serving_net_cart_count": str(serving_net_cart_count),
        "serving_cart_to_view_ratio": str(serving_cart_to_view_ratio),
        "serving_unique_categories": str(serving_unique_categories),
        "serving_unique_products": str(serving_unique_products),
        "serving_session_duration_sec": str(
            _session_duration_seconds(first_event_time, last_event_time)
        ),
        "serving_price": str(price),
        "serving_category_id": event["category_id"],
        "serving_category_code": _latest_nullable_text(event.get("category_code")),
        "serving_brand": _latest_nullable_text(event.get("brand")),
        "serving_event_type": event["event_type"],
    }
    category_value = normalize_category_value(
        None if _is_missing(event.get("category_code")) else event.get("category_code"),
        event["category_id"],
    )

    pipeline_factory = getattr(redis_client, "pipeline", None)
    if callable(pipeline_factory):
        pipeline = pipeline_factory(transaction=True)
        pipeline.hset(hash_key, mapping=mapping)
        pipeline.sadd(products_key, event["product_id"])
        pipeline.sadd(categories_key, category_value)
        for key in (hash_key, products_key, categories_key):
            pipeline.expire(key, ttl_seconds)
        pipeline.delete(f"cache:predict:session:{session}")
        pipeline.execute()
        return

    redis_client.hset(hash_key, mapping=mapping)
    redis_client.sadd(products_key, event["product_id"])
    redis_client.sadd(categories_key, category_value)

    for key in (hash_key, products_key, categories_key):
        redis_client.expire(key, ttl_seconds)
    redis_client.delete(f"cache:predict:session:{session}")
