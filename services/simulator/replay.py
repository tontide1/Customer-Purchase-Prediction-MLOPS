"""Normalize and replay raw November events into Kafka-compatible topics."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from shared import constants
from shared.event_id import compute_event_id

RAW_REQUIRED_FIELDS = (
    constants.FIELD_EVENT_TIME,
    "event_type",
    "product_id",
    constants.FIELD_CATEGORY_ID,
    "user_id",
    "user_session",
)

RAW_COLUMNS = [
    constants.FIELD_EVENT_TIME,
    "event_type",
    "product_id",
    constants.FIELD_CATEGORY_ID,
    "user_id",
    "user_session",
    "category_code",
    "brand",
    "price",
]


def _is_missing(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, str) and value == "")
        or bool(pd.isna(value))
    )


def _as_text(value: Any) -> str:
    return str(value)


def _normalize_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.isoformat()


def _nullable_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


def _nullable_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def normalize_raw_row(row: dict[str, Any], replay_time: str | None = None) -> dict[str, Any]:
    for field in RAW_REQUIRED_FIELDS:
        if field not in row or _is_missing(row[field]) or str(row[field]) == "":
            raise ValueError(f"Missing required raw field: {field}")

    event_type = _as_text(row["event_type"])
    if event_type not in constants.ALLOWED_EVENT_TYPES:
        raise ValueError(f"Invalid event_type: {event_type}")

    source_event_time = _normalize_timestamp(row[constants.FIELD_EVENT_TIME])
    normalized = {
        constants.FIELD_SOURCE_EVENT_TIME: source_event_time,
        "event_type": event_type,
        "product_id": _as_text(row["product_id"]),
        constants.FIELD_CATEGORY_ID: _as_text(row[constants.FIELD_CATEGORY_ID]),
        "user_id": _as_text(row["user_id"]),
        "user_session": _as_text(row["user_session"]),
        "category_code": _nullable_text(row.get("category_code")),
        "brand": _nullable_text(row.get("brand")),
        "price": _nullable_float(row.get("price")),
        "replay_time": replay_time or dt.datetime.utcnow().replace(microsecond=0).isoformat(),
        "source": "replay",
    }
    normalized["event_id"] = compute_event_id(
        user_session=normalized["user_session"],
        source_event_time=normalized[constants.FIELD_SOURCE_EVENT_TIME],
        event_type=normalized["event_type"],
        product_id=normalized["product_id"],
        user_id=normalized["user_id"],
    )
    return normalized


def iter_replay_events(
    csv_path: str | Path,
    *,
    limit: int,
    replay_time: str | None = None,
) -> Iterable[dict[str, Any]]:
    frame = pd.read_csv(csv_path, usecols=RAW_COLUMNS, nrows=limit)
    frame[constants.FIELD_EVENT_TIME] = pd.to_datetime(frame[constants.FIELD_EVENT_TIME], utc=True)
    frame = frame.sort_values(["user_session", constants.FIELD_EVENT_TIME], kind="mergesort")
    for row in frame.to_dict(orient="records"):
        yield normalize_raw_row(row, replay_time=replay_time)


def publish_events(
    events: Iterable[dict[str, Any]],
    *,
    producer,
    topic: Any,
) -> int:
    if not hasattr(topic, "name") or not hasattr(topic, "serialize"):
        raise TypeError("topic must provide .name and .serialize(key=..., value=...)")

    count = 0
    for event in events:
        key = event["user_session"]
        message = topic.serialize(key=key, value=event)
        producer.produce(topic=topic.name, key=message.key, value=message.value)
        count += 1
    producer.flush()
    return count
