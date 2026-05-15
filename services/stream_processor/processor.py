"""Replay event processing policy for the stream processor."""

from __future__ import annotations

import datetime as dt
from typing import Any

from services.stream_processor.state import apply_event_to_session_state

STREAM_PROCESSOR_STATUS_FIELD = "_stream_processor_status"


def _parse_iso_timestamp(value: Any) -> dt.datetime:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).replace("Z", "+00:00")
    timestamp = dt.datetime.fromisoformat(text)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.astimezone(dt.timezone.utc)


def _last_event_time(redis_client, user_session: str) -> dt.datetime | None:
    state = redis_client.hgetall(f"session:{user_session}")
    value = state.get("last_event_time")
    if value is None or value == "":
        return None
    return _parse_iso_timestamp(value)


def _is_late(redis_client, event: dict[str, Any], *, late_threshold_seconds: int) -> bool:
    last_event_time = _last_event_time(redis_client, event["user_session"])
    if last_event_time is None:
        return False
    source_event_time = _parse_iso_timestamp(event["source_event_time"])
    return source_event_time < last_event_time - dt.timedelta(seconds=late_threshold_seconds)


def process_event(
    redis_client,
    replay_store,
    event: dict[str, Any],
    *,
    ttl_seconds: int = 1800,
    late_threshold_seconds: int = 60,
) -> str:
    dedup_key = f"dedup:event:{event['event_id']}"
    if not redis_client.set(dedup_key, "1", nx=True, ex=ttl_seconds):
        event[STREAM_PROCESSOR_STATUS_FIELD] = "duplicate"
        return "duplicate"

    try:
        if _is_late(redis_client, event, late_threshold_seconds=late_threshold_seconds):
            event["late_reason"] = (
                f"older_than_last_event_time_by_more_than_{late_threshold_seconds}s"
            )
            event[STREAM_PROCESSOR_STATUS_FIELD] = "late"
            return "late"

        replay_store.append(event)
        # Persist the append first so a failed write does not leave Redis ahead of Postgres.
        apply_event_to_session_state(redis_client, event, ttl_seconds=ttl_seconds)
        event[STREAM_PROCESSOR_STATUS_FIELD] = "accepted"
        return "accepted"
    except Exception:
        redis_client.delete(dedup_key)
        raise
