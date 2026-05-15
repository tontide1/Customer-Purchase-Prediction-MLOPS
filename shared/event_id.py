"""Canonical deterministic event identifiers for offline and online events."""

from __future__ import annotations

import hashlib


def compute_event_id(
    *,
    user_session: str,
    source_event_time: str,
    event_type: str,
    product_id: str,
    user_id: str,
) -> str:
    payload = f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
