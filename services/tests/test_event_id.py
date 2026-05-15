"""Tests for the canonical replay event ID helper."""

from __future__ import annotations

import hashlib

from shared.event_id import compute_event_id


def test_compute_event_id_is_deterministic():
    value = compute_event_id(
        user_session="session-1",
        source_event_time="2019-11-01T00:00:00",
        event_type="view",
        product_id="100",
        user_id="42",
    )

    expected_payload = "session-1|2019-11-01T00:00:00|view|100|42"
    assert value == hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    assert value == compute_event_id(
        user_session="session-1",
        source_event_time="2019-11-01T00:00:00",
        event_type="view",
        product_id="100",
        user_id="42",
    )
