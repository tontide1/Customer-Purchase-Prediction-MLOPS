"""Tests for bounded raw replay normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from services.simulator.replay import normalize_raw_row


def test_normalize_raw_row_renames_event_time_and_hashes_after_rename():
    row = {
        "event_time": "2019-11-01 00:00:00 UTC",
        "event_type": "view",
        "product_id": "100",
        "category_id": "200",
        "category_code": None,
        "brand": "sony",
        "price": 10.5,
        "user_id": "42",
        "user_session": "session-1",
    }

    event = normalize_raw_row(row, replay_time="2026-05-15T09:00:00")

    assert "event_time" not in event
    assert event["source_event_time"] == "2019-11-01T00:00:00"
    assert event["replay_time"] == "2026-05-15T09:00:00"
    assert event["source"] == "replay"
    assert event["event_id"]


def test_normalize_raw_row_rejects_missing_required_category_id():
    row = {
        "event_time": "2019-11-01 00:00:00 UTC",
        "event_type": "view",
        "product_id": "100",
        "category_id": None,
        "user_id": "42",
        "user_session": "session-1",
    }

    with pytest.raises(ValueError, match="Missing required raw field: category_id"):
        normalize_raw_row(row, replay_time="2026-05-15T09:00:00")


def test_normalize_raw_row_rejects_unknown_event_type():
    row = {
        "event_time": "2019-11-01 00:00:00 UTC",
        "event_type": "wishlist",
        "product_id": "100",
        "category_id": "200",
        "user_id": "42",
        "user_session": "session-1",
    }

    with pytest.raises(ValueError, match="Invalid event_type: wishlist"):
        normalize_raw_row(row, replay_time="2026-05-15T09:00:00")


def test_normalize_raw_row_keeps_nullable_fields_as_none():
    row = {
        "event_time": "2019-11-01 00:00:00 UTC",
        "event_type": "cart",
        "product_id": "100",
        "category_id": "200",
        "category_code": pd.NA,
        "brand": pd.NA,
        "price": pd.NA,
        "user_id": "42",
        "user_session": "session-1",
    }

    event = normalize_raw_row(row, replay_time="2026-05-15T09:00:00")

    assert event["category_code"] is None
    assert event["brand"] is None
    assert event["price"] is None
