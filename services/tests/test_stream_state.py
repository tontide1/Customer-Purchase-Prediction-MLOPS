"""Tests for Redis online session feature state."""

from __future__ import annotations

from services.stream_processor.state import apply_event_to_session_state


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.sets = {}
        self.ttls = {}
        self.deleted = []

    def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def expire(self, key, ttl_seconds):
        self.ttls[key] = ttl_seconds

    def delete(self, key):
        self.deleted.append(key)


def _event(**overrides):
    event = {
        "event_id": "event-1",
        "user_session": "session-1",
        "source_event_time": "2019-11-01T00:00:00",
        "event_type": "view",
        "product_id": "100",
        "user_id": "42",
        "category_id": "cat-id",
        "category_code": None,
        "brand": None,
        "price": None,
        "replay_time": "2026-05-15T09:00:00",
        "source": "replay",
    }
    event.update(overrides)
    return event


def test_apply_event_initializes_hash_sets_ttl_and_invalidates_cache():
    redis = FakeRedis()

    apply_event_to_session_state(redis, _event(), ttl_seconds=1800)

    assert redis.hashes["session:session-1"] == {
        "first_event_time": "2019-11-01T00:00:00",
        "last_event_time": "2019-11-01T00:00:00",
        "count_view": "1",
        "count_cart": "0",
        "count_remove_from_cart": "0",
        "latest_price": "0",
        "latest_category_id": "cat-id",
        "latest_category_code": "",
        "latest_brand": "",
        "latest_event_type": "view",
    }
    assert redis.sets["session:session-1:products"] == {"100"}
    assert redis.sets["session:session-1:categories"] == {"cat-id"}
    assert redis.ttls["session:session-1"] == 1800
    assert redis.ttls["session:session-1:products"] == 1800
    assert redis.ttls["session:session-1:categories"] == 1800
    assert redis.deleted == ["cache:predict:session:session-1"]


def test_apply_event_updates_counts_preserves_first_event_and_normalizes_category_set():
    redis = FakeRedis()

    apply_event_to_session_state(redis, _event(event_id="e1", event_type="view"), ttl_seconds=1800)
    apply_event_to_session_state(
        redis,
        _event(
            event_id="e2",
            event_type="cart",
            product_id="200",
            category_id="fallback-id",
            category_code="category.code",
            brand="brand-a",
            price=12.5,
            source_event_time="2019-11-01T00:02:00",
        ),
        ttl_seconds=1800,
    )
    apply_event_to_session_state(
        redis,
        _event(
            event_id="e3",
            event_type="remove_from_cart",
            product_id="200",
            source_event_time="2019-11-01T00:03:00",
        ),
        ttl_seconds=1800,
    )

    state = redis.hashes["session:session-1"]
    assert state["first_event_time"] == "2019-11-01T00:00:00"
    assert state["last_event_time"] == "2019-11-01T00:03:00"
    assert state["count_view"] == "1"
    assert state["count_cart"] == "1"
    assert state["count_remove_from_cart"] == "1"
    assert state["latest_price"] == "0"
    assert state["latest_category_code"] == ""
    assert state["latest_brand"] == ""
    assert redis.sets["session:session-1:categories"] == {"cat-id", "category.code"}
