"""Tests for replay stream event processing."""

from __future__ import annotations

import pytest

from services.stream_processor.processor import process_event
from services.stream_processor.db import ReplayEventStore


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.set_calls = []
        self.hashes = {}
        self.sets = {}
        self.expiries = {}
        self.deleted = []

    def set(self, key, value, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def expire(self, key, ttl_seconds):
        self.expiries[key] = ttl_seconds

    def delete(self, key):
        self.values.pop(key, None)
        self.deleted.append(key)


class FakeLateProducer:
    def __init__(self):
        self.messages = []

    def produce(self, *, topic, key, value):
        self.messages.append({"topic": topic, "key": key, "value": value})


class FakeReplayStore:
    def __init__(self):
        self.events = []

    def append(self, event):
        self.events.append(event)


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
        "price": 1.0,
        "replay_time": "2026-05-15T09:00:00",
        "source": "replay",
    }
    event.update(overrides)
    return event


def test_process_event_suppresses_duplicate_event_ids():
    redis = FakeRedis()
    late = FakeLateProducer()
    store = FakeReplayStore()

    assert process_event(redis, store, late, _event(), late_topic="late_events") == "accepted"
    assert process_event(redis, store, late, _event(), late_topic="late_events") == "duplicate"

    assert store.events == [_event()]
    assert late.messages == []
    assert redis.set_calls[0] == ("dedup:event:event-1", "1", True, 1800)
    assert redis.set_calls[1] == ("dedup:event:event-1", "1", True, 1800)


def test_process_event_routes_late_event_and_skips_state_and_postgres():
    redis = FakeRedis()
    late = FakeLateProducer()
    store = FakeReplayStore()

    accepted = _event(event_id="e1", source_event_time="2019-11-01T00:02:00")
    late_event = _event(event_id="e2", source_event_time="2019-11-01T00:00:30")
    redis.hashes["session:session-1"] = {"last_event_time": accepted["source_event_time"]}

    assert process_event(redis, store, late, accepted, late_topic="late_events") == "accepted"
    assert process_event(redis, store, late, late_event, late_topic="late_events") == "late"

    assert store.events == [accepted]
    assert late.messages == [
        {
            "topic": "late_events",
            "key": "session-1",
            "value": {
                **late_event,
                "late_reason": "older_than_last_event_time_by_more_than_60s",
            },
        }
    ]
    assert "session:session-1" in redis.hashes
    assert "session:session-1:products" in redis.sets
    assert redis.deleted == ["cache:predict:session:session-1"]


def test_process_event_cleans_up_dedup_key_and_leaves_state_unmutated_when_append_fails():
    class FailingReplayStore(FakeReplayStore):
        def append(self, event):
            raise RuntimeError("append failed")

    redis = FakeRedis()
    late = FakeLateProducer()
    store = FailingReplayStore()

    with pytest.raises(RuntimeError, match="append failed"):
        process_event(redis, store, late, _event(), late_topic="late_events")

    assert "dedup:event:event-1" not in redis.values
    assert "dedup:event:event-1" in redis.deleted
    assert redis.hashes == {}
    assert late.messages == []


def test_process_event_cleans_up_dedup_key_when_pipeline_execute_fails():
    class FailingPipeline:
        def __init__(self):
            self.ops = []

        def hset(self, key, mapping):
            self.ops.append(("hset", key, mapping))

        def sadd(self, key, value):
            self.ops.append(("sadd", key, value))

        def expire(self, key, ttl_seconds):
            self.ops.append(("expire", key, ttl_seconds))

        def delete(self, key):
            self.ops.append(("delete", key))

        def execute(self):
            raise RuntimeError("pipeline failed")

    class PipelineRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.pipeline_calls = []

        def pipeline(self, transaction=True):
            self.pipeline_calls.append(transaction)
            return FailingPipeline()

    redis = PipelineRedis()
    late = FakeLateProducer()
    store = FakeReplayStore()

    with pytest.raises(RuntimeError, match="pipeline failed"):
        process_event(redis, store, late, _event(), late_topic="late_events")

    assert redis.pipeline_calls == [True]
    assert redis.hashes == {}
    assert redis.sets == {}
    assert "dedup:event:event-1" not in redis.values
    assert "dedup:event:event-1" in redis.deleted
    assert late.messages == []


def test_replay_event_store_rolls_back_when_execute_fails():
    class FailingCursor:
        def execute(self, sql, params):
            raise RuntimeError("execute failed")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FailingConnection:
        def __init__(self):
            self.rollback_calls = 0
            self.commit_calls = 0

        def cursor(self):
            return FailingCursor()

        def rollback(self):
            self.rollback_calls += 1

        def commit(self):
            self.commit_calls += 1

    connection = FailingConnection()
    store = ReplayEventStore(connection)

    with pytest.raises(RuntimeError, match="execute failed"):
        store.append(_event())

    assert connection.rollback_calls == 1
    assert connection.commit_calls == 0


def test_replay_event_store_rolls_back_when_commit_fails():
    class FakeCursor:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params):
            self.executed.append((sql, params))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FailingCommitConnection:
        def __init__(self):
            self.cursor_obj = FakeCursor()
            self.rollback_calls = 0

        def cursor(self):
            return self.cursor_obj

        def rollback(self):
            self.rollback_calls += 1

        def commit(self):
            raise RuntimeError("commit failed")

    connection = FailingCommitConnection()
    store = ReplayEventStore(connection)

    with pytest.raises(RuntimeError, match="commit failed"):
        store.append(_event())

    assert connection.cursor_obj.executed
    assert connection.rollback_calls == 1


def test_replay_event_store_appends_with_conflict_do_nothing():
    class FakeCursor:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params):
            self.executed.append((sql, params))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def __init__(self):
            self.cursor_obj = FakeCursor()
            self.commits = 0

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.commits += 1

    connection = FakeConnection()
    store = ReplayEventStore(connection)
    event = _event()

    store.append(event)

    assert "ON CONFLICT (event_id) DO NOTHING" in connection.cursor_obj.executed[0][0]
    assert connection.cursor_obj.executed[0][1] == event
    assert connection.commits == 1
