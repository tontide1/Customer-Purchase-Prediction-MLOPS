"""Tests for replay stream event processing."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from services.stream_processor.processor import STREAM_PROCESSOR_STATUS_FIELD, process_event
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


class FakeReplayStore:
    def __init__(self):
        self.events = []

    def append(self, event):
        self.events.append(dict(event))


class FakeConnectionPool:
    def __init__(self, connection):
        self.connection_obj = connection
        self.connection_calls = 0

    @contextmanager
    def connection(self):
        self.connection_calls += 1
        yield self.connection_obj


def _event(**overrides):
    event = {
        "event_id": "event-1",
        "user_session": "session-1",
        "source_event_time": "2019-11-01T00:00:00+00:00",
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
    store = FakeReplayStore()
    accepted = _event()
    duplicate = _event()

    assert process_event(redis, store, accepted) == "accepted"
    assert process_event(redis, store, duplicate) == "duplicate"

    assert store.events == [_event()]
    assert accepted[STREAM_PROCESSOR_STATUS_FIELD] == "accepted"
    assert duplicate[STREAM_PROCESSOR_STATUS_FIELD] == "duplicate"
    assert "late_reason" not in duplicate
    assert redis.set_calls[0] == ("dedup:event:event-1", "1", True, 1800)
    assert redis.set_calls[1] == ("dedup:event:event-1", "1", True, 1800)


def test_process_event_routes_late_event_and_skips_state_and_postgres():
    redis = FakeRedis()
    store = FakeReplayStore()

    accepted = _event(event_id="e1", source_event_time="2019-11-01T00:02:00+00:00")
    late_event = _event(event_id="e2", source_event_time="2019-11-01T00:00:30+00:00")
    redis.hashes["session:session-1"] = {"last_event_time": accepted["source_event_time"]}

    assert process_event(redis, store, accepted) == "accepted"
    assert process_event(redis, store, late_event) == "late"

    assert store.events == [
        _event(event_id="e1", source_event_time="2019-11-01T00:02:00+00:00")
    ]
    assert late_event[STREAM_PROCESSOR_STATUS_FIELD] == "late"
    assert late_event["late_reason"] == "older_than_last_event_time_by_more_than_60s"
    assert "session:session-1" in redis.hashes
    assert "session:session-1:products" in redis.sets
    assert redis.deleted == ["cache:predict:session:session-1"]


def test_process_event_cleans_up_dedup_key_and_leaves_state_unmutated_when_append_fails():
    class FailingReplayStore(FakeReplayStore):
        def append(self, event):
            raise RuntimeError("append failed")

    redis = FakeRedis()
    store = FailingReplayStore()
    event = _event()

    with pytest.raises(RuntimeError, match="append failed"):
        process_event(redis, store, event)

    assert "dedup:event:event-1" not in redis.values
    assert "dedup:event:event-1" in redis.deleted
    assert redis.hashes == {}
    assert STREAM_PROCESSOR_STATUS_FIELD not in event


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
    store = FakeReplayStore()
    event = _event()

    with pytest.raises(RuntimeError, match="pipeline failed"):
        process_event(redis, store, event)

    assert redis.pipeline_calls == [True]
    assert redis.hashes == {}
    assert redis.sets == {}
    assert "dedup:event:event-1" not in redis.values
    assert "dedup:event:event-1" in redis.deleted
    assert STREAM_PROCESSOR_STATUS_FIELD not in event


def test_process_event_uses_pipeline_for_accepted_events():
    class RecordingPipeline:
        def __init__(self):
            self.commands = []
            self.execute_called = False

        def hset(self, key, mapping):
            self.commands.append(("hset", key, mapping))
            return self

        def sadd(self, key, value):
            self.commands.append(("sadd", key, value))
            return self

        def expire(self, key, ttl_seconds):
            self.commands.append(("expire", key, ttl_seconds))
            return self

        def delete(self, key):
            self.commands.append(("delete", key))
            return self

        def execute(self):
            self.execute_called = True
            return True

    class PipelineRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.pipeline_obj = RecordingPipeline()
            self.pipeline_calls = []

        def pipeline(self, transaction=True):
            self.pipeline_calls.append(transaction)
            return self.pipeline_obj

    redis = PipelineRedis()
    store = FakeReplayStore()
    event = _event()

    assert process_event(redis, store, event) == "accepted"

    assert redis.pipeline_calls == [True]
    assert redis.pipeline_obj.execute_called is True
    assert ("hset", "session:session-1", redis.pipeline_obj.commands[0][2]) in [
        (cmd[0], cmd[1], cmd[2]) for cmd in redis.pipeline_obj.commands if cmd[0] == "hset"
    ]
    assert ("sadd", "session:session-1:products", "100") in redis.pipeline_obj.commands
    assert ("sadd", "session:session-1:categories", "cat-id") in redis.pipeline_obj.commands
    assert ("expire", "session:session-1", 1800) in redis.pipeline_obj.commands
    assert ("delete", "cache:predict:session:session-1") in redis.pipeline_obj.commands
    assert store.events == [_event()]
    assert event[STREAM_PROCESSOR_STATUS_FIELD] == "accepted"


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
    store = ReplayEventStore(FakeConnectionPool(connection))

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
    store = ReplayEventStore(FakeConnectionPool(connection))

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
    pool = FakeConnectionPool(connection)
    store = ReplayEventStore(pool)
    event = _event()

    store.append(event)

    assert "ON CONFLICT (event_id) DO NOTHING" in connection.cursor_obj.executed[0][0]
    assert connection.cursor_obj.executed[0][1] == event
    assert connection.commits == 1


def test_replay_event_store_checks_out_a_new_pool_connection_per_append():
    class FakeCursor:
        def __init__(self, marker):
            self.marker = marker
            self.executed = []

        def execute(self, sql, params):
            self.executed.append((sql, params))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def __init__(self, marker):
            self.marker = marker
            self.cursor_obj = FakeCursor(marker)
            self.commits = 0

        def cursor(self):
            return self.cursor_obj

        def rollback(self):
            raise AssertionError("rollback should not be called")

        def commit(self):
            self.commits += 1

    class CountingPool:
        def __init__(self):
            self.connection_calls = 0
            self.connections = [FakeConnection("first"), FakeConnection("second")]

        @contextmanager
        def connection(self):
            connection = self.connections[self.connection_calls]
            self.connection_calls += 1
            yield connection

    pool = CountingPool()
    store = ReplayEventStore(pool)
    event = _event()

    store.append(event)
    store.append(event)

    assert pool.connection_calls == 2
    assert pool.connections[0].commits == 1
    assert pool.connections[1].commits == 1
