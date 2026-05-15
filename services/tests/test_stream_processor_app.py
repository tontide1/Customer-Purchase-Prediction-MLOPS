"""Tests for stream processor runtime settings."""

from __future__ import annotations

import sys
import types

from services.stream_processor.app import StreamProcessorSettings


def test_stream_processor_settings_defaults():
    settings = StreamProcessorSettings.from_env({})

    assert settings.kafka_broker == "redpanda:9092"
    assert settings.raw_topic == "raw_events"
    assert settings.late_topic == "late_events"
    assert settings.consumer_group == "stream-processor"
    assert settings.redis_url == "redis://redis:6379/0"
    assert settings.postgres_dsn == "postgresql://mlops:mlops@postgres:5432/mlops"
    assert settings.session_ttl_seconds == 1800
    assert settings.late_threshold_seconds == 60
    assert settings.processing_guarantee == "exactly-once"


def test_create_connection_pool_configures_health_check(monkeypatch):
    captured = {}

    class FakePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.wait_called = False

        def wait(self):
            self.wait_called = True

    monkeypatch.setitem(
        sys.modules,
        "psycopg_pool",
        types.SimpleNamespace(ConnectionPool=FakePool),
    )

    from services.stream_processor.app import create_connection_pool

    pool = create_connection_pool("postgresql://mlops:mlops@postgres:5432/mlops")

    assert captured["conninfo"] == "postgresql://mlops:mlops@postgres:5432/mlops"
    assert captured["open"] is True
    assert captured["min_size"] == 1
    assert captured["max_size"] == 1
    assert callable(captured["check"])
    assert pool.wait_called is True


def test_build_app_uses_connection_pool_and_wires_runtime_dependencies(monkeypatch):
    created = {}

    class FakeTopic:
        def __init__(self, name):
            self.name = name

    class FakeApplyResult:
        def __init__(self, dataframe, fn):
            self.dataframe = dataframe
            self.fn = fn

        def to_topic(self, topic, key):
            self.dataframe.to_topic_call = {
                "topic": topic.name,
                "key": key,
                "value": self.fn,
            }
            return self

    class FakeDataFrame:
        def __init__(self):
            self.updated = None
            self.apply_calls = []
            self.filtered_by = None
            self.to_topic_call = None

        def apply(self, fn):
            self.apply_calls.append(fn)
            return FakeApplyResult(self, fn)

        def update(self, fn):
            self.updated = fn
            return self

        def __getitem__(self, predicate):
            self.filtered_by = predicate.fn
            created["late_filter_fn"] = predicate.fn
            late_branch = FakeDataFrame()
            created["late_branch"] = late_branch
            return late_branch

    class FakeApplication:
        def __init__(self, **kwargs):
            created["application_kwargs"] = kwargs
            self.topics = []
            self.dataframe_obj = FakeDataFrame()

        def topic(self, name, **kwargs):
            self.topics.append((name, kwargs))
            return FakeTopic(name)

        def get_producer(self):
            raise AssertionError("get_producer should not be used")

        def dataframe(self, topic):
            created["dataframe_topic"] = topic.name
            return self.dataframe_obj

    class FakeRedisModule:
        class Redis:
            @staticmethod
            def from_url(url, decode_responses):
                created["redis_url"] = url
                created["decode_responses"] = decode_responses
                return object()

    fake_quixstreams = types.SimpleNamespace(Application=FakeApplication)
    monkeypatch.setitem(sys.modules, "quixstreams", fake_quixstreams)
    monkeypatch.setitem(sys.modules, "redis", FakeRedisModule)
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        types.SimpleNamespace(connect=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("connect should not be used"))),
    )

    from services.stream_processor import app as app_module

    pool = object()

    def fake_create_connection_pool(postgres_dsn):
        created["postgres_dsn"] = postgres_dsn
        return pool

    class FakeReplayStore:
        def __init__(self, pool_arg):
            created["replay_store_pool"] = pool_arg

    def fake_process_event(redis_client, replay_store, event, **kwargs):
        created["process_event_wired"] = True
        created["process_event_kwargs"] = kwargs
        event["_stream_processor_status"] = "late"
        event["late_reason"] = "test_late_reason"
        return "late"

    monkeypatch.setattr(app_module, "create_connection_pool", fake_create_connection_pool)
    monkeypatch.setattr(app_module, "ReplayEventStore", FakeReplayStore)
    monkeypatch.setattr(app_module, "process_event", fake_process_event)

    settings = StreamProcessorSettings.from_env({})
    application = app_module.build_app(settings)
    event = {"event_id": "e1", "user_session": "session-1"}
    application.dataframe_obj.updated(event)

    assert created["postgres_dsn"] == settings.postgres_dsn
    assert created["replay_store_pool"] is pool
    assert created["dataframe_topic"] == "raw_events"
    assert created["application_kwargs"] == {
        "broker_address": settings.kafka_broker,
        "consumer_group": settings.consumer_group,
        "auto_offset_reset": "earliest",
        "processing_guarantee": "exactly-once",
    }
    assert created["redis_url"] == settings.redis_url
    assert created["decode_responses"] is True
    assert application.dataframe_obj.updated is not None
    assert created["process_event_wired"] is True
    assert created["process_event_kwargs"] == {
        "ttl_seconds": settings.session_ttl_seconds,
        "late_threshold_seconds": settings.late_threshold_seconds,
    }
    assert event["_stream_processor_status"] == "late"
    assert created["late_filter_fn"](event) is True
    assert created["late_filter_fn"]({"_stream_processor_status": "accepted"}) is False
    assert application.dataframe_obj.filtered_by is created["late_filter_fn"]
    assert len(application.dataframe_obj.apply_calls) == 1

    late_branch = created["late_branch"]
    assert late_branch.to_topic_call["topic"] == "late_events"
    assert late_branch.to_topic_call["key"](event) == "session-1"
    assert late_branch.to_topic_call["value"](event) == {
        "event_id": "e1",
        "user_session": "session-1",
        "late_reason": "test_late_reason",
    }
