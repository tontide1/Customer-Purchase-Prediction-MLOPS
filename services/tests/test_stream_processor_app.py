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

    class FakeDataFrame:
        def __init__(self):
            self.applied = None

        def apply(self, fn):
            self.applied = fn
            return self

    class FakeApplication:
        def __init__(self, **kwargs):
            created["application_kwargs"] = kwargs
            self.topics = []
            self.dataframe_obj = FakeDataFrame()

        def topic(self, name, **kwargs):
            self.topics.append((name, kwargs))
            return FakeTopic(name)

        def get_producer(self):
            created["producer_requested"] = True
            return object()

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

    def fake_process_event(*args, **kwargs):
        created["process_event_wired"] = True

    monkeypatch.setattr(app_module, "create_connection_pool", fake_create_connection_pool)
    monkeypatch.setattr(app_module, "ReplayEventStore", FakeReplayStore)
    monkeypatch.setattr(app_module, "process_event", fake_process_event)

    settings = StreamProcessorSettings.from_env({})
    application = app_module.build_app(settings)
    application.dataframe_obj.applied({"event_id": "e1"})

    assert created["postgres_dsn"] == settings.postgres_dsn
    assert created["replay_store_pool"] is pool
    assert created["dataframe_topic"] == "raw_events"
    assert created["application_kwargs"] == {
        "broker_address": settings.kafka_broker,
        "consumer_group": settings.consumer_group,
        "auto_offset_reset": "earliest",
    }
    assert created["redis_url"] == settings.redis_url
    assert created["decode_responses"] is True
    assert application.dataframe_obj.applied is not None
    assert created["process_event_wired"] is True
