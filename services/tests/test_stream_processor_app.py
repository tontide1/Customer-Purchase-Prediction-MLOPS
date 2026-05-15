"""Tests for stream processor runtime settings."""

from __future__ import annotations

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
