"""Quix Streams entrypoint for replay event processing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from services.stream_processor.db import ReplayEventStore
from services.stream_processor.processor import STREAM_PROCESSOR_STATUS_FIELD, process_event


@dataclass(frozen=True)
class StreamProcessorSettings:
    kafka_broker: str
    raw_topic: str
    late_topic: str
    consumer_group: str
    redis_url: str
    postgres_dsn: str
    session_ttl_seconds: int
    late_threshold_seconds: int
    processing_guarantee: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "StreamProcessorSettings":
        source = os.environ if env is None else env
        return cls(
            kafka_broker=source.get("KAFKA_BROKER", "redpanda:9092"),
            raw_topic=source.get("RAW_EVENTS_TOPIC", "raw_events"),
            late_topic=source.get("LATE_EVENTS_TOPIC", "late_events"),
            consumer_group=source.get("STREAM_PROCESSOR_GROUP", "stream-processor"),
            redis_url=source.get("REDIS_URL", "redis://redis:6379/0"),
            postgres_dsn=source.get(
                "POSTGRES_DSN",
                "postgresql://mlops:mlops@postgres:5432/mlops",
            ),
            session_ttl_seconds=int(source.get("SESSION_TTL_SECONDS", "1800")),
            late_threshold_seconds=int(source.get("LATE_EVENT_THRESHOLD_SECONDS", "60")),
            processing_guarantee=source.get("PROCESSING_GUARANTEE", "exactly-once"),
        )


def _check_connection(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")


def create_connection_pool(postgres_dsn: str):
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=postgres_dsn,
        min_size=1,
        max_size=1,
        open=True,
        check=_check_connection,
        reconnect_timeout=30.0,
    )
    pool.wait()
    return pool


def strip_internal_status(event: dict[str, Any]) -> dict[str, Any]:
    clean_event = dict(event)
    clean_event.pop(STREAM_PROCESSOR_STATUS_FIELD, None)
    return clean_event


def build_app(settings: StreamProcessorSettings) -> Any:
    from quixstreams import Application
    import redis

    application = Application(
        broker_address=settings.kafka_broker,
        consumer_group=settings.consumer_group,
        auto_offset_reset="earliest",
        processing_guarantee=settings.processing_guarantee,
    )
    raw_topic = application.topic(
        settings.raw_topic,
        value_deserializer="json",
        key_deserializer="str",
    )
    late_topic = application.topic(
        settings.late_topic,
        value_serializer="json",
        key_serializer="str",
    )
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    connection_pool = create_connection_pool(settings.postgres_dsn)
    replay_store = ReplayEventStore(connection_pool)

    sdf = application.dataframe(raw_topic)
    sdf = sdf.update(
        lambda event: process_event(
            redis_client,
            replay_store,
            event,
            ttl_seconds=settings.session_ttl_seconds,
            late_threshold_seconds=settings.late_threshold_seconds,
        )
    )
    late_sdf = sdf[
        sdf.apply(lambda event: event.get(STREAM_PROCESSOR_STATUS_FIELD) == "late")
    ]
    late_sdf.apply(strip_internal_status).to_topic(
        late_topic,
        key=lambda event: event["user_session"],
    )
    return application


def main() -> int:
    settings = StreamProcessorSettings.from_env()
    app = build_app(settings)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
