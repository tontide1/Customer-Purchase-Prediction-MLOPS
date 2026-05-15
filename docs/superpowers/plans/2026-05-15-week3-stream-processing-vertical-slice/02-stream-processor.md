# Week 3 Stream Processor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Quix Streams consumer path that deduplicates replay events, routes late events, updates Redis session state, and appends accepted events to PostgreSQL.

**Architecture:** This plan owns the stateful processing middle of the Week 3 slice. It keeps feature-state semantics aligned with Week 2 gold features while making duplicate and late-event policy explicit and bounded.

**Tech Stack:** Python 3.11, Quix Streams, Redis, PostgreSQL, psycopg, pytest, Docker.

---

### Task 4: Redis Session State Feature Semantics

**Files:**
- Create: `services/stream_processor/__init__.py`
- Create: `services/stream_processor/state.py`
- Test: `services/tests/test_stream_state.py`

- [ ] **Step 1: Write the failing tests**

Create `services/tests/test_stream_state.py`:

```python
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


def test_apply_event_updates_counts_and_normalized_category_set():
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
    assert state["count_view"] == "1"
    assert state["count_cart"] == "1"
    assert state["count_remove_from_cart"] == "1"
    assert state["latest_price"] == "0"
    assert state["latest_category_code"] == ""
    assert state["latest_brand"] == ""
    assert redis.sets["session:session-1:categories"] == {"cat-id", "category.code"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/tests/test_stream_state.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'services.stream_processor'`.

- [ ] **Step 3: Write minimal implementation**

Create `services/stream_processor/__init__.py`:

```python
"""Stream processor service package."""
```

Create `services/stream_processor/state.py`:

```python
"""Redis-backed online session state updates."""

from __future__ import annotations

from typing import Any

from training.src.features import normalize_category_value


def _to_text_int(value: Any, default: int = 0) -> str:
    if value is None or value == "":
        return str(default)
    return str(int(value))


def _redis_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _state_value(state: dict, key: str, default: str) -> str:
    value = state.get(key, default)
    return _redis_text(value)


def _latest_nullable_text(value: Any) -> str:
    return "" if value is None else str(value)


def apply_event_to_session_state(redis_client, event: dict[str, Any], *, ttl_seconds: int) -> None:
    session = event["user_session"]
    hash_key = f"session:{session}"
    products_key = f"{hash_key}:products"
    categories_key = f"{hash_key}:categories"

    current = redis_client.hgetall(hash_key)
    count_view = int(_state_value(current, "count_view", "0"))
    count_cart = int(_state_value(current, "count_cart", "0"))
    count_remove = int(_state_value(current, "count_remove_from_cart", "0"))

    if event["event_type"] == "view":
        count_view += 1
    elif event["event_type"] == "cart":
        count_cart += 1
    elif event["event_type"] == "remove_from_cart":
        count_remove += 1

    first_event_time = _state_value(
        current,
        "first_event_time",
        event["source_event_time"],
    )
    price = 0 if event.get("price") is None else event["price"]
    mapping = {
        "first_event_time": first_event_time,
        "last_event_time": event["source_event_time"],
        "count_view": str(count_view),
        "count_cart": str(count_cart),
        "count_remove_from_cart": str(count_remove),
        "latest_price": str(price),
        "latest_category_id": event["category_id"],
        "latest_category_code": _latest_nullable_text(event.get("category_code")),
        "latest_brand": _latest_nullable_text(event.get("brand")),
        "latest_event_type": event["event_type"],
    }
    redis_client.hset(hash_key, mapping=mapping)
    redis_client.sadd(products_key, event["product_id"])
    redis_client.sadd(
        categories_key,
        normalize_category_value(event.get("category_code"), event["category_id"]),
    )

    for key in (hash_key, products_key, categories_key):
        redis_client.expire(key, ttl_seconds)
    redis_client.delete(f"cache:predict:session:{session}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/tests/test_stream_state.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add services/stream_processor/__init__.py services/stream_processor/state.py services/tests/test_stream_state.py
git commit -m "feat: maintain redis session state"
```

### Task 5: Stream Processor Duplicate Suppression, Late Routing, And PostgreSQL Append

**Files:**
- Create: `services/stream_processor/db.py`
- Create: `services/stream_processor/processor.py`
- Create: `infra/postgres/init.sql`
- Test: `services/tests/test_stream_processor.py`

- [ ] **Step 1: Write the failing tests**

Create `services/tests/test_stream_processor.py`:

```python
"""Tests for stream processor routing and replay persistence."""

from __future__ import annotations

from services.stream_processor.processor import process_event


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.sets = {}
        self.deleted = []
        self.ttls = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        self.ttls[key] = ex
        return True

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


class FakeLateProducer:
    def __init__(self):
        self.messages = []

    def produce(self, *, topic, key, value):
        self.messages.append({"topic": topic, "key": key, "value": value})


class FakeReplayStore:
    def __init__(self):
        self.rows = []

    def append(self, event):
        self.rows.append(event)


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

    assert len(store.rows) == 1
    assert late.messages == []
    assert redis.ttls["dedup:event:event-1"] == 1800


def test_process_event_routes_late_event_away_from_state_and_postgres():
    redis = FakeRedis()
    late = FakeLateProducer()
    store = FakeReplayStore()

    process_event(
        redis,
        store,
        late,
        _event(event_id="e1", source_event_time="2019-11-01T00:02:00"),
        late_topic="late_events",
    )
    result = process_event(
        redis,
        store,
        late,
        _event(event_id="e2", source_event_time="2019-11-01T00:00:30"),
        late_topic="late_events",
        late_threshold_seconds=60,
    )

    assert result == "late"
    assert len(store.rows) == 1
    assert late.messages[0]["topic"] == "late_events"
    assert late.messages[0]["key"] == "session-1"
    assert late.messages[0]["value"]["late_reason"] == "older_than_last_event_time_by_more_than_60s"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/tests/test_stream_processor.py -q`

Expected: fail with `ModuleNotFoundError` for `services.stream_processor.processor`.

- [ ] **Step 3: Write processor implementation**

Create `services/stream_processor/processor.py`:

```python
"""Raw event stream processor policy."""

from __future__ import annotations

import datetime as dt
from typing import Any

from services.stream_processor.state import apply_event_to_session_state


def _parse_iso(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def _last_event_time(redis_client, session: str) -> dt.datetime | None:
    state = redis_client.hgetall(f"session:{session}")
    value = state.get("last_event_time")
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return _parse_iso(str(value))


def _is_late(redis_client, event: dict[str, Any], *, late_threshold_seconds: int) -> bool:
    last_time = _last_event_time(redis_client, event["user_session"])
    if last_time is None:
        return False
    event_time = _parse_iso(event["source_event_time"])
    return event_time < last_time - dt.timedelta(seconds=late_threshold_seconds)


def process_event(
    redis_client,
    replay_store,
    late_producer,
    event: dict[str, Any],
    *,
    late_topic: str,
    ttl_seconds: int = 1800,
    late_threshold_seconds: int = 60,
) -> str:
    dedup_key = f"dedup:event:{event['event_id']}"
    if not redis_client.set(dedup_key, "1", nx=True, ex=ttl_seconds):
        return "duplicate"

    if _is_late(redis_client, event, late_threshold_seconds=late_threshold_seconds):
        late_event = dict(event)
        late_event["late_reason"] = (
            f"older_than_last_event_time_by_more_than_{late_threshold_seconds}s"
        )
        late_producer.produce(
            topic=late_topic,
            key=event["user_session"],
            value=late_event,
        )
        return "late"

    apply_event_to_session_state(redis_client, event, ttl_seconds=ttl_seconds)
    replay_store.append(event)
    return "accepted"
```

Create `services/stream_processor/db.py`:

```python
"""PostgreSQL replay event append store."""

from __future__ import annotations

from typing import Any


class ReplayEventStore:
    def __init__(self, connection):
        self.connection = connection

    def append(self, event: dict[str, Any]) -> None:
        with self.connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO replay_events (
                    event_id, user_session, source_event_time, replay_time,
                    event_type, product_id, user_id, category_id, category_code,
                    brand, price, source
                )
                VALUES (
                    %(event_id)s, %(user_session)s, %(source_event_time)s, %(replay_time)s,
                    %(event_type)s, %(product_id)s, %(user_id)s, %(category_id)s,
                    %(category_code)s, %(brand)s, %(price)s, %(source)s
                )
                ON CONFLICT (event_id) DO NOTHING
                """,
                event,
            )
        self.connection.commit()
```

Create `infra/postgres/init.sql`:

```sql
CREATE TABLE IF NOT EXISTS replay_events (
    event_id text PRIMARY KEY,
    user_session text NOT NULL,
    source_event_time timestamp NOT NULL,
    replay_time timestamp NOT NULL,
    event_type text NOT NULL,
    product_id text NOT NULL,
    user_id text NOT NULL,
    category_id text NOT NULL,
    category_code text NULL,
    brand text NULL,
    price double precision NULL,
    source text NOT NULL,
    created_at timestamp NOT NULL DEFAULT current_timestamp
);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/tests/test_stream_processor.py services/tests/test_stream_state.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/stream_processor/db.py services/stream_processor/processor.py infra/postgres/init.sql services/tests/test_stream_processor.py
git commit -m "feat: process replay stream events"
```

### Task 6: Stream Processor Quix Entrypoint

**Files:**
- Create: `services/stream_processor/app.py`
- Create: `services/stream_processor/requirements.txt`
- Create: `services/stream_processor/Dockerfile`
- Test: `services/tests/test_stream_processor_app.py`

- [ ] **Step 1: Write the failing config test**

Create `services/tests/test_stream_processor_app.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/tests/test_stream_processor_app.py -q`

Expected: fail with `ModuleNotFoundError` for `services.stream_processor.app`.

- [ ] **Step 3: Write runtime entrypoint**

Create `services/stream_processor/app.py`:

```python
"""Quix Streams entrypoint for replay event processing."""

from __future__ import annotations

from dataclasses import dataclass
import os

import psycopg
import redis
from quixstreams import Application

from services.stream_processor.db import ReplayEventStore
from services.stream_processor.processor import process_event


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

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "StreamProcessorSettings":
        source = os.environ if env is None else env
        return cls(
            kafka_broker=source.get("KAFKA_BROKER", "redpanda:9092"),
            raw_topic=source.get("RAW_EVENTS_TOPIC", "raw_events"),
            late_topic=source.get("LATE_EVENTS_TOPIC", "late_events"),
            consumer_group=source.get("STREAM_PROCESSOR_GROUP", "stream-processor"),
            redis_url=source.get("REDIS_URL", "redis://redis:6379/0"),
            postgres_dsn=source.get("POSTGRES_DSN", "postgresql://mlops:mlops@postgres:5432/mlops"),
            session_ttl_seconds=int(source.get("SESSION_TTL_SECONDS", "1800")),
            late_threshold_seconds=int(source.get("LATE_EVENT_THRESHOLD_SECONDS", "60")),
        )


def build_app(settings: StreamProcessorSettings) -> Application:
    application = Application(
        broker_address=settings.kafka_broker,
        consumer_group=settings.consumer_group,
        auto_offset_reset="earliest",
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
    connection = psycopg.connect(settings.postgres_dsn)
    replay_store = ReplayEventStore(connection)
    late_producer = application.get_producer()

    sdf = application.dataframe(raw_topic)
    sdf.apply(
        lambda event: process_event(
            redis_client,
            replay_store,
            late_producer,
            event,
            late_topic=late_topic.name,
            ttl_seconds=settings.session_ttl_seconds,
            late_threshold_seconds=settings.late_threshold_seconds,
        )
    )
    return application


def main() -> int:
    settings = StreamProcessorSettings.from_env()
    app = build_app(settings)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `services/stream_processor/requirements.txt`:

```text
psycopg[binary]>=3.2.0
quixstreams>=3.25.0
redis>=5.0.0
```

Create `services/stream_processor/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY shared ./shared
COPY training ./training
COPY services ./services
COPY services/stream_processor/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -e . && python -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["python", "-m", "services.stream_processor.app"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/tests/test_stream_processor_app.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add services/stream_processor/app.py services/stream_processor/requirements.txt services/stream_processor/Dockerfile services/tests/test_stream_processor_app.py
git commit -m "feat: add stream processor service entrypoint"
```

