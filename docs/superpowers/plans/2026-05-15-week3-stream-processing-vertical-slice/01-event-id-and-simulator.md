# Week 3 Event ID And Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic replay event ID helper and bounded raw-event simulator that publishes keyed replay events to `raw_events`.

**Architecture:** This plan owns the first publish-side increment of the Week 3 slice. It creates a shared event-id helper, normalizes raw November CSV rows into the online event contract, and adds a simulator CLI/container that publishes events with `user_session` as the Kafka key.

**Tech Stack:** Python 3.11, pandas, Quix Streams producer APIs, pytest, Docker.

---

### Task 1: Shared Deterministic Event ID

**Files:**
- Create: `shared/event_id.py`
- Test: `services/tests/test_event_id.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add service tests to pytest discovery**

Update `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["shared*", "training*", "services*"]

[tool.pytest.ini_options]
testpaths = ["training/tests", "services/tests"]
```

- [ ] **Step 2: Write the failing test**

Create `services/tests/test_event_id.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest services/tests/test_event_id.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'shared.event_id'`.

- [ ] **Step 4: Write minimal implementation**

Create `shared/event_id.py`:

```python
"""Canonical deterministic event identifiers for offline and online events."""

from __future__ import annotations

import hashlib


def compute_event_id(
    *,
    user_session: str,
    source_event_time: str,
    event_type: str,
    product_id: str,
    user_id: str,
) -> str:
    payload = f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest services/tests/test_event_id.py -q`

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml shared/event_id.py services/tests/test_event_id.py
git commit -m "feat: add deterministic replay event ids"
```

### Task 2: Simulator Row Normalization And Validation

**Files:**
- Create: `services/__init__.py`
- Create: `services/simulator/__init__.py`
- Create: `services/simulator/replay.py`
- Test: `services/tests/test_simulator_replay.py`

- [ ] **Step 1: Write the failing tests**

Create `services/tests/test_simulator_replay.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/tests/test_simulator_replay.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'services.simulator'`.

- [ ] **Step 3: Write minimal implementation**

Create `services/__init__.py`:

```python
"""Week 3 service packages."""
```

Create `services/simulator/__init__.py`:

```python
"""Bounded raw replay simulator."""
```

Create `services/simulator/replay.py`:

```python
"""Normalize and replay raw November events into Kafka-compatible topics."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from shared import constants
from shared.event_id import compute_event_id

RAW_REQUIRED_FIELDS = (
    constants.FIELD_EVENT_TIME,
    "event_type",
    "product_id",
    constants.FIELD_CATEGORY_ID,
    "user_id",
    "user_session",
)

RAW_COLUMNS = [
    constants.FIELD_EVENT_TIME,
    "event_type",
    "product_id",
    constants.FIELD_CATEGORY_ID,
    "user_id",
    "user_session",
    "category_code",
    "brand",
    "price",
]


def _is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _as_text(value: Any) -> str:
    return str(value)


def _normalize_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.isoformat()


def _nullable_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


def _nullable_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def normalize_raw_row(row: dict[str, Any], replay_time: str | None = None) -> dict[str, Any]:
    for field in RAW_REQUIRED_FIELDS:
        if field not in row or _is_missing(row[field]) or str(row[field]) == "":
            raise ValueError(f"Missing required raw field: {field}")

    event_type = _as_text(row["event_type"])
    if event_type not in constants.ALLOWED_EVENT_TYPES:
        raise ValueError(f"Invalid event_type: {event_type}")

    source_event_time = _normalize_timestamp(row[constants.FIELD_EVENT_TIME])
    normalized = {
        constants.FIELD_SOURCE_EVENT_TIME: source_event_time,
        "event_type": event_type,
        "product_id": _as_text(row["product_id"]),
        constants.FIELD_CATEGORY_ID: _as_text(row[constants.FIELD_CATEGORY_ID]),
        "user_id": _as_text(row["user_id"]),
        "user_session": _as_text(row["user_session"]),
        "category_code": _nullable_text(row.get("category_code")),
        "brand": _nullable_text(row.get("brand")),
        "price": _nullable_float(row.get("price")),
        "replay_time": replay_time or dt.datetime.utcnow().replace(microsecond=0).isoformat(),
        "source": "replay",
    }
    normalized["event_id"] = compute_event_id(
        user_session=normalized["user_session"],
        source_event_time=normalized[constants.FIELD_SOURCE_EVENT_TIME],
        event_type=normalized["event_type"],
        product_id=normalized["product_id"],
        user_id=normalized["user_id"],
    )
    return normalized


def iter_replay_events(
    csv_path: str | Path,
    *,
    limit: int,
    replay_time: str | None = None,
) -> Iterable[dict[str, Any]]:
    frame = pd.read_csv(csv_path, usecols=RAW_COLUMNS, nrows=limit)
    frame[constants.FIELD_EVENT_TIME] = pd.to_datetime(frame[constants.FIELD_EVENT_TIME], utc=True)
    frame = frame.sort_values(["user_session", constants.FIELD_EVENT_TIME], kind="mergesort")
    for row in frame.to_dict(orient="records"):
        yield normalize_raw_row(row, replay_time=replay_time)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/tests/test_simulator_replay.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add services/__init__.py services/simulator/__init__.py services/simulator/replay.py services/tests/test_simulator_replay.py
git commit -m "feat: normalize bounded replay rows"
```

### Task 3: Simulator Bounded Ordering And Kafka Publish

**Files:**
- Modify: `services/simulator/replay.py`
- Create: `services/simulator/app.py`
- Create: `services/simulator/requirements.txt`
- Create: `services/simulator/Dockerfile`
- Test: `services/tests/test_simulator_publish.py`

- [ ] **Step 1: Write the failing tests**

Create `services/tests/test_simulator_publish.py`:

```python
"""Tests for simulator replay ordering and publishing."""

from __future__ import annotations

import pandas as pd

from services.simulator.replay import iter_replay_events, publish_events


class FakeProducer:
    def __init__(self):
        self.messages = []

    def produce(self, *, topic, key, value):
        self.messages.append({"topic": topic, "key": key, "value": value})

    def flush(self):
        self.flushed = True


def test_iter_replay_events_is_bounded_and_sorted_within_session(tmp_path):
    csv_path = tmp_path / "2019-Nov.csv.gz"
    pd.DataFrame(
        [
            {
                "event_time": "2019-11-01 00:03:00 UTC",
                "event_type": "cart",
                "product_id": "2",
                "category_id": "20",
                "user_id": "9",
                "user_session": "s1",
                "category_code": "cat.b",
                "brand": "b",
                "price": 2.0,
            },
            {
                "event_time": "2019-11-01 00:01:00 UTC",
                "event_type": "view",
                "product_id": "1",
                "category_id": "10",
                "user_id": "9",
                "user_session": "s1",
                "category_code": "cat.a",
                "brand": "a",
                "price": 1.0,
            },
            {
                "event_time": "2019-11-01 00:02:00 UTC",
                "event_type": "view",
                "product_id": "3",
                "category_id": "30",
                "user_id": "8",
                "user_session": "s2",
                "category_code": "cat.c",
                "brand": "c",
                "price": 3.0,
            },
        ]
    ).to_csv(csv_path, index=False, compression="gzip")

    events = list(iter_replay_events(csv_path, limit=2, replay_time="2026-05-15T09:00:00"))

    assert len(events) == 2
    assert [event["source_event_time"] for event in events if event["user_session"] == "s1"] == [
        "2019-11-01T00:01:00",
        "2019-11-01T00:03:00",
    ]


def test_publish_events_uses_user_session_as_key():
    producer = FakeProducer()
    event = {
        "event_id": "event-1",
        "user_session": "session-1",
        "source_event_time": "2019-11-01T00:00:00",
    }

    count = publish_events([event], producer=producer, topic="raw_events")

    assert count == 1
    assert producer.messages == [
        {"topic": "raw_events", "key": "session-1", "value": event}
    ]
    assert producer.flushed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/tests/test_simulator_publish.py -q`

Expected: fail with `ImportError` for `publish_events`.

- [ ] **Step 3: Add publish helper and CLI**

Append to `services/simulator/replay.py`:

```python
def publish_events(
    events: Iterable[dict[str, Any]],
    *,
    producer,
    topic: str,
) -> int:
    count = 0
    for event in events:
        producer.produce(topic=topic, key=event["user_session"], value=event)
        count += 1
    producer.flush()
    return count
```

Create `services/simulator/app.py`:

```python
"""CLI for bounded November replay into the raw_events topic."""

from __future__ import annotations

import argparse
import logging
import os

from quixstreams import Application

from services.simulator.replay import iter_replay_events, publish_events

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.getenv("SIMULATION_RAW_DATA_PATH", "data/simulation_raw/2019-Nov.csv.gz"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("REPLAY_LIMIT", "1000")))
    parser.add_argument("--broker", default=os.getenv("KAFKA_BROKER", "redpanda:9092"))
    parser.add_argument("--topic", default=os.getenv("RAW_EVENTS_TOPIC", "raw_events"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    app = Application(broker_address=args.broker)
    app.topic(args.topic, value_serializer="json", key_serializer="str")
    events = iter_replay_events(args.input, limit=args.limit)
    with app.get_producer() as producer:
        count = publish_events(events, producer=producer, topic=args.topic)
    logger.info("Published %d replay events to %s", count, args.topic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `services/simulator/requirements.txt`:

```text
pandas>=2.2.0
quixstreams>=3.25.0
```

Create `services/simulator/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY shared ./shared
COPY services ./services
COPY services/simulator/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -e . && python -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["python", "-m", "services.simulator.app"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/tests/test_simulator_publish.py services/tests/test_simulator_replay.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/simulator/replay.py services/simulator/app.py services/simulator/requirements.txt services/simulator/Dockerfile services/tests/test_simulator_publish.py
git commit -m "feat: publish bounded replay events"
```

