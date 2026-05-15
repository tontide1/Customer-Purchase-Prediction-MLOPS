"""Tests for simulator replay ordering and publishing."""

from __future__ import annotations

import pandas as pd

from services.simulator.replay import iter_replay_events, publish_events


class FakeProducer:
    def __init__(self):
        self.messages = []
        self.flush_count = 0

    def produce(self, *, topic, key, value):
        self.messages.append({"topic": topic, "key": key, "value": value})

    def flush(self):
        self.flush_count += 1
        self.flushed = True


class FakeSerializedMessage:
    def __init__(self, *, key, value):
        self.key = key
        self.value = value


class FakeTopic:
    name = "raw_events"

    def serialize(self, *, key, value):
        return FakeSerializedMessage(
            key=f"serialized-key:{key}",
            value={"serialized": value},
        )


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

    events = list(
        iter_replay_events(csv_path, limit=2, replay_time="2026-05-15T09:00:00")
    )

    assert len(events) == 2
    assert [
        event["source_event_time"] for event in events if event["user_session"] == "s1"
    ] == [
        "2019-11-01T00:01:00",
        "2019-11-01T00:03:00",
    ]


def test_publish_events_serializes_key_with_quix_topic():
    producer = FakeProducer()
    event = {
        "event_id": "event-1",
        "user_session": "session-1",
        "source_event_time": "2019-11-01T00:00:00",
    }

    count = publish_events([event], producer=producer, topic=FakeTopic())

    assert count == 1
    assert producer.messages == [
        {
            "topic": "raw_events",
            "key": "serialized-key:session-1",
            "value": {"serialized": event},
        }
    ]
    assert producer.flush_count == 1
    assert producer.flushed is True


def test_publish_events_rejects_raw_string_topic():
    producer = FakeProducer()
    event = {
        "event_id": "event-1",
        "user_session": "session-1",
        "source_event_time": "2019-11-01T00:00:00",
    }

    try:
        publish_events([event], producer=producer, topic="raw_events")
    except TypeError as exc:
        assert str(exc) == "topic must provide .name and .serialize(key=..., value=...)"
    else:
        raise AssertionError("publish_events() did not reject a raw string topic")


def test_publish_events_rejects_raw_string_topic_without_events():
    producer = FakeProducer()

    try:
        publish_events([], producer=producer, topic="raw_events")
    except TypeError as exc:
        assert str(exc) == "topic must provide .name and .serialize(key=..., value=...)"
    else:
        raise AssertionError("publish_events() did not reject a raw string topic")


def test_publish_events_serializes_with_quix_topic_and_flushes_once():
    producer = FakeProducer()
    events = [
        {
            "event_id": "event-1",
            "user_session": "session-1",
            "source_event_time": "2019-11-01T00:00:00",
        },
        {
            "event_id": "event-2",
            "user_session": "session-2",
            "source_event_time": "2019-11-01T00:01:00",
        },
    ]

    count = publish_events(events, producer=producer, topic=FakeTopic())

    assert count == 2
    assert producer.messages == [
        {
            "topic": "raw_events",
            "key": "serialized-key:session-1",
            "value": {"serialized": events[0]},
        },
        {
            "topic": "raw_events",
            "key": "serialized-key:session-2",
            "value": {"serialized": events[1]},
        },
    ]
    assert producer.flush_count == 1
