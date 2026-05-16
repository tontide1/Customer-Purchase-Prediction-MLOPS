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

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def expire(self, key, ttl_seconds):
        self.ttls[key] = ttl_seconds

    def delete(self, key):
        self.hashes.pop(key, None)
        self.sets.pop(key, None)
        self.ttls.pop(key, None)
        self.deleted.append(key)


class _RecordingPipeline:
    def __init__(self, backend_redis):
        self._backend = backend_redis
        self.commands = []
        self.execute_called = False

    def hset(self, name, mapping=None, **kwargs):
        if mapping is None:
            mapping = kwargs
        self.commands.append(("hset", name, mapping))
        return self

    def sadd(self, name, *values):
        self.commands.append(("sadd", name, set(values)))
        return self

    def expire(self, name, ttl_seconds):
        self.commands.append(("expire", name, ttl_seconds))
        return self

    def delete(self, *names):
        for name in names:
            self.commands.append(("delete", name))
        return self

    def execute(self):
        self.execute_called = True
        for op, *args in self.commands:
            if op == "hset":
                name, mapping = args
                self._backend.hset(name, mapping)
            elif op == "sadd":
                name, values = args
                for value in values:
                    self._backend.sadd(name, value)
            elif op == "expire":
                name, ttl_seconds = args
                self._backend.expire(name, ttl_seconds)
            elif op == "delete":
                (name,) = args
                self._backend.delete(name)
        return True


class FakeRedisWithPipeline(FakeRedis):
    def __init__(self):
        super().__init__()
        self._pipeline = _RecordingPipeline(self)

    def pipeline(self, transaction=True):
        return self._pipeline


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
        "first_event_time": "2019-11-01T00:00:00+00:00",
        "last_event_time": "2019-11-01T00:00:00+00:00",
        "count_view": "1",
        "count_cart": "0",
        "count_remove_from_cart": "0",
        "latest_price": "0",
        "latest_category_id": "cat-id",
        "latest_category_code": "",
        "latest_brand": "",
        "latest_event_type": "view",
        "serving_total_views": "0",
        "serving_total_carts": "0",
        "serving_total_removes": "0",
        "serving_net_cart_count": "0",
        "serving_cart_to_view_ratio": "0.0",
        "serving_unique_categories": "0",
        "serving_unique_products": "0",
        "serving_session_duration_sec": "0.0",
        "serving_price": "0",
        "serving_category_id": "cat-id",
        "serving_category_code": "",
        "serving_brand": "",
        "serving_event_type": "view",
    }
    assert redis.sets["session:session-1:products"] == {"100"}
    assert redis.sets["session:session-1:categories"] == {"cat-id"}
    assert redis.ttls["session:session-1"] == 1800
    assert redis.ttls["session:session-1:products"] == 1800
    assert redis.ttls["session:session-1:categories"] == 1800
    assert redis.deleted == ["cache:predict:session:session-1"]


def test_apply_event_updates_counts_preserves_first_event_and_normalizes_category_set():
    redis = FakeRedis()

    apply_event_to_session_state(
        redis, _event(event_id="e1", event_type="view"), ttl_seconds=1800
    )
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
            source_event_time="2019-11-01T00:02:00+00:00",
        ),
        ttl_seconds=1800,
    )
    apply_event_to_session_state(
        redis,
        _event(
            event_id="e3",
            event_type="remove_from_cart",
            product_id="200",
            source_event_time="2019-11-01T00:03:00+00:00",
        ),
        ttl_seconds=1800,
    )

    state = redis.hashes["session:session-1"]
    assert state["first_event_time"] == "2019-11-01T00:00:00+00:00"
    assert state["last_event_time"] == "2019-11-01T00:03:00+00:00"
    assert state["count_view"] == "1"
    assert state["count_cart"] == "1"
    assert state["count_remove_from_cart"] == "1"
    assert state["latest_price"] == "0"
    assert state["latest_category_code"] == ""
    assert state["latest_brand"] == ""
    assert state["serving_total_views"] == "1"
    assert state["serving_total_carts"] == "1"
    assert state["serving_total_removes"] == "0"
    assert state["serving_net_cart_count"] == "1"
    assert state["serving_cart_to_view_ratio"] == "1.0"
    assert state["serving_unique_categories"] == "2"
    assert state["serving_unique_products"] == "2"
    assert state["serving_session_duration_sec"] == "180.0"
    assert state["serving_price"] == "0"
    assert state["serving_category_id"] == "cat-id"
    assert state["serving_category_code"] == ""
    assert state["serving_brand"] == ""
    assert state["serving_event_type"] == "remove_from_cart"
    assert redis.sets["session:session-1:categories"] == {"cat-id", "category.code"}


def test_apply_event_uses_pipeline_and_queues_correct_operations():
    redis = FakeRedisWithPipeline()

    apply_event_to_session_state(redis, _event(), ttl_seconds=1800)

    pipeline = redis._pipeline
    expected_hash = {
        "first_event_time": "2019-11-01T00:00:00+00:00",
        "last_event_time": "2019-11-01T00:00:00+00:00",
        "count_view": "1",
        "count_cart": "0",
        "count_remove_from_cart": "0",
        "latest_price": "0",
        "latest_category_id": "cat-id",
        "latest_category_code": "",
        "latest_brand": "",
        "latest_event_type": "view",
        "serving_total_views": "0",
        "serving_total_carts": "0",
        "serving_total_removes": "0",
        "serving_net_cart_count": "0",
        "serving_cart_to_view_ratio": "0.0",
        "serving_unique_categories": "0",
        "serving_unique_products": "0",
        "serving_session_duration_sec": "0.0",
        "serving_price": "0",
        "serving_category_id": "cat-id",
        "serving_category_code": "",
        "serving_brand": "",
        "serving_event_type": "view",
    }

    assert pipeline.execute_called is True
    assert ("hset", "session:session-1", expected_hash) in pipeline.commands
    assert ("sadd", "session:session-1:products", {"100"}) in pipeline.commands
    assert ("sadd", "session:session-1:categories", {"cat-id"}) in pipeline.commands
    assert ("expire", "session:session-1", 1800) in pipeline.commands
    assert ("delete", "cache:predict:session:session-1") in pipeline.commands
    assert redis.hashes["session:session-1"] == expected_hash
    assert redis.ttls["session:session-1"] == 1800
