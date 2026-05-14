# Week 3: Stream Processing Vertical Slice

## Summary

Week 3 will add the first real online path on top of the existing Week 1 and Week 2 foundation. The scope is a staged vertical slice: bounded replay from the November raw window, Kafka-compatible transport, Quix Streams processing, Redis session features, PostgreSQL replay persistence, and a minimal authenticated prediction endpoint that loads a smoke-trained MLflow artifact bundle.

The goal is to prove that replay events can be ingested, transformed into session state, and queried through a model-backed API without pulling in the full Week 4+ serving stack. Explainability, hot reload, rate limiting, dashboards, monitoring, and retraining orchestration stay out of scope.

## Key Changes

### Streaming Path

- Add `services/simulator` to read `data/simulation_raw/2019-Nov.csv.gz`, validate rows, rename raw `event_time` to `source_event_time`, attach `replay_time`, generate deterministic `event_id`, and publish bounded replay batches to a `raw_events` topic.
- The simulator must publish events in `source_event_time` order within each `user_session`.
- Redpanda messages must use `user_session` as the message key so same-session events land on the same partition.
- Add `services/stream-processor` using Quix Streams to consume `raw_events`, deduplicate by `event_id`, route late events to `late_events`, update Redis session state, invalidate cached predictions for that session, and append accepted replay events to PostgreSQL.
- Keep the late-event policy explicit: late events are logged and routed away from the live session state path; they do not update Redis or the append-only processed replay log.
- `late_events` is a separate Kafka topic. It stores the original normalized event payload plus `late_reason`.
- Redpanda topics are created by compose/init automation, not implicit production defaults:
  - `raw_events`: 3 partitions, replication factor 1 for local dev.
  - `late_events`: 3 partitions, replication factor 1 for local dev.
  - Auto-create may remain enabled locally, but tests should not depend on implicit topic creation.
- Event deduplication must use bounded storage. The default strategy is a Redis key `dedup:event:{event_id}` with a TTL matching or exceeding the session-state TTL, rather than one unbounded global set.

### Online State Contract

- Redis session state is stored in a Hash at `session:{user_session}`. This hash is the canonical online feature snapshot for scalar counters, timestamps, and latest event values.
- Exact unique sets live under `session:{user_session}:products` and `session:{user_session}:categories`.
- The `session:{user_session}` hash must include session metadata and event counters needed to match offline gold features:
  - `first_event_time` for `session_duration_sec`.
  - `last_event_time` for ordering and late-event checks.
  - `count_view`.
  - `count_cart`.
  - `count_remove_from_cart`.
- The `session:{user_session}` hash must include latest event values needed to build the model feature vector:
  - `latest_price`.
  - `latest_category_id`.
  - `latest_category_code`.
  - `latest_brand`.
  - `latest_event_type`.
- The processor must maintain the same feature semantics used by Week 2 gold snapshots:
  - numeric: `total_views`, `total_carts`, `net_cart_count`, `cart_to_view_ratio`, `unique_categories`, `unique_products`, `session_duration_sec`, `price`
  - categorical/latest: `category_id`, `category_code`, `brand`, `event_type`
- `net_cart_count` must be computed as `total_carts - total_remove_from_cart`.
- `session_duration_sec` must be computed from the current event `source_event_time` minus `session:{user_session}` field `first_event_time`.
- `unique_categories` must use the same normalization as `training/src/features.py`: use `category_code` when present, otherwise fall back to `category_id`.
- Derived features (`net_cart_count`, `cart_to_view_ratio`, `session_duration_sec`) are computed by the prediction API when assembling the feature vector, using the raw counters and timestamps stored in Redis. The stream processor stores only the raw state; no derived values are persisted in Redis.
- Redis null handling must mirror the training preprocessing contract:
  - `price` null is stored as `latest_price=0`, matching numeric `fillna(0)`.
  - `category_code` null is stored as an empty string in Redis. When assembling the feature vector, the API must pre-process the value: if the value read from Redis is the empty string `""`, replace it with the `missing_token` from the serving bundle's `categorical_encoding.json` (currently `__MISSING__`) before looking up the encoding map.
  - `brand` null is stored as an empty string in Redis. The API applies the same pre-process as `category_code`: empty string → `missing_token` before encoding lookup.
  - `category_id` and `event_type` are required for accepted events; invalid rows missing them are skipped before Redis update.
- Every `session:*` key written for a session must receive a TTL. The default TTL is 30 minutes, configurable by environment variable.
- Every accepted update must delete `cache:predict:session:{user_session}` so the API never serves stale session features.

### PostgreSQL Replay Log

- Accepted replay events are appended to PostgreSQL table `replay_events`.
- The minimum schema is:
  - `event_id` text primary key
  - `user_session` text not null
  - `source_event_time` timestamp not null
  - `replay_time` timestamp not null
  - `event_type` text not null
  - `product_id` text not null
  - `user_id` text not null
  - `category_id` text not null
  - `category_code` text nullable
  - `brand` text nullable
  - `price` double precision nullable
  - `source` text not null
  - `created_at` timestamp not null default current_timestamp
- `source` identifies the event origin. For bounded replay emitted by `services/simulator`, the value is `replay`.
- PostgreSQL is the future retraining export source for accepted replay events. Late events are not inserted into `replay_events`.

### Prediction API

- Add `services/prediction-api` with `GET /health` and `GET /api/v1/predict/{user_session}`.
- Require `X-API-Key` on business endpoints.
- The API key is configured through an `API_KEY` environment variable on the prediction-api container. The value must not be hardcoded or logged.
- The API loads a minimal serving bundle from MLflow produced by a training smoke run, not from a mocked object and not from the full registry promotion flow.
- Successful predict responses must include `purchase_probability`, `prediction_time`, `prediction_horizon_minutes`, `model_uri`, `model_version`, `prediction_mode="model"`, `fallback_reason=null`, and `cached=false`.
- Fallback behavior is explicit:
  - Redis miss returns `200` with score `0.5` and `fallback_reason="redis_miss"`.
  - Model load or prediction failure returns `200` with score `0.5` and `fallback_reason="model_unavailable"`.
  - Invalid `user_session` returns `422`.
  - Missing or invalid API key returns `401`.

### Packaging And Infra

- Add per-service `requirements.txt` files and Dockerfiles for the Week 3 services.
- Extend `docker-compose.yml` with Redpanda, Redis, PostgreSQL, the simulator, the stream processor, and the prediction API.
- Keep the existing MinIO + MLflow artifact-store setup from Week 2 unchanged.
- Extend the Week 2 training artifact bundle so the API can load a real model contract without adding full model registry promotion or hot reload yet.

### Training Artifact Bundle

- Extend the training pipeline to log an explicit serving bundle as MLflow artifacts for the selected winner. Current model logging is not enough for the API because it does not independently preserve the feature column order, categorical maps, threshold, and horizon contract. **This is a cross-cutting change: Week 3 must modify `training/src/train.py` to log the serving assets so the API can consume them.**
- The serving bundle artifacts are logged during the `{winner_name}_test_evaluation` MLflow run using the already-available `data.categorical_columns`, `data.numeric_columns`, `data.categorical_artifacts` (`CategoricalEncodingArtifacts`), and `winner_threshold` from the existing training orchestration.
- The serving bundle must be logged on the existing `{winner_name}_test_evaluation` MLflow run so the API has one canonical winner run to inspect. Do not create a separate `{winner_name}_serving_bundle` run in Week 3.
- The serving bundle must include:
  - `serving/model_metadata.json`: model type/name, run ID, model URI, and artifact path.
  - `serving/feature_column_order.json`: `NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS`.
  - `serving/categorical_encoding.json`: `CategoricalEncodingArtifacts.category_maps` plus missing/unknown token metadata.
  - `serving/threshold.json`: selected winner `optimal_threshold`.
  - `serving/prediction_contract.json`: `PREDICTION_HORIZON_MINUTES` and `response_contract_version`.
- `response_contract_version` is the fixed string `v1` for Week 3. The API must reject bundles with a different response contract version.
- This is a minimal bundle for API consumption. It is not a model registry promotion workflow and does not introduce hot reload.

## Test Plan

- Unit tests should cover:
  - deterministic `event_id`
  - raw `event_time` renamed to `source_event_time` before hashing
  - simulator row validation
  - bounded replay behavior
  - per-session publish ordering by `source_event_time`
  - duplicate event suppression
  - late-event routing
  - Redis feature-state math parity with existing gold semantics
  - `total_remove_from_cart`, `first_event_time`, latest feature fields, TTL, null handling, and normalized category semantics
  - PostgreSQL append behavior
  - MLflow serving bundle artifact contents
  - API key validation
  - Redis miss fallback
  - model-unavailable fallback
  - model-backed prediction response shape

- Compose integration smoke in CI should:
  - install root and service dependencies
  - verify `data/simulation_raw/2019-Nov.csv.gz` exists for local runs or provision a small CI CSV fixture with the same raw input schema as `2019-Nov.csv.gz`
  - the CI fixture must include at least two sessions, at least one `view`, `cart`, `remove_from_cart`, and `purchase`, at least one nullable `category_code` or `brand`, and at least one out-of-order event that triggers `late_events`
  - start the full local stack
  - run a training smoke to create the MLflow serving bundle
  - run bounded replay through the simulator and processor
  - verify Redis state, PostgreSQL append log, and `late_events`
  - call authenticated `GET /api/v1/predict/{user_session}` and verify a model-backed response

- Existing repository checks should continue to run:
  - `ruff check .`
  - `pytest training/tests -q`
  - service tests
  - compose smoke test
  - `dvc dag`

## Assumptions

- Redpanda will be used as the Kafka-compatible local broker.
- Quix Streams will be used for the processor implementation.
- Replay defaults to a fast bounded mode with `--limit`; timestamp-paced replay is out of scope for this week.
- Late events are considered events that fall behind the session’s latest accepted `source_event_time` by more than the configured threshold, defaulting to 60 seconds.
- PostgreSQL is only the replay append log for future retraining export, not the retraining pipeline itself.
- The API uses the smoke-trained MLflow artifact bundle; full model registry promotion and hot reload are deferred to the next week.
- `2019-Nov.csv.gz` must be placed under `data/simulation_raw/` before local Week 3 replay work begins. Documentation should keep the existing Kaggle download guidance in sync with this prerequisite.
