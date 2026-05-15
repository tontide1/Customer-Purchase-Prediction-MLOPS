# Week 3 Stream Processing Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Week 3 online vertical slice: bounded November replay, Kafka-compatible transport, Quix Streams processing, Redis online session state, PostgreSQL replay persistence, and an authenticated model-backed prediction endpoint.

**Architecture:** This index keeps the original Week 3 plan navigable while the executable work is split into smaller child plans. Execute the child files in order unless a later task is explicitly independent; each child keeps the same test-first, bite-sized steps from the original plan.

**Tech Stack:** Python 3.11, pandas, Quix Streams, Redpanda, Redis, PostgreSQL, psycopg, FastAPI, Uvicorn, MLflow, scikit-learn model APIs, pytest, Ruff, Docker Compose.

---

## Source Spec

- Spec: `docs/superpowers/specs/2026-05-15-week3-stream-processing-design.md`
- Split plan directory: `docs/superpowers/plans/2026-05-15-week3-stream-processing-vertical-slice/`

## Execution Order

1. `docs/superpowers/plans/2026-05-15-week3-stream-processing-vertical-slice/01-event-id-and-simulator.md`
   - Tasks 1-3: shared deterministic `event_id`, simulator normalization, bounded ordering, and Kafka publish.
2. `docs/superpowers/plans/2026-05-15-week3-stream-processing-vertical-slice/02-stream-processor.md`
   - Tasks 4-6: Redis state semantics, deduplication, late routing, PostgreSQL append, and Quix entrypoint.
3. `docs/superpowers/plans/2026-05-15-week3-stream-processing-vertical-slice/03-serving-bundle-and-prediction-api.md`
   - Tasks 7-9: MLflow serving bundle, Redis feature assembly, authenticated prediction API, and fallbacks.
4. `docs/superpowers/plans/2026-05-15-week3-stream-processing-vertical-slice/04-infra-ci-docs-and-verification.md`
   - Tasks 10-14: compose, topic creation, smoke helper, CI, docs, and final verification.
5. `docs/superpowers/plans/2026-05-15-week3-stream-processing-vertical-slice/05-self-review.md`
   - Coverage and consistency review for the split plan set.

## Dependency Notes

- The simulator child plan can be implemented first because downstream services consume its normalized event contract.
- The stream processor child plan depends on `shared/event_id.py` only through event payloads, not imports, but should follow the simulator plan to keep event keys stable.
- The prediction API child plan depends on the serving bundle task in `training/src/train.py` and the Redis state contract from the stream processor plan.
- The infra/docs child plan should run after service files exist, because compose, CI, README, and blueprint updates reference those paths.

## Original File Structure

- `shared/event_id.py`: canonical deterministic event ID helper used by simulator tests and any later replay export code.
- `services/__init__.py`: makes service modules importable in tests.
- `services/simulator/replay.py`: raw CSV row validation, `event_time` to `source_event_time` normalization, bounded replay iteration, per-session ordering, and publish helper.
- `services/simulator/app.py`: CLI entrypoint for bounded replay into `raw_events`.
- `services/simulator/requirements.txt`: simulator container dependencies.
- `services/simulator/Dockerfile`: simulator image.
- `services/stream_processor/state.py`: Redis session-state update and prediction-cache invalidation.
- `services/stream_processor/db.py`: PostgreSQL `replay_events` append helper and schema bootstrap.
- `services/stream_processor/processor.py`: duplicate suppression, late-event policy, Redis update, PostgreSQL append, and late-topic routing.
- `services/stream_processor/app.py`: Quix Streams consumer/processor entrypoint.
- `services/stream_processor/requirements.txt`: stream processor dependencies.
- `services/stream_processor/Dockerfile`: stream processor image.
- `services/prediction_api/bundle.py`: MLflow serving bundle loading and contract validation.
- `services/prediction_api/features.py`: Redis hash to model feature vector assembly.
- `services/prediction_api/app.py`: FastAPI health and prediction endpoints.
- `services/prediction_api/requirements.txt`: API dependencies.
- `services/prediction_api/Dockerfile`: API image.
- `services/tests/`: focused service unit tests plus a small compose smoke test.
- `infra/redpanda/init-topics.sh`: explicit local topic creation for `raw_events` and `late_events`.
- `infra/postgres/init.sql`: `replay_events` table schema.
- `scripts/week3_compose_smoke.py`: deterministic end-to-end smoke check against the local compose stack.
- `training/src/train.py`: log the Week 3 serving bundle during the winner test-evaluation MLflow run.
- `training/tests/test_train.py`: validate serving bundle artifact contents.
- `pyproject.toml`: include `services*` packages and service tests in pytest config.
- `docker-compose.yml`: add Redpanda, Redis, PostgreSQL, simulator, stream processor, and prediction API while keeping existing MinIO and MLflow unchanged.
- `.env.example`: document Week 3 non-secret defaults, including `API_KEY`.
- `.github/workflows/ci.yml`: install service dependencies and run the lightweight service tests plus compose config validation.
- `README.md`: add Week 3 runbook commands.
- `docs/BLUEPRINT/01_OVERVIEW.md`, `docs/BLUEPRINT/02_ARCHITECTURE.md`, `docs/BLUEPRINT/04_PIPELINES.md`, `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`, `docs/BLUEPRINT/07_TESTING.md`: sync architecture docs because this changes serving and online data contracts.
