# Week 3 Split Plan Self-Review

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the split Week 3 implementation plan traceable to the original spec and verify that no task coverage was lost during decomposition.

**Architecture:** This file is a review artifact for the split plan set. It records the original coverage review and the file-level mapping so execution agents can understand dependencies before starting a child plan.

**Tech Stack:** Markdown, pytest, Ruff, Docker Compose, DVC.

---

## Child Plan Mapping

- `01-event-id-and-simulator.md`: Tasks 1-3, publish-side replay path.
- `02-stream-processor.md`: Tasks 4-6, stateful stream processing and persistence.
- `03-serving-bundle-and-prediction-api.md`: Tasks 7-9, training bundle and API.
- `04-infra-ci-docs-and-verification.md`: Tasks 10-14, compose, CI, docs, and final verification.

## Original Plan Self-Review

- Spec coverage: Tasks 1-3 cover deterministic IDs, raw `event_time` rename, simulator validation, bounded replay, publish keying, and per-session ordering. Tasks 4-6 cover Redis state, TTLs, cache invalidation, duplicate suppression, late routing, PostgreSQL append, and Quix Streams runtime. Tasks 7-9 cover MLflow serving bundle, feature assembly, authentication, API response shape, and fallbacks. Tasks 10-12 cover Redpanda topics, Redis/PostgreSQL/API compose services, and CI checks. Task 13 covers README and blueprint sync.
- Placeholder scan: The plan contains concrete file paths, code snippets, commands, and expected outputs. There are no deferred implementation markers.
- Type consistency: The same event keys are used across simulator, processor, Redis state, PostgreSQL append, and API feature assembly: `event_id`, `user_session`, `source_event_time`, `replay_time`, `event_type`, `product_id`, `user_id`, `category_id`, `category_code`, `brand`, `price`, and `source`.
- Categorical parity check: Task 8 must keep numeric columns numeric and rebuild categorical columns as `pd.Categorical` with `categories=list(category_maps[column].keys())`, matching `training/src/categorical_features.py`; integer category IDs are not the API feature contract.
- Probability-output check: Task 9 must use `predict_proba(row)[:, 1]` for `purchase_probability`; class-label `predict()` output is not acceptable.
- Model-backed smoke check: Task 11 full compose smoke must fail unless `MLFLOW_SERVING_BUNDLE_URI` points at a real smoke-training serving bundle and `/predict` returns `prediction_mode="model"` with `fallback_reason=null`.
- Late-event check: Integration smoke must inject a late event after normal replay state exists. It must not depend on unsorted simulator CSV input, because simulator replay is sorted per session.
- Topic-init check: Task 10 Redpanda init must wait for broker readiness, create `raw_events` and `late_events`, and verify both topics exist instead of silently ignoring topic-creation failures.
