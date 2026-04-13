# AGENTS

## Enviroment Python
- Use command "conda activate MLOPS" for python enviroment
- Present, the version python is 3.11.15

## Repo Reality (verify before acting)
- This repository is still blueprint-first: most `docs/BLUEPRINT/*.md` snippets are target-state examples, not implemented code.
- Currently present executable/scaffold files are limited to `docker-compose.yml` (MinIO + bucket init only), `dvc.yaml`, `infra/minio/init-bucket.sh`, `.env.example`, datasets, and notebooks.
- There is no repo-level `pyproject.toml`, `requirements*.txt`, `Makefile`, or CI workflow; do not invent lint/typecheck/test commands.

## Commands That Are Safe To Assume
- `docker compose up -d` from repo root starts the MinIO scaffold only.
- `docker compose ps` is the quick health check; MinIO uses ports `9000` (S3 API) and `9001` (console).
- `dvc.yaml` exists, but `dvc repro` stages depend on `training/src/*.py` scripts that are not fully implemented yet.

## High-Risk Contracts (Do Not Drift)
- Canonical `event_id`: `hash(f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}")`.
- Validation gate is fail-closed except first deployment; manual override requires all audit fields: `override_by`, `override_reason`, `override_time`.
- `/predict` may fallback; `/explain` must return `503` when explainer is unavailable.
- Fallback predictions must not be cached and must be excluded from model-quality metrics.
- Keep online evaluation split by `evaluation_mode` (`demo_replay` vs `offline_backfill`); never merge them into one metric series.
- DVC+MinIO is source of truth for data artifacts (`raw/bronze/silver/gold`); MLflow is source of truth for model registry/experiment metrics.
- Least privilege: prediction API config must not contain DVC/MinIO credentials.

## Editing Guardrails
- Prefer updating executable sources (`docker-compose.yml`, `dvc.yaml`, scripts) before prose if they conflict.
- When changing data/serving contracts, update the matching docs together: `01_OVERVIEW.md`, `02_ARCHITECTURE.md`, `04_PIPELINES.md`, `05_PROJECT_STRUCTURE.md`, `07_TESTING.md`.

## Karpathy Guidelines

### 1. Think Before Coding
- State assumptions explicitly. If uncertain, ask.
- Present multiple interpretations when ambiguous; don't pick silently.
- Push back when a simpler approach exists.
- Stop and ask when something is unclear.

### 2. Simplicity First
- Minimum code that solves the problem. No speculative abstractions.
- No features beyond what was asked.
- No "flexibility" that wasn't requested.
- If 200 lines could be 50, rewrite it.

### 3. Surgical Changes
- Touch only what you must.
- Don't "improve" adjacent code or refactor unrelated broken things.
- Match existing style.
- Remove only orphans your changes created.

### 4. Goal-Driven Execution
- Transform tasks into verifiable goals.
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then fix"
- For multi-step tasks: state plan with verification checkpoints.

## Guardrails for future edits
- Keep reject reasons machine-readable and sourced from `RejectReasonCode`.
- Do not treat blueprint examples as already implemented services; verify against actual files before coding.
- There is currently no repo-level `pyproject.toml`, `requirements*.txt`, CI workflow, or `Makefile`; do not assume lint/typecheck commands exist.
