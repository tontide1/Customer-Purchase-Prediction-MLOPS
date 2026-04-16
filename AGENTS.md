# AGENTS

## Working Style (IMPORTANT AND DO NOT TOUCH)
- Ask before guessing when requirements, contracts, or scope are ambiguous.
- Prefer the smallest correct change; do not add speculative abstractions or knobs.
- Keep edits surgical; do not refactor or "improve" unrelated code, docs, or formatting.
- Define success criteria up front and verify them with tests, reads, or direct checks before finishing.

## What Matters Here
- Treat `docs/BLUEPRINT/*.md` and `BLUEPRINT.md` as target-state design; when they conflict with code, trust executable sources.
- Real Week 1 code lives in `training/src/{config,bronze,silver}.py`, `shared/{constants,schemas}.py`, `training/tests/test_data_lake.py`, `training/tests/test_raw_window_selection.py`, and `dvc.yaml`.
- `docker-compose.yml` is MinIO + bucket bootstrap only. There is no repo-level `pyproject.toml`, `requirements*.txt`, `Makefile`, pre-commit config, or CI workflow.
- Run commands from repo root; scripts add the repo root to `sys.path`, so path assumptions are repo-relative.

## Commands
- Prefer `conda activate MLOPS`; if `pytest` is missing, use `conda run -n MLOPS python -m pytest ...`.
- Current working profile: `DEV_SMOKE` (default while iterating locally).
  - Train window: `2019-10` -> `2019-10` (uses TRAINING_WINDOW_START/END env vars)
  - Replay window: `2020-03` -> `2020-03`
  - Keep target-state contracts unchanged; this profile is only for faster dev loops.
- Canonical windows (for production/final evaluation):
  - Training: `2019-10` -> `2020-02`
  - Replay: `2020-03` -> `2020-04`
- Week 1 pipeline:
  - `python training/src/bronze.py --input data/raw --output data/bronze/events.parquet --window-profile dev_smoke`
  - `python training/src/silver.py --input data/bronze/events.parquet --output data/silver/events.parquet`
  - `dvc repro` runs bronze -> silver with DEV_SMOKE window (default).
- MinIO local stack: `docker compose up -d`; verify with `docker compose ps`.
- Focused tests:
  - `conda run -n MLOPS python -m pytest training/tests/test_data_lake.py -v`
  - `conda run -n MLOPS python -m pytest training/tests/test_raw_window_selection.py -v`

## Contracts That Break Easily
- Keep schema column order from `schemas.BRONZE_SCHEMA` / `schemas.SILVER_SCHEMA`; do not derive order from sets.
- Raw layer uses `event_time`; downstream layers use `source_event_time`.
- Bronze ingest only accepts raw files named `YYYY-Mon.csv` or `YYYY-Mon.csv.gz`; unsupported names are skipped.
- Canonical raw window defaults remain `training` (`2019-10` -> `2020-02`) and `replay` (`2020-03` -> `2020-04`); `DEV_SMOKE` is a local override for faster iteration.
- Bronze writes are strict on dtypes: keep IDs/categorical fields string-like before `pa.Table.from_pandas(...)`.

## When Changing Contracts
- Prefer `dvc.yaml`, `training/src/*.py`, `docker-compose.yml`, and scripts over prose.
- If changing data/serving contracts, update `docs/BLUEPRINT/01_OVERVIEW.md`, `02_ARCHITECTURE.md`, `04_PIPELINES.md`, `05_PROJECT_STRUCTURE.md`, and `07_TESTING.md` in the same change.
- Preserve these repo contracts: canonical `event_id`; fail-closed validation gate except first deploy/manual override; `/explain` returns `503` when unavailable; fallback predictions are not cached and are excluded from model-quality metrics; online evaluation stays split by `evaluation_mode`; prediction runtime config must not include DVC/MinIO credentials.
