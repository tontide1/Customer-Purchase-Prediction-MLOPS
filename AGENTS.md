# AGENTS

## Core Working Rules (STRICT AND DO NOT TOUCH)

- **Minimal Impact:** Make the smallest correct change. NEVER refactor unrelated code, formatting, or docs.
- **No Guessing:** If requirements, contracts, or scope are ambiguous, stop and ask.
- **Verify First:** Prove your changes work via direct reads, checks, or tests before finishing.
- **Keep It Simple:** Do not over-engineer. Write straightforward solutions.

## What is real in this repo (today)

- Treat `docs/BLUEPRINT/*.md` and `BLUEPRINT.md` as target-state design; many snippets are illustrative, not executable.
- Executable Week 1 data foundation exists in:
  - `training/src/config.py`, `training/src/bronze.py`, `training/src/silver.py`
  - `shared/constants.py`, `shared/schemas.py`
  - `training/tests/test_data_lake.py`, `training/tests/test_raw_window_selection.py`, `training/tests/test_silver_dataset_io.py`
  - `dvc.yaml` (only `bronze` and `silver` stages)
- Infra scaffold currently implemented: `docker-compose.yml` + `infra/minio/init-bucket.sh` (MinIO + bucket init only).
- Repo-level packaging/tooling now exists:
  - `pyproject.toml` supports editable install with `shared*` and `training*`, runtime deps, dev deps, pytest, Ruff, and mypy config.
  - `.pre-commit-config.yaml` runs hygiene hooks plus `ruff-check` and `ruff-format`.
  - `.github/workflows/ci.yml` runs Python 3.11/3.12 matrix checks: install, Ruff, pytest, and `dvc dag`.
  - `README.md` is the top-level quick-start for new users.
- There is still no repo-level `Makefile` or `requirements*.txt`.

## Environment and command baseline

- Preferred Python env: `conda activate MLOPS` (expected Python 3.11.x in this env).
- Install package/dev tooling with `python -m pip install -e ".[dev]"`.
- If `python` is unavailable outside conda, use `python3` for scripts.
- Start local object storage: `docker compose up -d` (MinIO only).
- Quick health check: `docker compose ps` (MinIO ports `9000` API, `9001` console).
- Week 1 pipeline commands:
  - `python -m training.src.bronze --input data/train_raw --output data/bronze/events.parquet`
  - `python -m training.src.silver --input data/bronze/events.parquet --output data/silver/events.parquet`
  - `dvc repro` runs only `bronze` -> `silver` per current `dvc.yaml`.
- Use module-mode commands (`python -m training.src...`) rather than direct script paths. The old `sys.path.insert(...)` import hacks were removed from `training/src` and tests.
- Common verification commands:
  - `ruff check .`
  - `pytest training/tests -q`
  - `dvc dag`
  - `PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit run --all-files` when the default home cache is not writable.

## Data pipeline gotchas that cause real failures

- Keep schema field order from `schemas.BRONZE_SCHEMA` / `schemas.SILVER_SCHEMA` when selecting columns; do not build column order from sets.
- Bronze write path is strict on dtypes (PyArrow schema cast): ensure IDs/categorical fields stay string-like before `pa.Table.from_pandas(...)`.
- Bronze input reader supports both `*.csv` and `*.csv.gz` under `data/train_raw/`; baseline training is intentionally scoped to `2019-Oct.csv.gz` and must not mix in `data/simulation_raw/`.
- Raw layer contract: keep source `event_time`; internal layers must use `source_event_time`.
- Silver supports parquet file input and parquet dataset directory input.
- Silver clean phase reads bronze in batches. Final global dedup/sort is implemented as external sort + k-way merge over temporary parquet runs, so `finalize_silver_parts` must not be changed back to `read_bronze_parquet(parts_path)` / full pandas load.
- Silver duplicate semantics are locked: canonical duplicate keys keep the first surviving record in bronze input order. The temporary `_silver_input_order` column exists only to preserve this during external merge and must not be written to the final `SILVER_SCHEMA` output.

## Contracts to preserve when touching architecture/docs

- Canonical `event_id`: `hash(f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}")`.
- Validation gate is fail-closed except first deployment; manual override requires `override_by`, `override_reason`, `override_time`.
- `/predict` may fallback; `/explain` must return `503` when explainer is unavailable.
- Fallback predictions must not be cached and must be excluded from model-quality metrics.
- Keep online evaluation separated by `evaluation_mode` (`demo_replay` vs `offline_backfill`); never merge metric series.
- Least privilege rule: prediction API runtime config must not include DVC/MinIO credentials.

## Editing policy for agents

- Prefer executable source of truth (`dvc.yaml`, `training/src/*.py`, `docker-compose.yml`, scripts) over prose if they conflict.
- When changing data/serving contracts, sync these blueprint docs in the same change:
  - `docs/BLUEPRINT/01_OVERVIEW.md`
  - `docs/BLUEPRINT/02_ARCHITECTURE.md`
  - `docs/BLUEPRINT/04_PIPELINES.md`
  - `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
  - `docs/BLUEPRINT/07_TESTING.md`
