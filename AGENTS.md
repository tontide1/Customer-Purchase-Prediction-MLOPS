# AGENTS

## Core Working Rules (STRICT AND DO NOT TOUCH)

- **Minimal Impact:** Make the smallest correct change. NEVER refactor unrelated code, formatting, or docs.
- **No Guessing:** If requirements, contracts, or scope are ambiguous, stop and ask.
- **Verify First:** Prove your changes work via direct reads, checks, or tests before finishing.
- **Keep It Simple:** Do not over-engineer. Write straightforward solutions.

## Before editing code

- Use Serena MCP to inspect symbols, references, and related files.
- Do not rely only on grep, glob, or raw file reads for cross-file changes.
- For refactor tasks, identify all call sites before editing.
- Prefer small, surgical patches.
- Run tests or relevant checks after changes.
- Explain changed files and why they were changed.

## What is real in this repo (today)

- Treat `docs/BLUEPRINT/*.md` and `BLUEPRINT.md` as target-state design; many snippets are illustrative, not executable.
- **Sprint 2a** is implemented in `main`:
  - bronze/silver write dataset directories
  - `training/src/session_split.py` builds `data/gold/session_split_map.parquet`
  - `training/src/gold.py` writes `data/gold/train.parquet`, `data/gold/val.parquet`, `data/gold/test.parquet`
  - `shared/parquet.py` holds the shared parquet file/dataset reader
- **Sprint 2b** (training pipeline) is implemented in branch `feature/sprint2b`:
  - training deps: scikit-learn, xgboost, lightgbm, catboost, mlflow, shap, optuna, matplotlib
  - MLflow service in `docker-compose.yml` (port 5000, SQLite backend, named volume)
  - MLflow/Optuna configs in `training/src/config.py` (env-aware via `os.getenv()`)
  - `training/src/evaluate.py`: `compute_metrics()` returns `tuple[dict, float]` — metrics dict + threshold
  - `training/src/model_validation.py`: fail-closed validation gate with manual override
  - `training/src/data_lineage.py`: manifest hashing (chunked) + row counting via metadata
  - `training/src/explainability.py`: SHAP artifacts + summary plot for winner model
  - `training/src/train.py`: orchestration (CatBoost, LightGBM, XGBoost + Optuna + MLflow)
  - `dvc.yaml` has `train` stage depending on gold outputs
  - Gold streaming refactor: uses `pq.ParquetWriter` to avoid OOM on 42M rows
  - Gold files are file-based (`data/bronze/events.parquet`, `data/gold/*.parquet`) not directories
- **Week 3 stream-processing simulator** is implemented in branch `week3-01` / PR #8:
  - `shared/event_id.py` adds the canonical replay event-id helper
  - `services/simulator/replay.py` handles bounded replay normalization and Quix-safe publish serialization
  - `services/simulator/app.py`, `services/simulator/Dockerfile`, and `services/simulator/requirements.txt` provide the replay CLI/container entrypoint
  - `services/tests/` includes coverage for replay ordering, publish keys, and serialized topic handling
- **Week 3 stream processor** is implemented in branch/worktree `week3-02`:
  - `week3-02` includes the Week 3 simulator files plus the stream processor increment.
  - `services/stream_processor/state.py` maintains Redis session hashes/sets, TTLs, and prediction-cache invalidation.
  - `services/stream_processor/processor.py` handles duplicate suppression, late-event policy, PostgreSQL append, and status marking.
  - `services/stream_processor/app.py` wires Quix Streams consumption from `raw_events`, routes late events through a filtered StreamingDataFrame branch to `late_events`, and defaults `PROCESSING_GUARANTEE=exactly-once`.
  - `services/stream_processor/db.py` appends accepted replay events to PostgreSQL table `replay_events` with `ON CONFLICT (event_id) DO NOTHING`.
  - `infra/postgres/init.sql` creates the `replay_events` append-log table.
  - `services/stream_processor/requirements.txt` adds service runtime deps not currently listed in base project deps: `redis`, `psycopg`, and `psycopg_pool`.
- **Week 3 compose infrastructure** is implemented in branch `week3-04` / PR #13:
  - `docker-compose.yml` extends Redpanda (`redpanda`, `redpanda-init`), Redis, PostgreSQL, simulator, stream-processor, and prediction-api services.
  - `prediction-api` is published on host port `8080` and maps to container port `8000`; do not revert this to `8080:8080`.
  - `mlflow` must allow internal Docker host headers with `--allowed-hosts mlflow,mlflow:5000,localhost,127.0.0.1`; otherwise the prediction API gets `403 Invalid Host header` when it loads `runs:/...` bundles.
  - `redpanda-init` must override the image entrypoint to `["/bin/sh", "-c"]`; otherwise `command: sh /infra/redpanda/init-topics.sh` is interpreted by `rpk` and fails as `unknown command "sh"`.
  - `infra/redpanda/init-topics.sh` explicitly creates `raw_events` and `late_events` topics with 3 partitions / 1 replica via `rpk`.
  - `.env.example` extended with Week 3 runtime settings (Kafka broker, Redis URL, PostgreSQL DSN, API key, MLflow serving bundle URI).
  - `scripts/week3_compose_smoke.py` provides a local smoke helper: builds CI fixture, spins up compose stack, runs bounded replay, injects a late event, verifies late-event routing, and calls the prediction API. It requires a real `MLFLOW_SERVING_BUNDLE_URI` and uses Python-side `timeout=10` for the `late_events` consume check.
  - `services/tests/test_compose_contract.py` and `services/tests/test_week3_smoke_script.py` provide static contract tests for compose and smoke script.
  - CI updated to install service deps, run all tests with coverage, and validate Docker Compose config.
- **Week 3 serving/prediction API** is implemented in branch/worktree `week3-03`:
  - `services/stream_processor/state.py` writes the pre-event `serving_*` snapshot into Redis before incrementing the canonical post-event counters/sets.
  - `services/prediction_api/features.py` and `services/prediction_api/app.py` read only `serving_*` fields, return the existing `redis_miss` fallback on incomplete state, use constant-time API key validation, and fail fast when `MLFLOW_SERVING_BUNDLE_URI` is missing.
  - `services/prediction_api/Dockerfile` uses the factory entrypoint `services.prediction_api.app:create_runtime_app`.
  - `training/src/train.py` raises when logging the serving bundle without an active MLflow run.
  - `training/tests/test_train.py` isolates pytest report output to `reports/train_metrics_pytest.json` so the tracked `reports/train_metrics.json` stays untouched during test runs.
- Executable data & training foundation exists in:
  - `training/src/config.py`, `training/src/bronze.py`, `training/src/silver.py`
  - `training/src/features.py`, `training/src/session_split.py`, `training/src/gold.py`
  - `training/src/evaluate.py`, `training/src/model_validation.py`, `training/src/data_lineage.py`
  - `training/src/explainability.py`, `training/src/train.py`
  - `shared/constants.py`, `shared/schemas.py`, `shared/parquet.py`
  - `training/tests/` (expanded Week 2 coverage across training, explainability, lineage, and data-lake contracts)
  - `dvc.yaml` (`bronze` -> `silver` -> `session_split` -> `gold` -> `train`)
- Infra scaffold: `docker-compose.yml` (MinIO + MLflow) + `infra/minio/init-bucket.sh`
  - MLflow metadata uses SQLite in the `mlflow_backend` volume.
  - MLflow run artifacts use MinIO/S3 at `s3://mlops-artifacts/mlflow` via the MLflow artifact proxy.
  - Existing MLflow experiments keep their original artifact root; reset `mlflow_backend` or use a new experiment name after changing artifact storage.
- Repo-level packaging/tooling:
  - `pyproject.toml` with `shared*` and `training*`, runtime/optional deps, pytest, Ruff, mypy
  - `.pre-commit-config.yaml`: hygiene + `ruff-check` + `ruff-format`
  - `.github/workflows/ci.yml`: Python 3.11/3.12 matrix — install, Ruff, pytest, `dvc dag`
  - `README.md` as top-level quick-start
- No repo-level `Makefile` or `requirements*.txt`.

## Environment and command baseline

- Preferred Python env: `conda activate MLOPS` (expected Python 3.11.x in this env).
- Install package/dev tooling with `python -m pip install -e ".[dev]"`.
- For Week 3 simulator/stream-processor worktrees, also install service deps:
  - `python -m pip install -r services/simulator/requirements.txt`
  - `python -m pip install -r services/stream_processor/requirements.txt`
- Additional dev dependencies added in `week3-04`: `PyYAML>=6.0.0` (compose contract tests), `requests>=2.31.0` (smoke script).
- If `python` is unavailable outside conda, use `python3` for scripts.
- Start local object storage: `docker compose up -d` (MinIO + MLflow).
- Start full Week 3 online stack: `docker compose up -d --build redpanda redpanda-init redis postgres minio minio-init mlflow stream-processor prediction-api`
- Quick health check: `docker compose ps` (MinIO ports `9000` API, `9001` console; MLflow port `5000`).
- Sprint 2a pipeline commands (> main branch; Sprint 2b adds train stage):
  - `python -m training.src.bronze --input data/train_raw --output data/bronze/events.parquet`
  - `python -m training.src.silver --input data/bronze/events.parquet --output data/silver`
  - `python -m training.src.session_split --input data/silver --output data/gold/session_split_map.parquet`
  - `python -m training.src.gold --input data/silver --split-map data/gold/session_split_map.parquet --output data/gold`
  - `dvc repro` now runs `bronze` -> `silver` -> `session_split` -> `gold`.
- Sprint 2b pipeline (branch `feature/sprint2b`):
  - `python -m training.src.train --train data/gold/train.parquet --val data/gold/val.parquet --test data/gold/test.parquet --session-split-map data/gold/session_split_map.parquet --smoke-mode`
  - Without `--smoke-mode`: full Optuna search (50 trials, ~1hr)
- Week 3 service commands (branch/worktree `week3-02`; require Redpanda/Redis/PostgreSQL for runtime smoke):
  - Simulator help/import smoke: `python -m services.simulator.app --help`
  - Bounded replay publish: `python -m services.simulator.app --input data/simulation_raw/2019-Nov.csv.gz --limit 100 --broker localhost:9092 --topic raw_events`
  - Stream processor runtime: `KAFKA_BROKER=localhost:9092 REDIS_URL=redis://localhost:6379/0 POSTGRES_DSN=postgresql://mlops:mlops@localhost:5432/mlops python -m services.stream_processor.app`
  - Prediction API health check: `curl http://localhost:8080/health`
  - Prediction API predict check: `curl -H "X-API-Key: ${API_KEY:-local-dev-api-key}" http://localhost:8080/api/v1/predict/<user_session>`
  - Redpanda topic re-init: `docker compose run --rm redpanda-init`
- Use module-mode commands (`python -m training.src...`) rather than direct script paths. The old `sys.path.insert(...)` import hacks were removed from `training/src` and tests.
- Common verification commands:
  - `ruff check .`
  - `pytest training/tests -q`
  - `pytest services/tests/test_event_id.py services/tests/test_simulator_replay.py services/tests/test_simulator_publish.py services/tests/test_stream_state.py services/tests/test_stream_processor.py services/tests/test_stream_processor_app.py services/tests/test_compose_contract.py services/tests/test_week3_smoke_script.py -q`
  - `docker compose config` (validate compose graph)
  - `dvc dag`
  - `PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit run --all-files` when the default home cache is not writable.
- Week 3 smoke helper (requires Docker + real `MLFLOW_SERVING_BUNDLE_URI`):
  - `python scripts/week3_compose_smoke.py`

## Data pipeline gotchas that cause real failures

- Keep schema field order from `schemas.BRONZE_SCHEMA` / `schemas.SILVER_SCHEMA` when selecting columns; do not build column order from sets.
- Bronze write path is strict on dtypes (PyArrow schema cast): ensure IDs/categorical fields stay string-like before `pa.Table.from_pandas(...)`.
- Bronze input reader supports both `*.csv` and `*.csv.gz` under `data/train_raw/`; baseline training is intentionally scoped to `2019-Oct.csv.gz` and must not mix in `data/simulation_raw/`.
- Raw layer contract: keep source `event_time`; internal layers must use `source_event_time`.
- Silver supports parquet file input and parquet dataset directory input.
- Silver clean phase reads bronze in batches. Final global dedup/sort is implemented as external sort + k-way merge over temporary parquet runs, so `finalize_silver_parts` must not be changed back to `read_bronze_parquet(parts_path)` / full pandas load.
- Silver duplicate semantics are locked: canonical duplicate keys keep the first surviving record in bronze input order. The temporary `_silver_input_order` column exists only to preserve this during external merge and must not be written to the final `SILVER_SCHEMA` output.
- `compute_metrics()` returns `tuple[dict, float]` — metrics dict + threshold. The dict contains `confusion_matrix` (numpy array) which must be filtered out before `mlflow.log_metrics()` (only accepts scalars).
- Gold streaming: use `pa.Table.from_pydict(data, schema=schema)` not `pl.DataFrame().to_arrow()` for ParquetWriter input; Polars type inference breaks on null columns.
- LightGBM v4.x: `is_unbalance` conflicts with `scale_pos_weight` — use one or the other, not both.
- Categorical-aware training keeps `category_id`, `category_code`, and `brand` as first-class inputs; do not coerce the full frame to `float32` before model-specific preprocessing.
- SHAP binary classification can return list or 3D array outputs across CatBoost/LightGBM/XGBoost; extract class 1 via `shap_values[:, :, 1]` when the explainer returns a 3D array.
- Week 3 simulator must sort bounded replay events by `user_session` and raw `event_time` before publishing; do not use unsorted CSV order as proof of late-event routing.
- Week 3 replay events must keep raw `event_time` out of the online payload and use `source_event_time` plus `replay_time`.
- Week 3 stream processor stores Redis state in `session:{user_session}` plus `session:{user_session}:products` and `session:{user_session}:categories`; every session key must receive TTL.
- Week 3 Redis null handling is intentional: numeric `price` null becomes `latest_price="0"` to match training numeric `fillna(0)`, while nullable categorical text fields use `""` and are converted to the serving bundle missing token later.
- Week 3 late events are marked with internal `_stream_processor_status="late"`, enriched with `late_reason`, routed to `late_events`, and must not update Redis or PostgreSQL `replay_events`.
- Quix `PROCESSING_GUARANTEE=exactly-once` covers Kafka output/checkpointing only; Redis/PostgreSQL remain external side effects protected by Redis dedup keys and PostgreSQL conflict handling.

## Contracts to preserve when touching architecture/docs

- Canonical Week 3 `event_id`: SHA-256 hex digest of `f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}"`, implemented by `shared.event_id.compute_event_id(...)`.
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
