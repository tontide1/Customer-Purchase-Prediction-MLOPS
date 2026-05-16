# Customer Purchase Prediction MLOps

Customer purchase prediction MLOps project. The repository now includes the
Week 1 data foundation plus the Week 2 categorical-aware training pipeline.
The executable surface spans raw-to-bronze, bronze-to-silver, session split,
gold snapshots, and model training; blueprint documents describe the target
state for the remaining serving/monitoring pieces.

## What Works Today

- Bronze pipeline: reads training raw CSV files in chunks, validates event
  types, renames `event_time` to `source_event_time`, and writes schema-strict
  parquet.
- Silver pipeline: reads bronze parquet files or dataset directories in
  batches, filters invalid rows, globally deduplicates by the canonical event
  key, globally sorts deterministically, and writes silver parquet.
- Session split and gold pipelines: build `data/gold/session_split_map.parquet`
  and per-event gold snapshots for train/val/test.
- Training pipeline: compares `CatBoost`, `LightGBM`, and `XGBoost` on the
  gold snapshots, logs to MLflow, and selects the winner by validation PR-AUC.
- Week 3 stream processing and serving: the simulator, stream processor, and
  Redis-backed prediction API are implemented in `services/`. The stream
  processor stores pre-event `serving_*` snapshots for model inference, and the
  prediction API falls back cleanly when the Redis snapshot is missing or
  incomplete.
- DVC pipeline: `bronze -> silver -> session_split -> gold -> train`.
- Local MinIO scaffold for future object storage integration.

## Repository Layout

```text
shared/               Shared constants and PyArrow schemas
training/src/         Bronze, silver, gold, and training pipeline code
training/tests/       Pytest coverage for Week 1 and Week 2 contracts
docs/                 Blueprint and implementation notes
infra/minio/          Local MinIO bucket initialization
dvc.yaml              Executable Week 1 and Week 2 DVC stages
docker-compose.yml    Local MinIO services
```

## Environment

Preferred local environment:

```bash
conda activate MLOPS
python -m pip install -e ".[dev]"
```

The project supports Python `>=3.11,<3.13`. Runtime and dev dependencies are
declared in `pyproject.toml`.

## Quick Start

Stage the raw dataset before running anything. Week 1 expects the public Kaggle
eCommerce events datasets by `mkechinov` — schema-compatible options include
[eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
and [eCommerce events history in cosmetics shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop).
Download `2019-Oct.csv.gz` (training) and `2019-Nov.csv.gz` (replay) from one
of those and place them as shown:

```bash
mkdir -p data/train_raw data/simulation_raw
# Place 2019-Oct.csv.gz under data/train_raw (baseline training source)
# Place 2019-Nov.csv.gz under data/simulation_raw (online replay source)
```

The bronze pipeline defaults to the `training` window profile and only reads
`data/train_raw`. The replay file in `data/simulation_raw` is intentionally
not ingested by the baseline pipeline.

Start local object storage and MLflow:

```bash
docker compose up -d
docker compose ps
```

The MLflow server uses SQLite for tracking metadata and stores run artifacts in
MinIO at `s3://mlops-artifacts/mlflow` through MLflow's artifact proxy. If you
change MinIO credentials, keep `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` aligned for the MLflow service.

If `sprint2b_training` was already created before this S3 artifact setup,
reset the local MLflow backend once so the experiment is recreated with the
new artifact location:

```bash
docker compose down
docker volume rm iuh_final_mlflow_backend
docker compose up -d --build
```

Run the full pipeline, which executes Week 1 data preparation and Week 2
training:

```bash
dvc repro
```

If you want to run only Week 1 step by step, use the direct module commands:

```bash
python -m training.src.bronze \
  --input data/train_raw \
  --output data/bronze/events.parquet

python -m training.src.silver \
  --input data/bronze/events.parquet \
  --output data/silver

python -m training.src.session_split \
  --input data/silver \
  --output data/gold/session_split_map.parquet

python -m training.src.gold \
  --input data/silver \
  --split-map data/gold/session_split_map.parquet \
  --output data/gold
```

If you want to run only Week 2 after Week 1 has produced the gold files, use:

```bash
python -m training.src.train \
  --train data/gold/train.parquet \
  --val data/gold/val.parquet \
  --test data/gold/test.parquet \
  --session-split-map data/gold/session_split_map.parquet
```

## Week 3 Online Replay Slice

Week 3 adds a local online path on top of the existing offline pipeline:

- `services/simulator` reads `data/simulation_raw/2019-Nov.csv.gz`, normalizes raw rows, and publishes bounded replay events to `raw_events`.
- `services/stream_processor` consumes `raw_events`, suppresses duplicate `event_id` values with Redis TTL keys, routes late events to `late_events`, updates Redis session state, invalidates prediction cache keys, and appends accepted replay events to PostgreSQL.
- `services/prediction_api` serves `GET /health` and authenticated `GET /api/v1/predict/{user_session}` using the MLflow serving bundle logged by a smoke training run.

Run the local online stack:

```bash
docker compose up -d --build redpanda redpanda-init redis postgres minio minio-init mlflow stream-processor prediction-api
```

Create a serving bundle with a smoke training run:

```bash
python -m training.src.train \
  --train data/gold/train.parquet \
  --val data/gold/val.parquet \
  --test data/gold/test.parquet \
  --session-split-map data/gold/session_split_map.parquet \
  --smoke-mode \
  --device cpu
```

Set `MLFLOW_SERVING_BUNDLE_URI` to the winner `{model}_test_evaluation` run URI. The placeholder `runs:/replace-with-week3-smoke-run` is only a compose default and must fail full smoke. Then replay a bounded batch:

```bash
docker compose run --rm simulator python -m services.simulator.app --limit 1000
```

Call the API:

```bash
curl -H "X-API-Key: ${API_KEY:-local-dev-api-key}" \
  http://localhost:8080/api/v1/predict/<user_session>
```

Run the Week 3 smoke helper:

```bash
python scripts/week3_compose_smoke.py
```

Run checks:

```bash
ruff check .
pytest training/tests services/tests -q
dvc dag
```

Run pre-commit locally:

```bash
PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit run --all-files
```

## Data Window Contract

- Baseline training reads `data/train_raw` and is scoped to `2019-Oct.csv.gz`
  by default.
- Online replay uses `data/simulation_raw/2019-Nov.csv.gz`.
- Retraining should use replay events exported into `data/retrain_raw`.
- Baseline bronze ingestion must not read from `data/simulation_raw`.
- Week 2 training consumes the gold outputs from Week 1 and does not read raw
  CSV files directly.

## Current Limits

- Gold/session features, categorical-aware model training, and explainability
  are implemented; monitoring and online hot-reload remain roadmap items.
- Week 3 stream-processing simulator work lives in branch `week3-01` / PR #8:
  deterministic replay event IDs, bounded replay normalization, Quix-safe
  publish serialization, and the simulator CLI/container entrypoint.
- Week 3 stream processor and serving work lives in branch/worktree
  `week3-03`: the Redis-backed pre-event serving snapshot, the prediction API,
  and the MLflow serving bundle logging guard are implemented there.
- `docs/BLUEPRINT/*.md` and `BLUEPRINT.md` are target-state design documents;
  prefer executable files such as `dvc.yaml` and `training/src/*.py` when there
  is a conflict.
