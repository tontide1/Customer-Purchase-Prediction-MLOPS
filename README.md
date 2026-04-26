# IUH Final MLOps Project

Customer purchase prediction MLOps project. The repository is currently in
Week 1: data foundation. The executable surface today is the raw-to-bronze and
bronze-to-silver data lake pipeline; blueprint documents describe the target
state for later weeks.

## What Works Today

- Bronze pipeline: reads training raw CSV files in chunks, validates event
  types, renames `event_time` to `source_event_time`, and writes schema-strict
  parquet.
- Silver pipeline: reads bronze parquet files or dataset directories in
  batches, filters invalid rows, globally deduplicates by the canonical event
  key, globally sorts deterministically, and writes silver parquet.
- DVC pipeline: `bronze -> silver`.
- Local MinIO scaffold for future object storage integration.

## Repository Layout

```text
shared/               Shared constants and PyArrow schemas
training/src/         Bronze and silver pipeline code
training/tests/       Pytest coverage for Week 1 contracts
docs/                 Blueprint and implementation notes
infra/minio/          Local MinIO bucket initialization
dvc.yaml              Executable Week 1 DVC stages
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

Stage the raw dataset before running anything:

```bash
mkdir -p data/train_raw data/simulation_raw
# Place 2019-Oct.csv.gz under data/train_raw (baseline training source)
# Place 2019-Nov.csv.gz under data/simulation_raw (online replay source)
```

The bronze pipeline defaults to the `training` window profile and only reads
`data/train_raw`. The replay file in `data/simulation_raw` is intentionally
not ingested by the baseline pipeline.

Start local object storage:

```bash
docker compose up -d
docker compose ps
```

Run the Week 1 data pipeline:

```bash
dvc repro
```

Equivalent direct commands:

```bash
python -m training.src.bronze \
  --input data/train_raw \
  --output data/bronze/events.parquet

python -m training.src.silver \
  --input data/bronze/events.parquet \
  --output data/silver/events.parquet
```

Run checks:

```bash
ruff check .
pytest training/tests -q
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

## Current Limits

- Gold/session features, model training, serving, monitoring, and explainability
  are roadmap items, not executable production code yet.
- `docs/BLUEPRINT/*.md` and `BLUEPRINT.md` are target-state design documents;
  prefer executable files such as `dvc.yaml` and `training/src/*.py` when there
  is a conflict.
