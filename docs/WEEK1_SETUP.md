# Week 1: Setup and First Run Guide

## Prerequisites

1. **Python 3.8+** with the following packages:
   ```bash
   pip install pandas pyarrow dvc python-dotenv
   ```

2. **Docker & Docker Compose** for MinIO bootstrap

3. **Git & DVC** initialized in repository

## Quick Start

### Step 1: Setup Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

The default values work for local demo mode.

### Step 2: Bootstrap MinIO

Start MinIO S3-compatible storage:
```bash
docker compose up -d
```

Verify MinIO is healthy:
```bash
docker compose ps
```

Access MinIO console at `http://localhost:9001` (credentials: minioadmin/minioadmin)

### Step 3: Configure DVC Remote

Initialize DVC if not already done:
```bash
dvc init
```

Add MinIO as remote storage:
```bash
dvc remote add -d minio-local s3://mlops-artifacts/dvc
dvc remote modify minio-local endpointurl http://localhost:9000
```

### Step 4: Prepare Raw Data

Raw data layer expects CSV files in `data/raw/` named `YYYY-Mon.csv` or `YYYY-Mon.csv.gz`.

**Option A: Copy sample data** (already done for testing):
```bash
# Baseline training file: data/train_raw/2019-Oct.csv.gz
ls data/train_raw/
```

**Option B: Add more raw data** (for full dataset):
```bash
# Copy only the baseline training file into data/train_raw/
cp dataset/2019-Oct.csv.gz data/train_raw/

# Replay source is kept separate for Online Simulation
cp dataset/2019-Nov.csv.gz data/simulation_raw/
```

### Step 5: Run Bronze Pipeline

Transform raw CSV to bronze parquet:
```bash
python3 training/src/bronze.py --input data/train_raw --window-profile training
```

Expected output:
```
data/bronze/events.parquet
```

### Step 6: Run Silver Pipeline

Clean bronze data:
```bash
python3 training/src/silver.py
```

Expected output:
```
data/silver/events.parquet
```

### Step 7: Use DVC to Run Entire Pipeline

Instead of running scripts manually, use DVC:
```bash
dvc repro
```

DVC will:
- Execute bronze stage (if inputs changed)
- Execute silver stage (if bronze changed)
- Track all artifacts with checksums

### Step 8: Push Artifacts to MinIO

Store artifacts in remote storage:
```bash
dvc push
```

### Step 9: Verify Artifacts

Pull artifacts on a clean workspace:
```bash
rm data/bronze/events.parquet data/silver/events.parquet
dvc pull
```

## Testing

Run foundation tests:
```bash
python3 -m pytest training/tests/test_data_lake.py -v
```

## File Structure

```
data/
├── train_raw/              # Baseline training source
│   └── 2019-Oct.csv.gz
├── simulation_raw/         # Online Simulation / replay source
│   └── 2019-Nov.csv.gz
├── retrain_raw/            # DB exports for retraining after 7 days
├── bronze/                 # Validated, standardized
│   └── events.parquet
├── silver/                 # Cleaned, deduplicated
│   └── events.parquet
└── gold/                   # For Week 2+: train/val/test splits

training/
├── src/
│   ├── config.py           # Centralized configuration
│   ├── bronze.py           # Raw → Bronze transformer
│   ├── silver.py           # Bronze → Silver cleaner
│   └── __init__.py
├── tests/
│   ├── test_data_lake.py   # Foundation tests
│   └── __init__.py
└── __init__.py

shared/
├── constants.py            # Field names, layer names, enum values
├── schemas.py              # PyArrow schema definitions
└── __init__.py
```

## Configuration

All paths and credentials are centralized in `training/src/config.py`:

- **TRAIN_RAW_DATA_PATH**: `data/train_raw` (baseline training source)
- **SIMULATION_RAW_DATA_PATH**: `data/simulation_raw/2019-Nov.csv.gz` (replay source)
- **RETRAIN_RAW_DATA_DIR**: `data/retrain_raw` (DB exports for retraining)
- **RETRAIN_WINDOW_DAYS**: 7
- **BRONZE_DATA_PATH**: `data/bronze/events.parquet`
- **SILVER_DATA_PATH**: `data/silver/events.parquet`
- **GOLD_DATA_DIR**: `data/gold` (for Week 2+)
- **PREDICTION_HORIZON_MINUTES**: 10 (locked contract)
- **DATA_WINDOW_PROFILE**: `training` by default (`training`, `replay`, `dev_smoke`, or `all`)
- **TRAINING_WINDOW_START/END**: `2019-10` -> `2019-10`
- **DEV_SMOKE_WINDOW_START/END**: `2019-10` -> `2019-10`
- **REPLAY_WINDOW_START/END**: `2019-11` -> `2019-11`

### Replay Window

If you want to materialize the replay/demo source window instead of training:
```bash
python3 training/src/bronze.py --input data/raw --window-profile replay
```

The Week 1 baseline pipeline must not train directly from `data/simulation_raw/2019-Nov.csv.gz`. Replay events are persisted operationally and exported to `data/retrain_raw/` before retraining after 7 days.

### Training Window

To materialize the canonical training source window:
```bash
python3 training/src/bronze.py --window-profile training
```

## Timestamp Contract

| Layer | Field Name | Format | Notes |
|-------|------------|--------|-------|
| Raw | `event_time` | timestamp with " UTC" suffix | Source field name |
| Bronze | `source_event_time` | UTC timestamp | Renamed, standardized |
| Silver | `source_event_time` | UTC timestamp | Same as bronze, cleaned |

## Data Quality Rules (Week 1)

### Bronze Layer
- ✓ event_type must be in {view, cart, remove_from_cart, purchase}
- ✓ All source field names preserved

### Silver Layer
- ✓ Required fields (event_time/source_event_time, event_type, product_id, user_id, user_session) must not be null
- ✓ price must be > 0 (or null for non-commerce events)
- ✓ Records sorted by user_session + source_event_time + event_type + product_id + user_id

## Troubleshooting

### MinIO not starting
```bash
docker compose logs minio
# Check if ports 9000, 9001 are already in use
```

### Python import errors
Ensure `training/src/` and `shared/` have `__init__.py` files and are in Python path.

### DVC remote connection failed
```bash
# Test S3 connection
dvc remote list -v
dvc remote modify minio-local list
```

## Next Steps (Week 2)

- Session-based split (train/val/test)
- Gold layer snapshot generation
- Model training & evaluation
- MLflow integration
- Prediction API skeleton

## Questions?

Refer to:
- `docs/BLUEPRINT/` for architecture details
- `docs/RAW_DATA_INTAKE.md` for data contracts
- `AGENTS.md` for editing guardrails
