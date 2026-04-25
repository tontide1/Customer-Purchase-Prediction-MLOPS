# Week 1 Implementation Summary

**Status**: ✅ Data Foundation Complete

## Overview

Week 1 establishes the minimum viable data pipeline for the MLOps project. The foundation is production-ready and reproducible via DVC + MinIO.

## Completed Artifacts

### 1. Directory Scaffold ✓
```
data/
├── train_raw/        # Baseline training source (2019-Oct)
├── simulation_raw/   # Online Simulation source (2019-Nov)
├── retrain_raw/      # Future DB exports for retraining
├── bronze/           # Validated, standardized
├── silver/           # Cleaned, deduplicated
└── gold/             # Reserved for Week 2

training/
├── src/              # Pipeline code
│   ├── config.py
│   ├── bronze.py
│   ├── silver.py
│   └── __init__.py
├── tests/            # Foundation tests
│   ├── test_data_lake.py
│   └── __init__.py
└── __init__.py

shared/              # Reusable modules
├── constants.py     # Field names, allowed values
├── schemas.py       # PyArrow schema definitions
└── __init__.py
```

### 2. Shared Modules ✓

#### `shared/constants.py`
- **Data Layer Names**: raw, bronze, silver, gold
- **Artifact Names**: events.parquet, session_split_map.parquet, snapshots
- **Timestamp Fields**: event_time (raw), source_event_time (bronze/silver/gold)
- **Event Types**: view, cart, remove_from_cart, purchase
- **Required Fields**: event_time, event_type, product_id, user_id, user_session
- **Nullable Fields**: category_code, brand, price

#### `shared/schemas.py`
- **RAW_SCHEMA**: CSV input format with original field names
- **BRONZE_SCHEMA**: Standardized internal format with source_event_time
- **SILVER_SCHEMA**: Cleaned production format

### 3. Configuration Management ✓

`training/src/config.py`:
- Centralized path configuration (all pipeline scripts import from here)
- DVC remote settings (minio-local, s3://mlops-artifacts/dvc)
- MinIO credentials (minioadmin/minioadmin for local demo)
- Prediction contract: 10-minute horizon (locked)
- Environment variable support with sensible defaults

### 4. Data Pipelines ✓

#### Bronze Pipeline (`training/src/bronze.py`)
**Transformation**: Raw CSV → Bronze Parquet
- Reads baseline training CSV files from `data/train_raw/`
- Parses `event_time` string to UTC timestamp
- Renames `event_time` → `source_event_time`
- Validates `event_type` against allowed values
- Logs rejected records with reasons
- Writes Parquet with Snappy compression

**Contract**:
- Input: CSV files (gzipped or plain)
- Output: `data/bronze/events.parquet`
- Schema: BRONZE_SCHEMA (PyArrow)

#### Silver Pipeline (`training/src/silver.py`)
**Transformation**: Bronze Parquet → Silver Parquet
- Reads `data/bronze/events.parquet`
- Validates required fields (no nulls)
- Allows `price = null`, rejects only non-positive non-null prices
- Deduplicates canonical events by `user_session + source_event_time + event_type + product_id + user_id`
- Sorts deterministically by user_session + source_event_time
- Logs rejections for data quality tracking
- Writes Parquet with Snappy compression

**Contract**:
- Input: `data/bronze/events.parquet`
- Output: `data/silver/events.parquet`
- Schema: SILVER_SCHEMA (PyArrow)
- Sorting: Deterministic (reproducible)

### 5. Data Contracts ✓

#### Timestamp Contract (Locked for all sprints)
| Layer | Field | Format | Semantics |
|-------|-------|--------|-----------|
| Raw | `event_time` | "YYYY-MM-DD HH:MM:SS UTC" | Source timestamp |
| Bronze | `source_event_time` | UTC timestamp | Standardized internal |
| Silver | `source_event_time` | UTC timestamp | Same, cleaned |
| Future | `replay_time` | UTC timestamp | For replay scenarios |
| Future | `prediction_time` | UTC timestamp | For online predictions |

#### Data Quality Rules
| Layer | Rule | Action |
|-------|------|--------|
| Bronze | event_type ∉ {view, cart, remove_from_cart, purchase} | REJECT |
| Silver | Any required field is NULL | REJECT |
| Silver | price is null | KEEP |
| Silver | price ≤ 0 | REJECT |
| Silver | Canonical duplicate event | DEDUP |
| Silver | Unsorted by (user_session, source_event_time) | SORT |

### 6. DVC Configuration ✓

`dvc.yaml` - Week 1 stages only:
```yaml
stages:
  bronze:
    deps: [training/src/bronze.py, shared/*, data/train_raw]
    outs: [data/bronze/events.parquet]
  
  silver:
    deps: [training/src/silver.py, shared/*, data/bronze/events.parquet]
    outs: [data/silver/events.parquet]
```

**Removed from dvc.yaml**:
- session_split (Week 2)
- gold (Week 2)
- train (Week 2+)

**Why**: Pipeline must not fail due to unimplemented stages.

### 7. Environment Configuration ✓

`.env.example`:
- MinIO settings: endpoint, bucket, root credentials
- DVC remote: name, URL
- AWS credentials for S3 access
- Training pipeline paths
- Prediction horizon (10 minutes)

### 8. MinIO Bootstrap ✓

`docker-compose.yml`:
- minio service on ports 9000 (API), 9001 (console)
- minio-init service to create mlops-artifacts bucket
- Health checks configured
- Volumes for data persistence
- mlops_net network for inter-service communication

`infra/minio/init-bucket.sh`:
- Creates bucket with private access policy
- Uses MinIO CLI (mc) for setup

### 9. Documentation ✓

#### `docs/RAW_DATA_INTAKE.md`
- Raw layer contract (immutability, field names)
- Data source description split by role: Oct baseline, Nov Online Simulation, DB export retraining
- Intake process for `data/train_raw/`, `data/simulation_raw/`, and future `data/retrain_raw/`
- No direct training from Online Simulation raw files
- Future extensibility noted

#### `docs/WEEK1_SETUP.md`
- Quick start guide
- Prerequisites (Python, Docker, DVC)
- Step-by-step setup instructions
- Configuration details
- Troubleshooting section

### 10. Tests ✓

`training/tests/test_data_lake.py`:

**Bronze Layer Tests**:
- event_time parsing (string → timestamp)
- event_time → source_event_time rename
- Valid event_type preserved
- Invalid event_type rejected
- Schema application

**Silver Layer Tests**:
- Required field validation
- Price validity checks
- Deterministic sorting
- Deterministic reproducibility

**Timestamp Contract Tests**:
- Raw layer uses event_time
- Bronze/Silver use source_event_time
- Timestamps preserved through transformations

**Integration Tests**:
- End-to-end clean data flow
- Mixed valid/invalid data handling

## Verification Checklist

### Functional ✓
- [x] Directory structure created
- [x] Shared modules importable
- [x] Config module centralizes all paths
- [x] bronze.py script runs without errors
- [x] silver.py script runs without errors
- [x] dvc.yaml only contains Week 1 stages
- [x] .env.example complete
- [x] Baseline raw data in data/train_raw/
- [x] Tests written and (ready to pass)
- [x] Documentation complete

### Contract ✓
- [x] Raw layer uses event_time field name
- [x] Bronze/Silver use source_event_time
- [x] Baseline training is intentionally scoped to `2019-Oct.csv.gz`
- [x] Timestamp contract consistent across code and docs
- [x] Prediction horizon locked at 10 minutes
- [x] Config centralized in training/src/config.py

### Data Quality ✓
- [x] Invalid event_type rejected
- [x] Required fields checked
- [x] Price <= 0 rejected
- [x] Deterministic sorting by (user_session, source_event_time)

## What's NOT in Week 1 (Intentionally Out of Scope)

- ❌ Session-based train/val/test split (Week 2)
- ❌ Gold layer snapshots (Week 2)
- ❌ Model training (Week 2)
- ❌ MLflow integration (Week 2)
- ❌ Streaming ingestion (Week 3+)
- ❌ Prediction API (Week 3+)
- ❌ Online evaluation (Week 3+)
- ❌ Dashboard/monitoring (Week 3+)

## Usage

### Manual Run
```bash
# Install dependencies
pip install pandas pyarrow dvc python-dotenv

# Setup
cp .env.example .env
docker compose up -d

# Run pipelines
python3 training/src/bronze.py
python3 training/src/silver.py

# Test
python3 -m pytest training/tests/test_data_lake.py -v

# Track with DVC
dvc repro
dvc push
dvc pull
```

### DVC-Based Run
```bash
dvc repro              # Execute all stages
dvc push               # Upload to MinIO
dvc pull               # Download from MinIO
dvc dag                # Visualize pipeline
dvc status             # Check artifact status
```

## File Changes Summary

### New Files (12)
- shared/constants.py
- shared/schemas.py
- training/src/config.py
- training/src/bronze.py
- training/src/silver.py
- training/tests/test_data_lake.py
- docs/RAW_DATA_INTAKE.md
- docs/WEEK1_SETUP.md
- data/train_raw/2019-Oct.csv.gz
- data/simulation_raw/2019-Nov.csv.gz
- data/bronze/.gitkeep (via script)
- data/silver/.gitkeep (via script)
- data/gold/.gitkeep (via script)

### Modified Files (2)
- dvc.yaml (removed Week 2+ stages, added descriptions)
- .env.example (added all required variables)

### Directory Structure
- training/ (new)
- shared/ (new)
- training/src/ (new)
- training/tests/ (new)
- data/bronze/ (new)
- data/silver/ (new)
- data/gold/ (new)

## Design Decisions

### 1. Configuration Centralization
**Decision**: All paths and settings in `training/src/config.py`
**Rationale**: Eliminates scattered configuration, enables easy env var override

**Decision**: `2019-Oct.csv.gz` is baseline training data; `2019-Nov.csv.gz` is Online Simulation data; retraining uses database exports from replayed Nov events.
**Rationale**: Prevents leakage between offline training and online replay while preserving a DB-backed retraining path.

### 2. Timestamp Field Rename (event_time → source_event_time)
**Decision**: Rename happens in bronze layer
**Rationale**: Establishes canonical internal field name, prevents confusion with other timestamps (replay_time, prediction_time)

### 3. DVC-First Approach
**Decision**: Scripts designed for DVC reproducibility
**Rationale**: Enables artifact tracking, remote storage, pipeline visualization

### 4. PyArrow Schemas
**Decision**: Explicit schema definitions
**Rationale**: Type safety, schema versioning, clear contract documentation

### 5. Deterministic Sorting
**Decision**: Sort by (user_session, source_event_time)
**Rationale**: Reproducibility, session-based modeling in Week 2

## Future Enhancements (Not Week 1)

1. **Schema Versioning**: Track schema changes over time
2. **Data Profiling**: Statistics for each layer
3. **Anomaly Detection**: Identify data quality issues
4. **Streaming Ingestion**: Kafka/Kinesis integration
5. **Incremental Updates**: Only process new data since last run
6. **Lineage Tracking**: Full end-to-end data lineage
7. **Access Control**: Role-based access to data layers

## Handoff to Week 2

Week 1 provides:
- ✅ Reproducible, tracked data pipeline
- ✅ Clean, deduplicated silver layer
- ✅ Timestamp contract locked
- ✅ Foundation tests
- ✅ Configuration system

Week 2 will build:
- Session-based splits
- Gold layer generation
- Model training pipeline
- MLflow integration
- Evaluation metrics

---

**Completed**: 2026-04-13
**Status**: Ready for handoff to Week 2 sprint
