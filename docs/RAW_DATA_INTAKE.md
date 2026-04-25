# Raw Data Intake Rule

## Overview
Raw CSVs are staged by role before they enter a pipeline. This avoids accidentally mixing baseline training data with Online Simulation data.

## Source Data
- **Location**: `dataset/*.csv.gz`
- **Format**: Compressed CSV with gzip
- **Fields**: 
  - `event_time`: UTC timestamp (format: "YYYY-MM-DD HH:MM:SS UTC")
  - `event_type`: Type of user event (view, cart, remove_from_cart, purchase)
  - `product_id`: Product identifier
  - `category_id`: Category identifier
  - `category_code`: Category hierarchy (nullable)
  - `brand`: Brand name (nullable)
  - `price`: Product price (nullable)
  - `user_id`: User identifier
  - `user_session`: Session identifier (UUID-like string)

## Intake Process
1. `dataset/2019-Oct.csv.gz` is staged into `data/train_raw/` for baseline training.
2. `dataset/2019-Nov.csv.gz` is staged into `data/simulation_raw/` for Online Simulation/Data Replay.
3. Database exports from replayed online events are staged into `data/retrain_raw/<window_id>/` for retraining.
4. Files preserve original field names and values exactly (no transformations).

## Raw Layer Contract
- **Baseline training**: `data/train_raw/2019-Oct.csv.gz`
- **Online Simulation**: `data/simulation_raw/2019-Nov.csv.gz`
- **Retraining**: `data/retrain_raw/<window_id>/events.csv.gz`, exported from PostgreSQL after Nov replay
- **Immutability**: Records in raw staging directories are never modified
- **Field Names**: Original source field names are preserved
- **Timestamp Field**: Uses `event_time` (not `source_event_time`)
- **No Filtering**: All records from source are kept (even invalid ones)
- **Audit Trail**: Retraining data must trace back to database export window and DVC artifact revision

## Downstream Transformation
- **Bronze Layer** (`training/src/bronze.py`): 
  - Renames `event_time` → `source_event_time`
  - Preserves `category_id` unchanged for downstream fallback logic
  - Validates `event_type` against allowed values
  - Rejects invalid records (logs count)
  - Writes to `data/bronze/events.parquet`

## Week 1 Scope
For Week 1, we assume:
- Baseline training raw data is present in `data/train_raw/`
- The `bronze.py` script processes all files in `data/train_raw/`
- Online Simulation source is staged separately in `data/simulation_raw/`
- Retraining export directories are documented but not automated yet

## Future (Week 2+)
- Streaming ingestion from message queue
- Scheduled batch intake from source systems
- Schema versioning and evolution
