# Raw Data Intake Rule

## Overview
The raw data layer is split by role so baseline training never mixes with online replay data.

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
2. `dataset/2019-Nov.csv.gz` is staged into `data/simulation_raw/` for Online Simulation / replay.
3. Replay events are persisted by the online system and exported into `data/retrain_raw/` for retraining after the 7-day operational window.
4. Files preserve original field names and values exactly (no transformations).
5. Bronze ingestion only considers files named `YYYY-Mon.csv` or `YYYY-Mon.csv.gz`; unsupported names are skipped.
6. Multiple CSV files can be ingested; selected files are read in chronological filename order.

## Raw Layer Contract
- **Immutability**: Records in raw role directories are never modified
- **Field Names**: Original source field names are preserved
- **Timestamp Field**: Uses `event_time` (not `source_event_time`)
- **No Row Filtering at Raw Layer**: Raw files are stored as-is; validation/filtering happens in bronze
- **Windowed Ingestion**: Bronze selects files by `--window-profile` (training, replay, dev_smoke, or all)
- **Window Contract**: `training` and `dev_smoke` select `2019-Oct`; `replay` selects `2019-Nov`
- **Retraining Contract**: Replay data is not used directly for baseline training; it is saved operationally, exported to `data/retrain_raw/`, and retrained after 7 days
- **Audit Trail**: Source file and ingestion time can be tracked if needed in future iterations

## Downstream Transformation
- **Bronze Layer** (`training/src/bronze.py`): 
  - Renames `event_time` → `source_event_time`
  - Validates `event_type` against allowed values
  - Rejects invalid records (logs count)
  - Writes to `data/bronze/events.parquet`

## Week 1 Scope
For Week 1, we assume:
- `data/train_raw/2019-Oct.csv.gz` is the only baseline training source
- `data/simulation_raw/2019-Nov.csv.gz` is reserved for replay / Online Simulation
- The `bronze.py` script processes only files in the active raw window profile when materializing a raw pool
- The raw layer is populated manually or via a setup script (not automated yet)

## Future (Week 2+)
- Streaming ingestion from message queue
- Scheduled batch intake from source systems
- Schema versioning and evolution
