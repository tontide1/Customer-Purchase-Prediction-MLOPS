# Raw Data Intake Rule

## Overview
The `data/raw/` layer is the entry point for the data pipeline. It serves as an immutable record of source data before any transformations.

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
1. Raw CSV files from `dataset/` are unpacked and staged into `data/raw/`
2. Files preserve original field names and values exactly (no transformations)
3. Multiple CSV files can be ingested; they are concatenated in the raw layer

## Raw Layer Contract
- **Immutability**: Records in `data/raw/` are never modified
- **Field Names**: Original source field names are preserved
- **Timestamp Field**: Uses `event_time` (not `source_event_time`)
- **No Filtering**: All records from source are kept (even invalid ones)
- **Audit Trail**: Source file and ingestion time can be tracked if needed in future iterations

## Downstream Transformation
- **Bronze Layer** (`training/src/bronze.py`): 
  - Renames `event_time` → `source_event_time`
  - Validates `event_type` against allowed values
  - Rejects invalid records (logs count)
  - Writes to `data/bronze/events.parquet`

## Week 1 Scope
For Week 1, we assume:
- A single raw input file (or multiple files) is present in `data/raw/`
- The `bronze.py` script processes all files in `data/raw/` directory
- The raw layer is populated manually or via a setup script (not automated yet)

## Future (Week 2+)
- Streaming ingestion from message queue
- Scheduled batch intake from source systems
- Schema versioning and evolution
