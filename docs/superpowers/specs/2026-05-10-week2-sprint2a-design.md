# Week 2 Sprint 2a MVP Design

**Goal:** Add the minimal offline data foundation for Week 2: directory-based bronze/silver artifacts, deterministic session splits, and per-event gold snapshots with 7 MVP features.

**Scope:** This spec covers only Sprint 2a. It does not include model training, Optuna, MLflow model registry usage, SHAP, or validation gating beyond the data artifacts needed later.

## Architecture

Sprint 2a keeps the existing Week 1 pipeline intact and extends it in a narrow way. Bronze and silver move from single-file outputs to parquet dataset directories so downstream stages can read batches safely. A new `session_split` stage builds a reproducible session-level split map from the silver layer. A new `gold` stage materializes per-event snapshots, computes the 7 MVP features, and writes one parquet file per split: `train.parquet`, `val.parquet`, and `test.parquet`.

The gold stage uses Polars with explicit list accumulation only where exact counts are required. To keep memory bounded without adding more infrastructure, silver is processed in 16 deterministic session-hash buckets. Each bucket is isolated, transformed, and written independently, then the split outputs are assembled from those bucket results.

## Contract

The locked MVP contract is:

- 7 features only: `total_views`, `total_carts`, `net_cart_count`, `cart_to_view_ratio`, `unique_categories`, `unique_products`, `session_duration_sec`
- `cart_to_view_ratio = 0.0` when `total_views = 0`
- `unique_categories` uses `category_code`, with `category_id` as fallback when `category_code` is null
- Label horizon is fixed at 10 minutes
- Gold snapshots are per-event, not session-level summaries
- Gold output is exactly 3 files: `train.parquet`, `val.parquet`, `test.parquet`
- Session split ratio is 80/10/10 by `session_start_time`

## Components

### 1. Directory-based bronze and silver

Bronze and silver remain the same logical layers, but their physical outputs change from single parquet files to directories. Bronze writes chunked parquet parts into `data/bronze/`. Silver reads `data/bronze/` as a parquet dataset and writes cleaned parquet parts into `data/silver/`.

This change is required so the later stages can stream data instead of loading a single growing file into memory. The implementation should stay minimal: no extra partitioning scheme, no extra metadata layer, and no change to the existing schema contract beyond output location.

### 2. `session_split`

`training/src/session_split.py` builds a session index by grouping silver on `user_session` and computing:

- `session_start_time = min(source_event_time)`
- `session_end_time = max(source_event_time)`

The sessions are sorted by `session_start_time`, then assigned `train`, `val`, or `test` in order using an 80/10/10 split. The output is `data/gold/session_split_map.parquet`.

The split is deterministic and session-disjoint. A session belongs to exactly one split, and the split map is the source of truth for the rest of the sprint.

### 3. `features`

`training/src/features.py` contains the small reusable helpers needed by gold materialization. Keep it simple and focused:

- `compute_cart_to_view_ratio(total_views, total_carts)` returns `0.0` when `total_views == 0`
- `normalize_category_value(category_code, category_id)` returns `category_code` when present, otherwise `category_id`

No generic feature framework is needed. This module only exists to keep the gold stage readable.

### 4. `gold`

`training/src/gold.py` reads silver and the split map, then materializes event snapshots for each split. The implementation uses 16 session-hash buckets to keep work bounded. For each bucket:

- sort rows by `user_session` and `source_event_time`
- compute cumulative counts per session
- accumulate exact lists for `product_id` and normalized category values
- derive exact unique counts from those lists
- compute the 10-minute label from same-session purchase events in `(source_event_time, source_event_time + 10 minutes]`
- write rows to the correct split file

The output schema must match `GOLD_SCHEMA` exactly. No internal helper columns may leak into the final parquet files.

## Data Flow

1. Raw CSV input is ingested into bronze directory files.
2. Bronze is cleaned into silver directory files.
3. Silver is reduced to one session split map.
4. Silver plus split map is converted into per-event gold snapshots.
5. Gold produces `train.parquet`, `val.parquet`, and `test.parquet`.

## Error Handling

The MVP should fail closed on bad input and avoid silent fallbacks.

- Missing required raw columns should fail in bronze ingestion.
- Invalid event types should fail fast in bronze.
- Silver must reject missing `user_session`, `user_id`, or `event_type`.
- `session_split` must fail if the silver input is empty.
- Gold must fail if the split map is missing or does not cover all sessions in silver.
- If label computation cannot find a valid same-session purchase horizon, the label is `0`, not an exception.

The plan intentionally avoids retry logic, distributed execution, or complex recovery paths. The pipeline is expected to be run locally and in CI on a known dataset.

## Testing

The tests should prove the contracts, not the implementation details.

- Bronze and silver directory outputs exist and still preserve row-count parity
- Session split is deterministic, disjoint, and approximately 80/10/10
- Gold schema matches `GOLD_SCHEMA`
- Gold features use only past events at snapshot time
- `cart_to_view_ratio` returns `0.0` when there are no views
- `unique_categories` falls back from `category_code` to `category_id`
- Label is `1` only when a same-session purchase occurs inside the 10-minute horizon
- End-to-end synthetic pipeline produces all three gold files

## Implementation Notes

- Keep the session-bucket count fixed at 16 for this sprint.
- Keep the gold outputs as three plain parquet files, not a partitioned directory tree.
- Keep the horizon hardcoded at 10 minutes for now.
- Do not introduce training code, Optuna search, MLflow experiment logging, or SHAP artifacts here.
