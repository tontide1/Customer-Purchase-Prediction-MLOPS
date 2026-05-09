# Week 2 Sprint 2a MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal offline training foundation for Week 2: session split, gold snapshots, and simple feature engineering on top of the existing Week 1 bronze/silver pipeline.

**Architecture:** Keep the pipeline small and deterministic. Refactor bronze/silver to directory outputs, build a session-level split map from silver, then materialize per-event gold snapshots with Polars using exact list accumulation for unique counts. Keep the feature set to the 7 MVP features we already agreed on, and avoid any extra modeling or serving complexity in this sprint.

**Tech Stack:** Python 3.11, Polars, PyArrow, DVC, MinIO, pytest, Ruff.

---

### Task 1: Lock the gold-layer contract

**Files:**
- Modify: `shared/constants.py`
- Modify: `shared/schemas.py`
- Test: `training/tests/test_gold_schema.py`

- [ ] **Step 1: Write the failing test**

```python
from shared import schemas


def test_gold_schema_has_mvp_fields_in_order():
    assert schemas.GOLD_SCHEMA.names == [
        "snapshot_id",
        "user_session",
        "user_id",
        "source_event_time",
        "event_type",
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "total_views",
        "total_carts",
        "net_cart_count",
        "cart_to_view_ratio",
        "unique_categories",
        "unique_products",
        "session_duration_sec",
        "label",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_gold_schema.py -v`
Expected: FAIL because `GOLD_SCHEMA` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# shared/constants.py
LAYER_GOLD = "gold"
GOLD_OUTPUT_FILES = (
    "train.parquet",
    "val.parquet",
    "test.parquet",
)

# shared/schemas.py
import pyarrow as pa

GOLD_SCHEMA = pa.schema([
    pa.field("snapshot_id", pa.string()),
    pa.field("user_session", pa.string()),
    pa.field("user_id", pa.string()),
    pa.field("source_event_time", pa.timestamp("us")),
    pa.field("event_type", pa.string()),
    pa.field("product_id", pa.string()),
    pa.field("category_id", pa.string()),
    pa.field("category_code", pa.string()),
    pa.field("brand", pa.string()),
    pa.field("price", pa.float64()),
    pa.field("total_views", pa.int64()),
    pa.field("total_carts", pa.int64()),
    pa.field("net_cart_count", pa.int64()),
    pa.field("cart_to_view_ratio", pa.float64()),
    pa.field("unique_categories", pa.int64()),
    pa.field("unique_products", pa.int64()),
    pa.field("session_duration_sec", pa.float64()),
    pa.field("label", pa.int8()),
])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_gold_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add shared/constants.py shared/schemas.py training/tests/test_gold_schema.py
git commit -m "feat: define the gold layer contract"
```

### Task 2: Convert bronze and silver outputs to directories

**Files:**
- Modify: `training/src/config.py`
- Modify: `training/src/bronze.py`
- Modify: `training/src/silver.py`
- Test: `training/tests/test_data_lake.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_bronze_and_silver_outputs_are_directories(tmp_path):
    bronze_out = tmp_path / "bronze"
    silver_out = tmp_path / "silver"

    assert not bronze_out.exists()
    assert not silver_out.exists()
```

Add one integration assertion in the same file that the pipeline writes parquet files into each directory rather than a single `events.parquet` file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_data_lake.py -v`
Expected: FAIL where existing code still assumes single-file paths.

- [ ] **Step 3: Write minimal implementation**

```python
# training/src/config.py
BRONZE_DATA_PATH = os.getenv("BRONZE_DATA_PATH", "data/bronze")
SILVER_DATA_PATH = os.getenv("SILVER_DATA_PATH", "data/silver")

# training/src/bronze.py
# Write parquet chunks into the output directory instead of one file.

# training/src/silver.py
# Read from a parquet dataset directory and write cleaned parquet parts into the output directory.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_data_lake.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/src/config.py training/src/bronze.py training/src/silver.py training/tests/test_data_lake.py
git commit -m "refactor: switch bronze and silver to dataset directories"
```

### Task 3: Build the session split map

**Files:**
- Create: `training/src/session_split.py`
- Test: `training/tests/test_session_split.py`

- [ ] **Step 1: Write the failing test**

```python
def test_session_split_is_deterministic_and_disjoint(tmp_path):
    result = run_session_split(tmp_path)
    assert set(result["split"]) <= {"train", "val", "test"}
    assert result.groupby("user_session").size().max() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_session_split.py -v`
Expected: FAIL because the module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from pathlib import Path

import polars as pl


def build_session_split_map(silver_path: str | Path, output_path: str | Path) -> None:
    sessions = (
        pl.scan_parquet(str(silver_path))
        .group_by("user_session")
        .agg(
            pl.col("source_event_time").min().alias("session_start_time"),
            pl.col("source_event_time").max().alias("session_end_time"),
        )
        .sort("session_start_time")
        .collect()
    )

    n = sessions.height
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    split = pl.when(pl.arange(0, n) < train_end).then("train").when(
        pl.arange(0, n) < val_end
    ).then("val").otherwise("test")

    out = sessions.with_row_index("_row").with_columns(split.alias("split")).drop("_row")
    out.write_parquet(str(output_path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_session_split.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/src/session_split.py training/tests/test_session_split.py
git commit -m "feat: add deterministic session split mapping"
```

### Task 4: Implement MVP feature engineering helpers

**Files:**
- Create: `training/src/features.py`
- Test: `training/tests/test_gold_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cart_to_view_ratio_uses_zero_when_no_views():
    assert compute_cart_to_view_ratio(0, 3) == 0.0


def test_unique_category_fallback_uses_category_id_when_code_missing():
    assert normalize_category_value(None, "123") == "123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_gold_features.py -v`
Expected: FAIL because helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations


def compute_cart_to_view_ratio(total_views: int, total_carts: int) -> float:
    if total_views == 0:
        return 0.0
    return total_carts / total_views


def normalize_category_value(category_code: str | None, category_id: str) -> str:
    return category_code if category_code is not None else category_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_gold_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/src/features.py training/tests/test_gold_features.py
git commit -m "feat: add minimal gold feature helpers"
```

### Task 5: Materialize gold snapshots

**Files:**
- Create: `training/src/gold.py`
- Test: `training/tests/test_gold_label.py`
- Test: `training/tests/test_gold_schema.py`

- [ ] **Step 1: Write the failing test**

```python
def test_purchase_inside_horizon_sets_label_one():
    result = build_gold_snapshots_for_session(sample_session_df)
    assert result["label"].to_list() == [0, 1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_gold_label.py -v`
Expected: FAIL because gold materialization does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from pathlib import Path

import polars as pl

from training.src.features import compute_cart_to_view_ratio, normalize_category_value


def build_gold_snapshots(silver_path: str | Path, split_map_path: str | Path, output_dir: str | Path) -> None:
    # Process silver in 16 session-hash buckets to keep memory bounded.
    # For each bucket, sort per session, accumulate exact category/product lists,
    # compute the 7 MVP features, derive the 10-minute label, then write split files.
    raise NotImplementedError
```

Keep the first implementation small: one bucket loop, exact list accumulation, and direct write to `train.parquet`, `val.parquet`, and `test.parquet`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_gold_label.py training/tests/test_gold_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/src/gold.py training/tests/test_gold_label.py training/tests/test_gold_schema.py
git commit -m "feat: materialize MVP gold snapshots"
```

### Task 6: Wire the DVC pipeline

**Files:**
- Modify: `dvc.yaml`
- Test: `training/tests/test_data_lake.py`

- [ ] **Step 1: Write the failing test**

```python
def test_dvc_has_session_split_and_gold_stages():
    text = Path("dvc.yaml").read_text()
    assert "session_split" in text
    assert "gold" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_data_lake.py -v`
Expected: FAIL if the new stages are missing.

- [ ] **Step 3: Write minimal implementation**

```yaml
stages:
  bronze:
    cmd: python -m training.src.bronze --input data/train_raw --output data/bronze
    outs:
      - data/bronze
  silver:
    cmd: python -m training.src.silver --input data/bronze --output data/silver
    outs:
      - data/silver
  session_split:
    cmd: python -m training.src.session_split --input data/silver --output data/gold/session_split_map.parquet
    outs:
      - data/gold/session_split_map.parquet
  gold:
    cmd: python -m training.src.gold --input data/silver --split-map data/gold/session_split_map.parquet --output data/gold
    outs:
      - data/gold/train.parquet
      - data/gold/val.parquet
      - data/gold/test.parquet
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_data_lake.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dvc.yaml training/tests/test_data_lake.py
git commit -m "chore: extend dvc pipeline for sprint 2a"
```

### Task 7: Verify the MVP end to end

**Files:**
- Test: `training/tests/test_pipeline_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_week2_sprint2a_pipeline_runs_on_synthetic_data(tmp_path):
    run_week2_sprint2a_pipeline(tmp_path)
    assert (tmp_path / "gold" / "train.parquet").exists()
    assert (tmp_path / "gold" / "val.parquet").exists()
    assert (tmp_path / "gold" / "test.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_pipeline_integration.py -v`
Expected: FAIL until all prior tasks are done.

- [ ] **Step 3: Run the full verification suite**

Run:
`ruff check .`
`pytest training/tests -q`
`dvc dag`

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add training/tests/test_pipeline_integration.py
git commit -m "test: cover the week 2 sprint 2a pipeline"
```

### Task 8: Update docs after code is stable

**Files:**
- Modify: `docs/BLUEPRINT/04_PIPELINES.md`
- Modify: `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
- Modify: `docs/BLUEPRINT/07_TESTING.md`

- [ ] **Step 1: Write the failing check**

Search for stale single-file paths and ensure docs mention directory-based bronze/silver and the 7 MVP features.

- [ ] **Step 2: Run the check**

Run: `rg "data/bronze/events\.parquet|data/silver/events\.parquet" docs/BLUEPRINT`
Expected: no remaining hits in the sections updated for Week 2.

- [ ] **Step 3: Write minimal documentation updates**

Update only the sections that describe the Week 2 pipeline and file layout. Do not rewrite unrelated roadmap content.

- [ ] **Step 4: Re-run verification**

Run: `rg "data/bronze/events\.parquet|data/silver/events\.parquet" docs/BLUEPRINT`
Expected: no stale references in the updated blueprint sections.

- [ ] **Step 5: Commit**

```bash
git add docs/BLUEPRINT/04_PIPELINES.md docs/BLUEPRINT/05_PROJECT_STRUCTURE.md docs/BLUEPRINT/07_TESTING.md
git commit -m "docs: align blueprint with week 2 sprint 2a"
```

## Self-Review Checklist

- Session split is deterministic and disjoint.
- Gold features stay at the 7 MVP fields we agreed on.
- Gold output is 3 files only, no extra partitioning complexity.
- The plan does not add training, Optuna, or SHAP work yet.
- Bronze/silver refactor is the smallest change needed to unlock downstream directory inputs.
- Tests are written before each implementation step.
