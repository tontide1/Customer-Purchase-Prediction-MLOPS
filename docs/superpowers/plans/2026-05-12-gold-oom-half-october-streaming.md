# Gold OOM and Half-October Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `gold`-stage OOM in `dvc repro` while changing the baseline training contract from full `2019-Oct` to the first half of October only.

**Architecture:** Do not split `data/train_raw/2019-Oct.csv.gz` physically. Instead, keep the raw file intact, filter complete sessions by `session_start_time < 2019-10-16T00:00:00` inside `session_split`, and rewrite `gold` to stream over already-sorted `silver` plus a sorted `session_split_map` without full-materializing either table in memory. The output contract remains the same: `data/gold/train.parquet`, `val.parquet`, and `test.parquet`.

**Tech Stack:** Python 3.11/3.12, Polars, PyArrow, DVC, pytest, Ruff.

---

## File Structure

- `training/src/config.py`
  - Add explicit defaults for the half-October cutoff and bounded `gold` batch size.
- `training/src/session_split.py`
  - Filter sessions by cutoff, keep deterministic 80/10/10 split, and write a split map sorted by `user_session`.
- `training/src/gold.py`
  - Replace full-table load/join/sort/group flow with a streaming merge between `silver` rows and split-map rows.
- `training/tests/test_session_split.py`
  - Lock the new half-October contract and sorted split-map output.
- `training/tests/test_gold_streaming.py`
  - Lock bounded-memory `gold` behavior, especially session continuity across parquet batches.
- `training/tests/test_gold_split_validation.py`
  - Keep split validation coverage on the new streaming implementation.
- `docs/BLUEPRINT/01_OVERVIEW.md`
- `docs/BLUEPRINT/02_ARCHITECTURE.md`
- `docs/BLUEPRINT/04_PIPELINES.md`
- `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
- `docs/BLUEPRINT/07_TESTING.md`
  - Sync the training-data contract and the `gold` materialization strategy.

### Task 1: Lock the half-October cutoff contract in tests and config

**Files:**
- Modify: `training/src/config.py`
- Modify: `training/tests/test_session_split.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `training/tests/test_session_split.py`:

```python
def test_build_session_split_map_keeps_only_sessions_before_cutoff(tmp_path):
    from training.src.session_split import build_session_split_map

    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            constants.FIELD_SOURCE_EVENT_TIME: dt.datetime(2019, 10, 15, 23, 59, 0),
            "event_type": "view",
            "product_id": "p-keep-1",
            constants.FIELD_CATEGORY_ID: "c-keep-1",
            "user_id": "u-keep-1",
            "user_session": "session-keep-1",
            "category_code": "cat-keep-1",
            "brand": "brand",
            "price": 1.0,
        },
        {
            constants.FIELD_SOURCE_EVENT_TIME: dt.datetime(2019, 10, 16, 0, 0, 0),
            "event_type": "view",
            "product_id": "p-drop-1",
            constants.FIELD_CATEGORY_ID: "c-drop-1",
            "user_id": "u-drop-1",
            "user_session": "session-drop-1",
            "category_code": "cat-drop-1",
            "brand": "brand",
            "price": 1.0,
        },
    ]
    pl.DataFrame(rows).write_parquet(silver_dir / "part-000.parquet")

    output_path = tmp_path / "session_split.parquet"
    build_session_split_map(str(silver_dir), str(output_path))

    result = pl.read_parquet(output_path)
    assert result.get_column("user_session").to_list() == ["session-keep-1"]


def test_build_session_split_map_writes_rows_sorted_by_user_session(tmp_path):
    from training.src.session_split import build_session_split_map

    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            constants.FIELD_SOURCE_EVENT_TIME: dt.datetime(2019, 10, 2, 10, 0, 0),
            "event_type": "view",
            "product_id": "p2",
            constants.FIELD_CATEGORY_ID: "c2",
            "user_id": "u2",
            "user_session": "session-b",
            "category_code": "cat-b",
            "brand": "brand",
            "price": 1.0,
        },
        {
            constants.FIELD_SOURCE_EVENT_TIME: dt.datetime(2019, 10, 1, 10, 0, 0),
            "event_type": "view",
            "product_id": "p1",
            constants.FIELD_CATEGORY_ID: "c1",
            "user_id": "u1",
            "user_session": "session-a",
            "category_code": "cat-a",
            "brand": "brand",
            "price": 1.0,
        },
    ]
    pl.DataFrame(rows).write_parquet(silver_dir / "part-000.parquet")

    output_path = tmp_path / "session_split.parquet"
    build_session_split_map(str(silver_dir), str(output_path))

    result = pl.read_parquet(output_path)
    assert result.get_column("user_session").to_list() == ["session-a", "session-b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_session_split.py -q`

Expected:
- `test_build_session_split_map_keeps_only_sessions_before_cutoff` fails because no cutoff exists yet.
- `test_build_session_split_map_writes_rows_sorted_by_user_session` fails because the current file is sorted by `session_start_time`, not `user_session`.

- [ ] **Step 3: Add the config defaults used by the new contract**

Modify `training/src/config.py`:

```python
class Config:
    # ========================================================================
    # Session Split / Gold Streaming Configuration
    # ========================================================================
    TRAINING_SESSION_CUTOFF = os.getenv(
        "TRAINING_SESSION_CUTOFF",
        "2019-10-16T00:00:00",
    )
    GOLD_BATCH_SIZE = int(os.getenv("GOLD_BATCH_SIZE", "50000"))
```

Also extend `get_all_settings()`:

```python
        "training_session_cutoff": cls.TRAINING_SESSION_CUTOFF,
        "gold_batch_size": cls.GOLD_BATCH_SIZE,
```

- [ ] **Step 4: Run the same tests again**

Run: `pytest training/tests/test_session_split.py -q`

Expected: still FAIL, because config exists but `session_split.py` has not started using it yet.

- [ ] **Step 5: Commit the test-and-config checkpoint**

```bash
git add training/src/config.py training/tests/test_session_split.py
git commit -m "test: lock half-october split contract"
```

### Task 2: Implement cutoff-aware deterministic session splitting

**Files:**
- Modify: `training/src/session_split.py`
- Test: `training/tests/test_session_split.py`

- [ ] **Step 1: Implement the minimal session filtering and sorted split-map write**

Replace the current function body in `training/src/session_split.py` with cutoff-aware logic that scans the `silver` parquet dataset lazily:

```python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from shared import constants
from training.src.config import Config


def build_session_split_map(
    silver_path: str | Path,
    output_path: str | Path,
    cutoff_iso: str = Config.TRAINING_SESSION_CUTOFF,
) -> None:
    silver_path_p = Path(silver_path)
    if not silver_path_p.exists():
        raise FileNotFoundError(f"silver input not found: {silver_path}")

    scan_target = (
        str(silver_path_p / "*.parquet")
        if silver_path_p.is_dir()
        else str(silver_path_p)
    )
    cutoff = datetime.fromisoformat(cutoff_iso)

    sessions = (
        pl.scan_parquet(scan_target)
        .group_by("user_session")
        .agg(
            pl.col(constants.FIELD_SOURCE_EVENT_TIME).min().alias("session_start_time"),
            pl.col(constants.FIELD_SOURCE_EVENT_TIME).max().alias("session_end_time"),
        )
        .sort(["session_start_time", "user_session"])
        .filter(pl.col("session_start_time") < cutoff)
        .collect()
    )

    if sessions.is_empty():
        raise ValueError("silver input is empty or no sessions remain after applying training cutoff")

    n_sessions = sessions.height
    train_end = int(n_sessions * 0.8)
    val_end = int(n_sessions * 0.9)

    sessions = (
        sessions.with_row_index("_row")
        .with_columns(
            pl.when(pl.col("_row") < train_end)
            .then(pl.lit("train"))
            .when(pl.col("_row") < val_end)
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        .drop("_row")
        .sort("user_session")
    )

    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)
    sessions.select(
        ["user_session", "session_start_time", "session_end_time", "split"]
    ).write_parquet(output_path_p)
```

- [ ] **Step 2: Keep the CLI contract explicit**

Extend the CLI in the same file:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic session split map")
    parser.add_argument("--input", default=Config.SILVER_DATA_PATH)
    parser.add_argument("--output", default=f"{Config.GOLD_DATA_DIR}/session_split_map.parquet")
    parser.add_argument("--cutoff-iso", default=Config.TRAINING_SESSION_CUTOFF)
    args = parser.parse_args()

    build_session_split_map(args.input, args.output, cutoff_iso=args.cutoff_iso)
```

- [ ] **Step 3: Run the focused tests**

Run: `pytest training/tests/test_session_split.py -q`

Expected: PASS.

- [ ] **Step 4: Run one direct module verification**

Run:

```bash
python -m training.src.session_split \
  --input data/silver \
  --output data/gold/session_split_map.parquet
```

Expected:
- Command exits `0`.
- `data/gold/session_split_map.parquet` is rewritten successfully.
- Resulting map contains only sessions with `session_start_time < 2019-10-16T00:00:00`.

- [ ] **Step 5: Commit**

```bash
git add training/src/session_split.py
git commit -m "feat: filter baseline sessions to first half of october"
```

### Task 3: Lock streaming gold behavior with batch-boundary tests

**Files:**
- Modify: `training/tests/test_gold_streaming.py`
- Modify: `training/tests/test_gold_split_validation.py`

- [ ] **Step 1: Write the failing streaming test**

Add this test to `training/tests/test_gold_streaming.py`:

```python
def test_streaming_gold_keeps_session_rows_together_across_small_batches(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "source_event_time": _ts("2019-10-01T10:00:00"),
            "event_type": "view",
            "product_id": "P001",
            "category_id": "C001",
            "user_id": "U001",
            "user_session": "S001",
            "category_code": "cat-1",
            "brand": "brand",
            "price": 10.0,
        },
        {
            "source_event_time": _ts("2019-10-01T10:01:00"),
            "event_type": "cart",
            "product_id": "P001",
            "category_id": "C001",
            "user_id": "U001",
            "user_session": "S001",
            "category_code": "cat-1",
            "brand": "brand",
            "price": 10.0,
        },
        {
            "source_event_time": _ts("2019-10-01T10:02:00"),
            "event_type": "purchase",
            "product_id": "P001",
            "category_id": "C001",
            "user_id": "U001",
            "user_session": "S001",
            "category_code": "cat-1",
            "brand": "brand",
            "price": 10.0,
        },
        {
            "source_event_time": _ts("2019-10-01T11:00:00"),
            "event_type": "view",
            "product_id": "P002",
            "category_id": "C002",
            "user_id": "U002",
            "user_session": "S002",
            "category_code": "cat-2",
            "brand": "brand",
            "price": 20.0,
        },
    ]

    silver_path = tmp_path / "silver.parquet"
    pq.write_table(pa.Table.from_pylist(rows), silver_path, row_group_size=2)

    split_map_df = pl.DataFrame(
        [
            {
                "user_session": "S001",
                "session_start_time": _ts("2019-10-01T10:00:00"),
                "session_end_time": _ts("2019-10-01T10:02:00"),
                "split": "train",
            },
            {
                "user_session": "S002",
                "session_start_time": _ts("2019-10-01T11:00:00"),
                "session_end_time": _ts("2019-10-01T11:00:00"),
                "split": "val",
            },
        ]
    )
    split_map_path = tmp_path / "split_map.parquet"
    split_map_df.write_parquet(split_map_path)

    output_dir = tmp_path / "gold_output"
    build_gold_snapshots(
        silver_path,
        split_map_path,
        output_dir,
        batch_size=2,
    )

    train_df = pl.read_parquet(output_dir / "train.parquet")
    val_df = pl.read_parquet(output_dir / "val.parquet")

    assert train_df.height == 3
    assert val_df.height == 1
    assert train_df.get_column("user_session").to_list() == ["S001", "S001", "S001"]
    assert val_df.get_column("user_session").to_list() == ["S002"]
```

- [ ] **Step 2: Tighten the split-validation test to use the streaming signature**

Update `training/tests/test_gold_split_validation.py` to call:

```python
with pytest.raises(ValueError, match="Unexpected split value: holdout"):
    build_gold_snapshots(
        silver_path,
        split_map_path,
        tmp_path / "gold",
        batch_size=2,
    )
```

- [ ] **Step 3: Run the focused gold tests to verify failure**

Run:

```bash
pytest training/tests/test_gold_streaming.py training/tests/test_gold_split_validation.py -q
```

Expected:
- The new small-batch streaming test fails because `build_gold_snapshots()` does not accept `batch_size`.
- Existing tests may still pass, confirming the new behavior is what is missing.

- [ ] **Step 4: Commit the test checkpoint**

```bash
git add training/tests/test_gold_streaming.py training/tests/test_gold_split_validation.py
git commit -m "test: lock gold streaming batch behavior"
```

### Task 4: Rewrite gold to stream silver rows instead of full-loading them

**Files:**
- Modify: `training/src/gold.py`
- Test: `training/tests/test_gold_streaming.py`
- Test: `training/tests/test_gold_split_validation.py`

- [ ] **Step 1: Add bounded row iterators and split-map validation helpers**

Add these helpers near the top of `training/src/gold.py`:

```python
import pyarrow.dataset as ds


def _iter_parquet_rows(path: str | Path, batch_size: int):
    source_path = Path(path)
    if source_path.is_file():
        parquet_file = pq.ParquetFile(source_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = pl.from_arrow(batch)
            yield from df.iter_rows(named=True)
        return

    dataset = ds.dataset(source_path, format="parquet")
    for batch in dataset.to_batches(batch_size=batch_size):
        df = pl.from_arrow(batch)
        yield from df.iter_rows(named=True)


def _iter_split_rows(split_map_path: str | Path, batch_size: int):
    for row in _iter_parquet_rows(split_map_path, batch_size=batch_size):
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected split value: {split}")
        yield row
```

- [ ] **Step 2: Change `build_gold_snapshots()` to a streaming merge**

Replace the full-materialization path with this structure:

```python
def build_gold_snapshots(
    silver_path: str | Path,
    split_map_path: str | Path,
    output_dir: str | Path,
    batch_size: int = Config.GOLD_BATCH_SIZE,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schema = schemas.GOLD_SCHEMA
    writers = {
        split: pq.ParquetWriter(output_path / f"{split}.parquet", schema)
        for split in ("train", "val", "test")
    }
    split_written = {split: False for split in ("train", "val", "test")}
    split_counts = {"train": 0, "val": 0, "test": 0}

    split_iter = iter(_iter_split_rows(split_map_path, batch_size=batch_size))
    current_split_row = next(split_iter, None)
    current_session = None
    current_session_rows: list[dict] = []
    total_sessions = 0

    def flush_current_session() -> None:
        nonlocal current_session, current_session_rows, total_sessions
        if not current_session_rows:
            return

        snapshots_list = _session_snapshots(pl.DataFrame(current_session_rows))
        split = current_split_row["split"]
        if snapshots_list:
            data = {name: [s[name] for s in snapshots_list] for name in schema.names}
            table = pa.Table.from_pydict(data, schema=schema)
            writers[split].write_table(table)
            split_counts[split] += len(snapshots_list)
            split_written[split] = True

        total_sessions += 1
        current_session = None
        current_session_rows = []

    try:
        for row in _iter_parquet_rows(silver_path, batch_size=batch_size):
            session_id = row["user_session"]

            while current_split_row is not None and current_split_row["user_session"] < session_id:
                raise ValueError(
                    "split map contains a session that does not exist in silver: "
                    f"{current_split_row['user_session']}"
                )

            if current_split_row is None or current_split_row["user_session"] != session_id:
                raise ValueError(
                    "split map does not cover all sessions in silver. "
                    f"Missing session {session_id}"
                )

            if current_session is None:
                current_session = session_id

            if session_id != current_session:
                flush_current_session()
                current_split_row = next(split_iter, None)
                current_session = session_id

            current_session_rows.append(row)

        flush_current_session()

        extra_split_row = next(split_iter, None)
        if extra_split_row is not None:
            raise ValueError(
                "split map contains a session that does not exist in silver: "
                f"{extra_split_row['user_session']}"
            )

        for split in ("train", "val", "test"):
            if not split_written[split]:
                writers[split].write_table(pa.Table.from_batches([], schema=schema))
    finally:
        for writer in writers.values():
            writer.close()
```

Implementation notes for this step:
- Keep `_session_snapshots()` and feature semantics unchanged.
- Remove the old `pl.read_parquet(split_map_path)`, `pl.read_parquet(silver_path)`, `join`, `sort`, and `group_by` flow entirely.
- Drop `gc.collect()` cleanup that only existed to mitigate the old full-load approach.

- [ ] **Step 3: Remove the redundant per-session sort inside `_session_snapshots()`**

Change:

```python
def _session_snapshots(session_df: pl.DataFrame) -> list[dict]:
    rows = session_df.sort(constants.FIELD_SOURCE_EVENT_TIME).to_dicts()
```

To:

```python
def _session_snapshots(session_df: pl.DataFrame) -> list[dict]:
    rows = session_df.to_dicts()
```

Reason:
- `silver` is already deterministically sorted by `user_session` then `source_event_time`.
- Re-sorting every session adds avoidable CPU and memory pressure.

- [ ] **Step 4: Extend the CLI with explicit batch-size control**

Update `main()` in `training/src/gold.py`:

```python
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Materialize Sprint 2a gold snapshots")
    parser.add_argument("--input", default=Config.SILVER_DATA_PATH)
    parser.add_argument("--split-map", default=f"{Config.GOLD_DATA_DIR}/session_split_map.parquet")
    parser.add_argument("--output", default=Config.GOLD_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=Config.GOLD_BATCH_SIZE)
    args = parser.parse_args()

    build_gold_snapshots(
        args.input,
        args.split_map,
        args.output,
        batch_size=args.batch_size,
    )
```

- [ ] **Step 5: Run the focused gold tests**

Run:

```bash
pytest training/tests/test_gold_streaming.py training/tests/test_gold_split_validation.py -q
```

Expected: PASS.

- [ ] **Step 6: Run an end-to-end stage verification**

Run:

```bash
python -m training.src.gold \
  --input data/silver \
  --split-map data/gold/session_split_map.parquet \
  --output data/gold
```

Expected:
- Command exits `0`.
- No full-table `silver` or `split_map` load occurs.
- `data/gold/train.parquet`, `val.parquet`, and `test.parquet` are regenerated.

- [ ] **Step 7: Run the DVC stage directly**

Run: `dvc repro session_split gold`

Expected:
- `session_split` and `gold` complete without OOM on the current machine.
- `data/gold/train.parquet` is materially smaller than the previous full-October artifact because only the first half of October is kept.

- [ ] **Step 8: Commit**

```bash
git add training/src/gold.py
git commit -m "feat: stream gold materialization by session"
```

### Task 5: Sync blueprint docs with the new baseline and streaming behavior

**Files:**
- Modify: `docs/BLUEPRINT/01_OVERVIEW.md`
- Modify: `docs/BLUEPRINT/02_ARCHITECTURE.md`
- Modify: `docs/BLUEPRINT/04_PIPELINES.md`
- Modify: `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
- Modify: `docs/BLUEPRINT/07_TESTING.md`

- [ ] **Step 1: Update the baseline data-window wording**

Apply these content changes:

```md
- Baseline training window: first half of October 2019 only.
- Operational rule: keep `data/train_raw/2019-Oct.csv.gz` intact; retain sessions whose `session_start_time < 2019-10-16T00:00:00`.
- Do not describe the baseline as "full 2019-Oct" anywhere after this change.
```

- [ ] **Step 2: Update the `gold` implementation description**

Apply this wording in the architecture/pipeline docs:

```md
- Gold snapshot generation must stream from already-sorted silver data.
- The implementation must not full-load silver or split_map into a single DataFrame.
- Gold writes directly to `data/gold/train.parquet`, `val.parquet`, and `test.parquet`.
```

- [ ] **Step 3: Update the testing blueprint**

Add these testing expectations to `docs/BLUEPRINT/07_TESTING.md`:

```md
- Test that session_split excludes sessions at or after `2019-10-16T00:00:00`.
- Test that session_split_map is sorted by `user_session` for downstream merge-join use.
- Test that gold preserves a session correctly when its events span multiple parquet batches.
```

- [ ] **Step 4: Verify the docs are internally consistent**

Run:

```bash
rg -n "full 2019-Oct|2019-Oct -> 2019-10|first half of October|2019-10-16T00:00:00|stream from already-sorted silver" \
  docs/BLUEPRINT/01_OVERVIEW.md \
  docs/BLUEPRINT/02_ARCHITECTURE.md \
  docs/BLUEPRINT/04_PIPELINES.md \
  docs/BLUEPRINT/05_PROJECT_STRUCTURE.md \
  docs/BLUEPRINT/07_TESTING.md
```

Expected:
- Old "full 2019-Oct" wording is removed from the synced files where it would contradict the new contract.
- New cutoff and streaming language appears in the updated files.

- [ ] **Step 5: Commit**

```bash
git add \
  docs/BLUEPRINT/01_OVERVIEW.md \
  docs/BLUEPRINT/02_ARCHITECTURE.md \
  docs/BLUEPRINT/04_PIPELINES.md \
  docs/BLUEPRINT/05_PROJECT_STRUCTURE.md \
  docs/BLUEPRINT/07_TESTING.md
git commit -m "docs: sync blueprint with half-october baseline"
```

### Task 6: Run the final verification set

**Files:**
- No file changes; verification only.

- [ ] **Step 1: Run the focused Python test set**

Run:

```bash
pytest \
  training/tests/test_session_split.py \
  training/tests/test_gold_streaming.py \
  training/tests/test_gold_split_validation.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run Ruff on the touched Python files**

Run:

```bash
ruff check training/src/config.py training/src/session_split.py training/src/gold.py \
  training/tests/test_session_split.py training/tests/test_gold_streaming.py \
  training/tests/test_gold_split_validation.py
```

Expected: PASS.

- [ ] **Step 3: Re-run the two DVC stages one more time from repo state**

Run: `dvc repro session_split gold`

Expected:
- PASS.
- No OOM at `gold`.
- Output artifacts under `data/gold/` are generated from the half-October baseline.

- [ ] **Step 4: Record the repo state**

Run:

```bash
git status --short
du -sh data/gold
```

Expected:
- Only intended tracked files are modified.
- `data/gold` size is visibly lower than the previous ~4.2G full-October output.
