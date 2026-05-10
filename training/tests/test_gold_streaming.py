"""Tests for the streaming gold snapshot builder."""

from datetime import datetime, timezone

import polars as pl
import pyarrow.parquet as pq
import pytest

from shared import schemas
from training.src.gold import build_gold_snapshots


def _make_silver(tmp_path, sessions):
    rows = []
    for session_id, events in sessions.items():
        for event in events:
            rows.append({
                "source_event_time": event["time"],
                "event_type": event["type"],
                "product_id": event.get("product_id", "P001"),
                "category_id": event.get("category_id", "C001"),
                "user_id": event.get("user_id", "U001"),
                "user_session": session_id,
                "category_code": event.get("category_code"),
                "brand": event.get("brand"),
                "price": event.get("price"),
            })
    df = pl.DataFrame(rows)
    path = tmp_path / "silver.parquet"
    df.write_parquet(path)
    return path


def _make_split_map(tmp_path, mapping):
    rows = [{"user_session": s, "split": sp} for s, sp in mapping.items()]
    df = pl.DataFrame(rows)
    path = tmp_path / "split_map.parquet"
    df.write_parquet(path)
    return path


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def test_streaming_gold_basic(tmp_path):
    sessions = {
        "S001": [
            {"time": _ts("2019-10-01T10:00:00"), "type": "view"},
            {"time": _ts("2019-10-01T10:01:00"), "type": "view"},
            {"time": _ts("2019-10-01T10:05:00"), "type": "cart"},
        ],
        "S002": [
            {"time": _ts("2019-10-01T11:00:00"), "type": "view"},
            {"time": _ts("2019-10-01T11:02:00"), "type": "purchase"},
        ],
        "S003": [
            {"time": _ts("2019-10-01T12:00:00"), "type": "view"},
        ],
    }
    split_map = {"S001": "train", "S002": "val", "S003": "test"}

    silver_path = _make_silver(tmp_path, sessions)
    split_map_path = _make_split_map(tmp_path, split_map)
    output_dir = tmp_path / "gold_output"

    build_gold_snapshots(silver_path, split_map_path, output_dir)

    for split in ("train", "val", "test"):
        path = output_dir / f"{split}.parquet"
        assert path.exists(), f"{split}.parquet not found"

    for split in ("train", "val", "test"):
        table = pq.read_table(str(output_dir / f"{split}.parquet"))
        assert table.schema == schemas.GOLD_SCHEMA, f"{split} schema mismatch"

    train_rows = pq.read_metadata(str(output_dir / "train.parquet")).num_rows
    val_rows = pq.read_metadata(str(output_dir / "val.parquet")).num_rows
    test_rows = pq.read_metadata(str(output_dir / "test.parquet")).num_rows

    assert train_rows == 3
    assert val_rows == 2
    assert test_rows == 1


def test_streaming_gold_missing_session_raises(tmp_path):
    """Session exists in silver but not in split map."""
    silver_path = _make_silver(tmp_path, {
        "S001": [{"time": _ts("2019-10-01T10:00:00"), "type": "view"}],
    })
    split_map_path = _make_split_map(tmp_path, {"OTHER": "train"})

    with pytest.raises(ValueError, match="split map does not cover all sessions"):
        build_gold_snapshots(silver_path, split_map_path, tmp_path / "out")


def test_streaming_gold_empty_split(tmp_path):
    sessions = {"S001": [
        {"time": _ts("2019-10-01T10:00:00"), "type": "view"},
    ]}
    split_map = {"S001": "train"}

    silver_path = _make_silver(tmp_path, sessions)
    split_map_path = _make_split_map(tmp_path, split_map)
    output_dir = tmp_path / "gold_output"

    build_gold_snapshots(silver_path, split_map_path, output_dir)

    for split in ("val", "test"):
        meta = pq.read_metadata(str(output_dir / f"{split}.parquet"))
        assert meta.num_rows == 0, f"{split} should be empty"
        table = pq.read_table(str(output_dir / f"{split}.parquet"))
        assert table.schema == schemas.GOLD_SCHEMA


def test_streaming_gold_empty_split_map_raises(tmp_path):
    silver_path = _make_silver(tmp_path, {})
    split_map_path = _make_split_map(tmp_path, {})

    with pytest.raises(ValueError, match="split map is missing or empty"):
        build_gold_snapshots(silver_path, split_map_path, tmp_path / "out")
