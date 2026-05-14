"""Tests for gold split validation."""

import datetime as dt

import polars as pl
import pyarrow.parquet as pq
import pytest

from training.src.gold import build_gold_snapshots


def test_build_gold_snapshots_rejects_unknown_split(tmp_path) -> None:
    silver_path = tmp_path / "silver.parquet"
    split_map_path = tmp_path / "split_map.parquet"

    silver_df = pl.DataFrame(
        [
            {
                "source_event_time": dt.datetime(2026, 1, 1, 0, 0, 0),
                "category_id": "c1",
                "user_session": "s1",
                "user_id": "u1",
                "event_type": "view",
                "product_id": "p1",
                "category_code": "code1",
                "brand": "brand1",
                "price": 10.0,
            }
        ]
    )
    split_map_df = pl.DataFrame(
        [
            {
                "user_session": "s1",
                "session_start_time": dt.datetime(2026, 1, 1, 0, 0, 0),
                "session_end_time": dt.datetime(2026, 1, 1, 0, 0, 0),
                "split": "holdout",
            }
        ]
    )

    pq.write_table(silver_df.to_arrow(), silver_path)
    pq.write_table(split_map_df.to_arrow(), split_map_path)

    with pytest.raises(ValueError, match="Unexpected split value: holdout"):
        build_gold_snapshots(
            silver_path, split_map_path, tmp_path / "gold", batch_size=2
        )


def test_build_gold_snapshots_rejects_extra_split_session(tmp_path) -> None:
    silver_path = tmp_path / "silver.parquet"
    split_map_path = tmp_path / "split_map.parquet"

    silver_df = pl.DataFrame(
        [
            {
                "source_event_time": dt.datetime(2026, 1, 1, 0, 0, 0),
                "category_id": "c1",
                "user_session": "s1",
                "user_id": "u1",
                "event_type": "view",
                "product_id": "p1",
                "category_code": "code1",
                "brand": "brand1",
                "price": 10.0,
            }
        ]
    )
    split_map_df = pl.DataFrame(
        [
            {
                "user_session": "s1",
                "session_start_time": dt.datetime(2026, 1, 1, 0, 0, 0),
                "session_end_time": dt.datetime(2026, 1, 1, 0, 0, 0),
                "split": "train",
            },
            {
                "user_session": "s2",
                "session_start_time": dt.datetime(2026, 1, 1, 0, 0, 0),
                "session_end_time": dt.datetime(2026, 1, 1, 0, 0, 0),
                "split": "val",
            },
        ]
    )

    pq.write_table(silver_df.to_arrow(), silver_path)
    pq.write_table(split_map_df.to_arrow(), split_map_path)

    with pytest.raises(
        ValueError, match="split map contains a session that does not exist in silver"
    ):
        build_gold_snapshots(
            silver_path, split_map_path, tmp_path / "gold", batch_size=2
        )
