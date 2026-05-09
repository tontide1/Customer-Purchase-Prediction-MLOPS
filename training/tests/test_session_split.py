"""Tests for the Sprint 2a session split stage."""

import datetime as dt

import polars as pl
import pytest

from shared import constants


def _write_silver_dataset(path, session_count: int = 10) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(session_count):
        rows.append(
            {
                constants.FIELD_SOURCE_EVENT_TIME: dt.datetime(2019, 10, 1, 10, idx, 0),
                "event_type": "view",
                "product_id": f"p{idx}",
                constants.FIELD_CATEGORY_ID: f"c{idx}",
                "user_id": f"u{idx}",
                "user_session": f"session-{idx}",
                "category_code": f"cat-{idx}",
                "brand": "brand",
                "price": 1.0,
            }
        )

    pl.DataFrame(rows).write_parquet(path / "part-000.parquet")


def test_build_session_split_map_is_deterministic_and_approximately_80_10_10(tmp_path):
    from training.src.session_split import build_session_split_map

    silver_dir = tmp_path / "silver"
    out_one = tmp_path / "session_split_one.parquet"
    out_two = tmp_path / "session_split_two.parquet"
    _write_silver_dataset(silver_dir, session_count=10)

    build_session_split_map(str(silver_dir), str(out_one))
    build_session_split_map(str(silver_dir), str(out_two))

    first = pl.read_parquet(out_one)
    second = pl.read_parquet(out_two)

    assert first.equals(second)
    assert first.height == 10
    counts = {
        row["split"]: row["count"]
        for row in first.get_column("split").value_counts().to_dicts()
    }
    assert counts == {"train": 8, "val": 1, "test": 1}
    assert first.sort("session_start_time").get_column("session_start_time").is_sorted()
    assert set(first.get_column("split").to_list()) == {"train", "val", "test"}


def test_build_session_split_map_rejects_empty_input(tmp_path):
    from training.src.session_split import build_session_split_map

    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="empty"):
        build_session_split_map(str(silver_dir), str(tmp_path / "session_split.parquet"))
