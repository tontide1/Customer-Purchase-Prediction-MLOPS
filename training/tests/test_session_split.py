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


def test_config_defaults_include_session_cutoff_and_gold_batch_size():
    from training.src.config import Config

    settings = Config.get_all_settings()

    assert Config.TRAINING_SESSION_CUTOFF == "2019-10-16T00:00:00"
    assert Config.GOLD_BATCH_SIZE == 50000
    assert settings["training_session_cutoff"] == "2019-10-16T00:00:00"
    assert settings["gold_batch_size"] == 50000


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


def test_build_session_split_map_writes_user_session_sorted_output(tmp_path):
    from training.src.session_split import build_session_split_map

    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            constants.FIELD_SOURCE_EVENT_TIME: dt.datetime(2019, 10, 1, 10, 0, 0),
            "event_type": "view",
            "product_id": "p1",
            constants.FIELD_CATEGORY_ID: "c1",
            "user_id": "u1",
            "user_session": "session-c",
            "category_code": "cat-1",
            "brand": "brand",
            "price": 1.0,
        },
        {
            constants.FIELD_SOURCE_EVENT_TIME: dt.datetime(2019, 10, 1, 9, 0, 0),
            "event_type": "view",
            "product_id": "p2",
            constants.FIELD_CATEGORY_ID: "c2",
            "user_id": "u2",
            "user_session": "session-a",
            "category_code": "cat-2",
            "brand": "brand",
            "price": 1.0,
        },
        {
            constants.FIELD_SOURCE_EVENT_TIME: dt.datetime(2019, 10, 1, 8, 0, 0),
            "event_type": "view",
            "product_id": "p3",
            constants.FIELD_CATEGORY_ID: "c3",
            "user_id": "u3",
            "user_session": "session-b",
            "category_code": "cat-3",
            "brand": "brand",
            "price": 1.0,
        },
    ]
    pl.DataFrame(rows).write_parquet(silver_dir / "part-000.parquet")

    output_path = tmp_path / "session_split.parquet"
    build_session_split_map(str(silver_dir), str(output_path))

    result = pl.read_parquet(output_path)
    assert result.get_column("user_session").to_list() == [
        "session-a",
        "session-b",
        "session-c",
    ]


def test_build_session_split_map_rejects_empty_input(tmp_path):
    from training.src.session_split import build_session_split_map

    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="empty"):
        build_session_split_map(str(silver_dir), str(tmp_path / "session_split.parquet"))
