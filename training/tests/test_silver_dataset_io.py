"""Tests for silver dataset-aware I/O and deterministic ordering."""

import datetime as dt

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from shared import constants, schemas
from training.src.bronze import finalize_bronze_polars_dtypes, bronze_polars_to_arrow_table
from training.src.silver import (
    enforce_silver_dtypes,
    get_silver_sort_columns,
    normalize_category_code,
    read_bronze_parquet,
    run_silver_pipeline,
    sort_deterministic,
    validate_silver_sort_columns,
    write_silver_parquet,
)


def _silver_pl_sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            constants.FIELD_SOURCE_EVENT_TIME: [
                dt.datetime(2019, 10, 1, 10, 0, 0),
                dt.datetime(2019, 10, 1, 10, 0, 0),
                dt.datetime(2019, 10, 1, 10, 0, 0),
            ],
            "event_type": ["cart", "cart", "view"],
            "product_id": ["2", "1", "3"],
            constants.FIELD_CATEGORY_ID: ["c2", "c1", "c3"],
            "user_id": ["u2", "u1", "u3"],
            "user_session": ["s1", "s1", "s1"],
            "category_code": [None, "cat-1", None],
            "brand": ["b2", "b1", "b3"],
            "price": [20.0, 10.0, 30.0],
        },
    )


def _to_bronze_arrow(df: pl.DataFrame) -> pa.Table:
    return bronze_polars_to_arrow_table(finalize_bronze_polars_dtypes(df))


def test_read_bronze_parquet_supports_dataset_directory(tmp_path) -> None:
    bronze_dir = tmp_path / "bronze_dataset"
    bronze_dir.mkdir()
    tbl = _to_bronze_arrow(_silver_pl_sample())
    pq.write_table(tbl.slice(0, 2), bronze_dir / "part-0.parquet")
    pq.write_table(tbl.slice(2, 1), bronze_dir / "part-1.parquet")

    df = read_bronze_parquet(str(bronze_dir))

    assert df.height == 3
    assert constants.FIELD_SOURCE_EVENT_TIME in df.columns


def test_write_silver_parquet_supports_dataset_directory(tmp_path) -> None:
    output_dir = tmp_path / "silver_dataset"
    df = _silver_pl_sample()
    df = enforce_silver_dtypes(df)
    df = sort_deterministic(df)

    write_silver_parquet(df, str(output_dir))

    written_files = sorted(output_dir.glob("*.parquet"))
    assert written_files

    round_trip = read_bronze_parquet(str(output_dir))
    assert round_trip.height == 3


def test_write_silver_parquet_creates_file_for_parquet_target(tmp_path) -> None:
    output_file = tmp_path / "events.parquet"
    df = enforce_silver_dtypes(_silver_pl_sample())
    df = sort_deterministic(df)

    write_silver_parquet(df, str(output_file))

    assert output_file.is_file()
    round_trip = read_bronze_parquet(str(output_file))
    assert round_trip.height == 3


def test_write_silver_parquet_fails_for_existing_nonempty_directory(tmp_path) -> None:
    output_dir = tmp_path / "silver_dataset"
    output_dir.mkdir()
    (output_dir / "stale.parquet").write_text("stale")

    df = enforce_silver_dtypes(_silver_pl_sample())
    df = sort_deterministic(df)

    with pytest.raises(FileExistsError):
        write_silver_parquet(df, str(output_dir))


def test_sort_deterministic_uses_tie_breakers() -> None:
    df = enforce_silver_dtypes(_silver_pl_sample())
    shuffled = pl.concat([df.slice(0, 1), df.slice(2, 1), df.slice(1, 1)])
    sorted_df = sort_deterministic(shuffled)

    assert get_silver_sort_columns(sorted_df) == [
        "user_session",
        constants.FIELD_SOURCE_EVENT_TIME,
        "event_type",
        "product_id",
        "user_id",
    ]
    assert sorted_df.select("product_id").to_series().to_list() == ["1", "2", "3"]


def test_validate_silver_sort_columns_fails_fast_on_missing_column() -> None:
    df = enforce_silver_dtypes(_silver_pl_sample().drop("user_id"))

    with pytest.raises(ValueError, match="missing: user_id"):
        validate_silver_sort_columns(df)

    with pytest.raises(ValueError, match="missing: user_id"):
        sort_deterministic(df)


def test_category_code_policy_keep_and_fill() -> None:
    df = enforce_silver_dtypes(_silver_pl_sample())

    kept = normalize_category_code(df.clone(), policy="keep")
    assert kept.filter(pl.col("category_code").is_null()).height == 2

    filled = normalize_category_code(df.clone(), policy="fill", fill_value="unknown")
    assert filled.select("category_code").to_series().to_list() == [
        "unknown",
        "cat-1",
        "unknown",
    ]


def test_run_silver_pipeline_streams_batches_and_global_dedups(tmp_path) -> None:
    bronze_dir = tmp_path / "bronze_dataset"
    output_file = tmp_path / "events.parquet"
    bronze_dir.mkdir()

    df = _silver_pl_sample().vstack(_silver_pl_sample().slice(0, 1))
    tbl = _to_bronze_arrow(df)
    pq.write_table(tbl.slice(0, 2), bronze_dir / "part-0.parquet")
    pq.write_table(tbl.slice(2, 2), bronze_dir / "part-1.parquet")

    stats = run_silver_pipeline(str(bronze_dir), str(output_file), batch_size=1)

    round_trip = read_bronze_parquet(str(output_file))
    assert stats.input_rows == 4
    assert stats.duplicate_rows == 1
    assert stats.output_rows == 3
    assert round_trip.select("product_id").to_series().to_list() == ["1", "2", "3"]

    output_schema = pq.read_table(output_file).schema.remove_metadata()
    assert output_schema == schemas.SILVER_SCHEMA


def test_run_silver_pipeline_keeps_first_duplicate_by_input_order(tmp_path) -> None:
    bronze_file = tmp_path / "bronze.parquet"
    output_file = tmp_path / "silver.parquet"

    df = _silver_pl_sample()
    dup = df.slice(0, 1).with_columns(
        pl.lit("later-brand").alias("brand"),
        pl.lit(99.0).alias("price"),
    )
    df = df.slice(0, 1).vstack(df.slice(1, 1)).vstack(dup)

    pq.write_table(_to_bronze_arrow(df), bronze_file)

    run_silver_pipeline(str(bronze_file), str(output_file), batch_size=1)

    round_trip = read_bronze_parquet(str(output_file))
    row = round_trip.filter(pl.col("product_id") == "2").to_dicts()[0]
    assert row["brand"] == "b2"
    assert row["price"] == 20.0


def test_run_silver_pipeline_is_stable_across_repeated_runs(tmp_path) -> None:
    bronze_file = tmp_path / "bronze.parquet"
    output_one = tmp_path / "silver-one.parquet"
    output_two = tmp_path / "silver-two.parquet"

    pq.write_table(_to_bronze_arrow(_silver_pl_sample()), bronze_file)

    run_silver_pipeline(str(bronze_file), str(output_one), batch_size=1)
    run_silver_pipeline(str(bronze_file), str(output_two), batch_size=2)

    first = read_bronze_parquet(str(output_one))
    second = read_bronze_parquet(str(output_two))
    assert_frame_equal(first, second)
