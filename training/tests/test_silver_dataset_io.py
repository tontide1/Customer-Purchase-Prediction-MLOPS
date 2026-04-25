"""Tests for silver dataset-aware I/O and deterministic ordering."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from shared import constants, schemas
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


def _sample_silver_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            constants.FIELD_SOURCE_EVENT_TIME: pd.to_datetime(
                [
                    "2019-10-01 10:00:00",
                    "2019-10-01 10:00:00",
                    "2019-10-01 10:00:00",
                ]
            ),
            "event_type": ["cart", "cart", "view"],
            "product_id": ["2", "1", "3"],
            constants.FIELD_CATEGORY_ID: ["c2", "c1", "c3"],
            "user_id": ["u2", "u1", "u3"],
            "user_session": ["s1", "s1", "s1"],
            "category_code": [None, "cat-1", None],
            "brand": ["b2", "b1", "b3"],
            "price": [20.0, 10.0, 30.0],
        }
    )


def test_read_bronze_parquet_supports_dataset_directory(tmp_path):
    bronze_dir = tmp_path / "bronze_dataset"
    bronze_dir.mkdir()
    table = pa.Table.from_pandas(_sample_silver_df(), schema=schemas.BRONZE_SCHEMA)
    pq.write_table(table.slice(0, 2), bronze_dir / "part-0.parquet")
    pq.write_table(table.slice(2, 1), bronze_dir / "part-1.parquet")

    df = read_bronze_parquet(str(bronze_dir))

    assert len(df) == 3
    assert constants.FIELD_SOURCE_EVENT_TIME in df.columns


def test_write_silver_parquet_supports_dataset_directory(tmp_path):
    output_dir = tmp_path / "silver_dataset"
    df = _sample_silver_df().copy()
    df = enforce_silver_dtypes(df)
    df = sort_deterministic(df)

    write_silver_parquet(df, str(output_dir))

    written_files = sorted(output_dir.glob("*.parquet"))
    assert written_files

    round_trip = read_bronze_parquet(str(output_dir))
    assert len(round_trip) == 3


def test_write_silver_parquet_creates_file_for_parquet_target(tmp_path):
    output_file = tmp_path / "events.parquet"
    df = enforce_silver_dtypes(_sample_silver_df().copy())
    df = sort_deterministic(df)

    write_silver_parquet(df, str(output_file))

    assert output_file.is_file()
    round_trip = read_bronze_parquet(str(output_file))
    assert len(round_trip) == 3


def test_write_silver_parquet_fails_for_existing_nonempty_directory(tmp_path):
    output_dir = tmp_path / "silver_dataset"
    output_dir.mkdir()
    (output_dir / "stale.parquet").write_text("stale")

    df = enforce_silver_dtypes(_sample_silver_df().copy())
    df = sort_deterministic(df)

    with pytest.raises(FileExistsError):
        write_silver_parquet(df, str(output_dir))


def test_sort_deterministic_uses_tie_breakers():
    df = enforce_silver_dtypes(_sample_silver_df().copy())

    shuffled = df.iloc[[0, 2, 1]].reset_index(drop=True)
    sorted_df = sort_deterministic(shuffled)

    assert get_silver_sort_columns(sorted_df) == [
        "user_session",
        constants.FIELD_SOURCE_EVENT_TIME,
        "event_type",
        "product_id",
        "user_id",
    ]
    assert sorted_df["product_id"].tolist() == ["1", "2", "3"]


def test_validate_silver_sort_columns_fails_fast_on_missing_column():
    df = enforce_silver_dtypes(_sample_silver_df().drop(columns=["user_id"]).copy())

    with pytest.raises(ValueError, match="missing: user_id"):
        validate_silver_sort_columns(df)

    with pytest.raises(ValueError, match="missing: user_id"):
        sort_deterministic(df)


def test_category_code_policy_keep_and_fill():
    df = enforce_silver_dtypes(_sample_silver_df().copy())

    kept = normalize_category_code(df.copy(), policy="keep")
    assert kept["category_code"].isna().sum() == 2

    filled = normalize_category_code(df.copy(), policy="fill", fill_value="unknown")
    assert filled["category_code"].tolist() == ["unknown", "cat-1", "unknown"]


def test_run_silver_pipeline_streams_batches_and_global_dedups(tmp_path):
    bronze_dir = tmp_path / "bronze_dataset"
    output_file = tmp_path / "events.parquet"
    bronze_dir.mkdir()

    df = _sample_silver_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    table = pa.Table.from_pandas(df, schema=schemas.BRONZE_SCHEMA)
    pq.write_table(table.slice(0, 2), bronze_dir / "part-0.parquet")
    pq.write_table(table.slice(2, 2), bronze_dir / "part-1.parquet")

    stats = run_silver_pipeline(str(bronze_dir), str(output_file), batch_size=1)

    round_trip = read_bronze_parquet(str(output_file))
    assert stats.input_rows == 4
    assert stats.duplicate_rows == 1
    assert stats.output_rows == 3
    assert round_trip["product_id"].tolist() == ["1", "2", "3"]

    output_schema = pq.read_table(output_file).schema.remove_metadata()
    assert output_schema == schemas.SILVER_SCHEMA


def test_run_silver_pipeline_keeps_first_duplicate_by_input_order(tmp_path):
    bronze_file = tmp_path / "bronze.parquet"
    output_file = tmp_path / "silver.parquet"

    df = _sample_silver_df()
    later_duplicate = df.iloc[[0]].copy()
    later_duplicate["brand"] = "later-brand"
    later_duplicate["price"] = 99.0
    df = pd.concat([df.iloc[[0]], df.iloc[[1]], later_duplicate], ignore_index=True)
    table = pa.Table.from_pandas(df, schema=schemas.BRONZE_SCHEMA)
    pq.write_table(table, bronze_file)

    run_silver_pipeline(str(bronze_file), str(output_file), batch_size=1)

    round_trip = read_bronze_parquet(str(output_file))
    row = round_trip[round_trip["product_id"] == "2"].iloc[0]
    assert row["brand"] == "b2"
    assert row["price"] == 20.0


def test_run_silver_pipeline_is_stable_across_repeated_runs(tmp_path):
    bronze_file = tmp_path / "bronze.parquet"
    output_one = tmp_path / "silver-one.parquet"
    output_two = tmp_path / "silver-two.parquet"

    table = pa.Table.from_pandas(_sample_silver_df(), schema=schemas.BRONZE_SCHEMA)
    pq.write_table(table, bronze_file)

    run_silver_pipeline(str(bronze_file), str(output_one), batch_size=1)
    run_silver_pipeline(str(bronze_file), str(output_two), batch_size=2)

    first = read_bronze_parquet(str(output_one))
    second = read_bronze_parquet(str(output_two))
    pd.testing.assert_frame_equal(first, second)
