"""Build a deterministic session split map from silver data."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from shared import constants
from training.src.config import Config


def _read_silver_dataset(path: str | Path) -> pl.DataFrame:
    silver_path = Path(path)
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver parquet not found: {path}")
    if silver_path.is_file():
        table = pq.read_table(silver_path)
    else:
        table = ds.dataset(silver_path, format="parquet").to_table()
    return pl.from_arrow(table)


def build_session_split_map(silver_path: str | Path, output_path: str | Path) -> None:
    df = _read_silver_dataset(silver_path)
    if df.is_empty():
        raise ValueError("silver input is empty")

    sessions = (
        df.group_by("user_session")
        .agg(
            pl.col(constants.FIELD_SOURCE_EVENT_TIME).min().alias("session_start_time"),
            pl.col(constants.FIELD_SOURCE_EVENT_TIME).max().alias("session_end_time"),
        )
        .sort(["session_start_time", "user_session"])
    )

    n_sessions = sessions.height
    train_end = int(n_sessions * 0.8)
    val_end = int(n_sessions * 0.9)

    sessions = sessions.with_row_index("_row").with_columns(
        pl.when(pl.col("_row") < train_end)
        .then(pl.lit("train"))
        .when(pl.col("_row") < val_end)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    ).drop("_row")

    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)
    sessions.select(["user_session", "session_start_time", "session_end_time", "split"]).write_parquet(output_path_p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic session split map")
    parser.add_argument("--input", default=Config.SILVER_DATA_PATH)
    parser.add_argument("--output", default=f"{Config.GOLD_DATA_DIR}/session_split_map.parquet")
    args = parser.parse_args()

    build_session_split_map(args.input, args.output)


if __name__ == "__main__":
    main()
