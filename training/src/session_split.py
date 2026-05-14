"""Build a deterministic session split map from silver data."""

from __future__ import annotations

import argparse
import datetime as dt
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

    if silver_path_p.is_dir() and not any(silver_path_p.glob("*.parquet")):
        raise ValueError("silver input is empty")

    scan_target = (
        str(silver_path_p / "*.parquet")
        if silver_path_p.is_dir()
        else str(silver_path_p)
    )

    cutoff = dt.datetime.fromisoformat(cutoff_iso)

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
        raise ValueError(
            "silver input is empty or no sessions remain after applying cutoff"
        )

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
    )

    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)
    sessions.select(
        ["user_session", "session_start_time", "session_end_time", "split"]
    ).sort("user_session").write_parquet(output_path_p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic session split map"
    )
    parser.add_argument("--input", default=Config.SILVER_DATA_PATH)
    parser.add_argument(
        "--output", default=f"{Config.GOLD_DATA_DIR}/session_split_map.parquet"
    )
    parser.add_argument("--cutoff-iso", default=Config.TRAINING_SESSION_CUTOFF)
    args = parser.parse_args()

    build_session_split_map(args.input, args.output, args.cutoff_iso)


if __name__ == "__main__":
    main()
