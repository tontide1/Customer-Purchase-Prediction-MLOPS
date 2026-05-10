"""Materialize Sprint 2a gold snapshots."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa

from shared import constants, schemas
from shared.parquet import read_parquet_dataset
from training.src.config import Config
from training.src.features import compute_cart_to_view_ratio, normalize_category_value

BUCKET_COUNT = 16
LABEL_HORIZON = timedelta(minutes=10)


def _empty_gold_frame() -> pl.DataFrame:
    return pl.from_arrow(pa.Table.from_batches([], schema=schemas.GOLD_SCHEMA))


def _session_snapshots(session_df: pl.DataFrame) -> list[dict]:
    rows = session_df.sort(constants.FIELD_SOURCE_EVENT_TIME).to_dicts()
    if not rows:
        return []

    session_start = rows[0][constants.FIELD_SOURCE_EVENT_TIME]
    purchase_times = [
        row[constants.FIELD_SOURCE_EVENT_TIME]
        for row in rows
        if row["event_type"] == "purchase"
    ]
    purchase_index = 0

    total_views = 0
    total_carts = 0
    total_removes = 0
    seen_products: set[str] = set()
    seen_categories: set[str] = set()
    snapshots: list[dict] = []

    for row in rows:
        current_time = row[constants.FIELD_SOURCE_EVENT_TIME]
        while purchase_index < len(purchase_times) and purchase_times[purchase_index] <= current_time:
            purchase_index += 1

        snapshots.append(
            {
                constants.FIELD_SOURCE_EVENT_TIME: current_time,
                constants.FIELD_CATEGORY_ID: row[constants.FIELD_CATEGORY_ID],
                "user_session": row["user_session"],
                "user_id": row["user_id"],
                "event_type": row["event_type"],
                "product_id": row["product_id"],
                "category_code": row.get("category_code"),
                "brand": row.get("brand"),
                "price": row.get("price"),
                "total_views": total_views,
                "total_carts": total_carts,
                "net_cart_count": total_carts - total_removes,
                "cart_to_view_ratio": compute_cart_to_view_ratio(total_views, total_carts),
                "unique_categories": len(seen_categories),
                "unique_products": len(seen_products),
                "session_duration_sec": (current_time - session_start).total_seconds(),
                "label": int(
                    purchase_index < len(purchase_times)
                    and purchase_times[purchase_index] <= current_time + LABEL_HORIZON
                ),
            }
        )

        if row["event_type"] == "view":
            total_views += 1
        elif row["event_type"] == "cart":
            total_carts += 1
        elif row["event_type"] == "remove_from_cart":
            total_removes += 1

        seen_products.add(row["product_id"])
        seen_categories.add(
            normalize_category_value(row.get("category_code"), row[constants.FIELD_CATEGORY_ID])
        )

    return snapshots


def build_gold_snapshots(
    silver_path: str | Path,
    split_map_path: str | Path,
    output_dir: str | Path,
) -> None:
    silver = read_parquet_dataset(silver_path)
    split_map = read_parquet_dataset(split_map_path)
    if split_map.is_empty():
        raise ValueError("split map is missing or empty")

    missing_sessions = silver.select("user_session").unique().join(
        split_map.select("user_session").unique(),
        on="user_session",
        how="anti",
    )
    if not missing_sessions.is_empty():
        raise ValueError("split map does not cover all sessions in silver")

    joined = silver.join(split_map.select(["user_session", "split"]), on="user_session", how="left")
    joined = joined.with_columns(
        pl.col("user_session")
        .hash()
        .mod(BUCKET_COUNT)
        .cast(pl.Int64)
        .alias("_bucket")
    ).sort(["_bucket", "user_session", constants.FIELD_SOURCE_EVENT_TIME])

    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for bucket in range(BUCKET_COUNT):
        bucket_df = joined.filter(pl.col("_bucket") == bucket)
        if bucket_df.is_empty():
            continue

        for session_df in bucket_df.partition_by("user_session", as_dict=False, maintain_order=True):
            split = session_df["split"][0]
            if split not in rows_by_split:
                raise ValueError(f"Unexpected split value: {split}")
            rows_by_split[split].extend(_session_snapshots(session_df.drop(["split", "_bucket"])))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        split_rows = rows_by_split[split_name]
        frame = _empty_gold_frame() if not split_rows else pl.DataFrame(split_rows).select(list(schemas.GOLD_SCHEMA.names))
        frame.write_parquet(output_path / f"{split_name}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Sprint 2a gold snapshots")
    parser.add_argument("--input", default=Config.SILVER_DATA_PATH)
    parser.add_argument("--split-map", default=f"{Config.GOLD_DATA_DIR}/session_split_map.parquet")
    parser.add_argument("--output", default=Config.GOLD_DATA_DIR)
    args = parser.parse_args()

    build_gold_snapshots(args.input, args.split_map, args.output)


if __name__ == "__main__":
    main()
