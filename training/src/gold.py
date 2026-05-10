"""Materialize Sprint 2a gold snapshots (streaming refactor)."""

from __future__ import annotations

import argparse
import gc
import logging
from datetime import timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from shared import constants, schemas
from training.src.config import Config
from training.src.features import compute_cart_to_view_ratio, normalize_category_value

logger = logging.getLogger(__name__)

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
    logger.info("Loading split map from %s", split_map_path)
    split_map = pl.read_parquet(split_map_path)
    if split_map.is_empty():
        raise ValueError("split map is missing or empty")
    logger.info("Loaded split map with %d rows", len(split_map))

    known_splits = {"train", "val", "test"}
    bad_splits = (
        split_map.select("split").unique().filter(~pl.col("split").is_in(known_splits))
    )
    if not bad_splits.is_empty():
        raise ValueError(f"Unexpected split value: {bad_splits.to_series().to_list()[0]}")

    logger.info("Loading silver data from %s", silver_path)
    silver = pl.read_parquet(silver_path)
    logger.info("Loaded %d silver rows", len(silver))

    missing_sessions = silver.select("user_session").unique().join(
        split_map.select("user_session").unique(),
        on="user_session",
        how="anti",
    )
    if not missing_sessions.is_empty():
        raise ValueError(
            f"split map does not cover all sessions in silver. "
            f"Missing {missing_sessions.height} sessions"
        )

    silver = silver.join(
        split_map.select(["user_session", "split"]),
        on="user_session",
        how="left",
    ).sort(["user_session", constants.FIELD_SOURCE_EVENT_TIME])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schema = schemas.GOLD_SCHEMA
    writers: dict[str, pq.ParquetWriter] = {}
    split_written: dict[str, bool] = {}
    for split in ("train", "val", "test"):
        writers[split] = pq.ParquetWriter(output_path / f"{split}.parquet", schema)
        split_written[split] = False

    logger.info("Processing sessions...")
    total_sessions = 0
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for (_session_id,), session_df in silver.group_by("user_session", maintain_order=True):
        split = session_df["split"][0]
        snapshots_list = _session_snapshots(session_df.drop("split"))

        if snapshots_list:
            data = {name: [s[name] for s in snapshots_list] for name in schema.names}
            table = pa.Table.from_pydict(data, schema=schema)
            writers[split].write_table(table)
            split_counts[split] += len(snapshots_list)
            split_written[split] = True

        total_sessions += 1
        if total_sessions % 100_000 == 0:
            logger.info("Processed %d sessions...", total_sessions)

    logger.info("Processed %d total sessions", total_sessions)

    for split in ("train", "val", "test"):
        if not split_written[split]:
            writers[split].write_table(pa.Table.from_batches([], schema=schema))
            logger.warning("Split '%s' has zero rows (no sessions assigned)", split)
        writers[split].close()

    logger.info(
        "Gold data written: train=%d, val=%d, test=%d rows",
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
    )

    del silver
    del split_map
    gc.collect()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Materialize Sprint 2a gold snapshots")
    parser.add_argument("--input", default=Config.SILVER_DATA_PATH)
    parser.add_argument("--split-map", default=f"{Config.GOLD_DATA_DIR}/session_split_map.parquet")
    parser.add_argument("--output", default=Config.GOLD_DATA_DIR)
    args = parser.parse_args()

    build_gold_snapshots(args.input, args.split_map, args.output)


if __name__ == "__main__":
    main()
