"""Materialize Sprint 2a gold snapshots (streaming refactor)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
import datetime as dt
from datetime import timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from shared import constants, schemas
from shared.parquet import iter_parquet_batches
from training.src.config import Config
from training.src.features import compute_cart_to_view_ratio, normalize_category_value

logger = logging.getLogger(__name__)

LABEL_HORIZON = timedelta(minutes=10)


def _empty_gold_frame() -> pl.DataFrame:
    return pl.from_arrow(pa.Table.from_batches([], schema=schemas.GOLD_SCHEMA))


def _iter_parquet_rows(path: str | Path, batch_size: int) -> Iterator[dict]:
    for batch in iter_parquet_batches(path, batch_size=batch_size):
        yield from batch.to_pylist()


def _iter_split_rows(path: str | Path, batch_size: int) -> Iterator[dict]:
    for row in _iter_parquet_rows(path, batch_size=batch_size):
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected split value: {split}")
        yield row


def _session_snapshots(rows: list[dict]) -> list[dict]:
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
        while (
            purchase_index < len(purchase_times)
            and purchase_times[purchase_index] <= current_time
        ):
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
                "cart_to_view_ratio": compute_cart_to_view_ratio(
                    total_views, total_carts
                ),
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
            normalize_category_value(
                row.get("category_code"), row[constants.FIELD_CATEGORY_ID]
            )
        )

    return snapshots


def build_gold_snapshots(
    silver_path: str | Path,
    split_map_path: str | Path,
    output_dir: str | Path,
    batch_size: int = Config.GOLD_BATCH_SIZE,
) -> None:
    training_cutoff = dt.datetime.fromisoformat(Config.TRAINING_SESSION_CUTOFF)

    logger.info("Streaming split map from %s", split_map_path)
    split_iter = _iter_split_rows(split_map_path, batch_size=batch_size)
    current_split_row = next(split_iter, None)
    if current_split_row is None:
        raise ValueError("split map is missing or empty")

    logger.info("Streaming silver data from %s", silver_path)

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
    current_session_id: str | None = None
    current_session_rows: list[dict] = []
    current_session_in_scope = False

    def align_split_row(session_id: str) -> None:
        nonlocal current_split_row

        while (
            current_split_row is not None
            and current_split_row["user_session"] < session_id
        ):
            current_split_row = next(split_iter, None)

        if current_split_row is None:
            raise ValueError(
                "split map does not cover all sessions in silver. "
                f"Missing session {session_id}"
            )

        split_session = current_split_row["user_session"]
        if split_session > session_id:
            raise ValueError(
                "split map does not cover all sessions in silver. "
                f"Missing session {session_id}"
            )

    def flush_current_session() -> None:
        nonlocal \
            current_session_id, \
            current_session_rows, \
            current_session_in_scope, \
            total_sessions

        if not current_session_rows:
            return

        snapshots_list = _session_snapshots(current_session_rows)
        split = current_split_row["split"]

        if snapshots_list:
            data = {name: [s[name] for s in snapshots_list] for name in schema.names}
            table = pa.Table.from_pydict(data, schema=schema)
            writers[split].write_table(table)
            split_counts[split] += len(snapshots_list)
            split_written[split] = True

        total_sessions += 1
        if total_sessions % 500_000 == 0:
            logger.info("Processed %d sessions...", total_sessions)
        current_session_id = None
        current_session_rows = []
        current_session_in_scope = False

    try:
        for row in _iter_parquet_rows(silver_path, batch_size=batch_size):
            session_id = row["user_session"]
            session_start_time = row[constants.FIELD_SOURCE_EVENT_TIME]

            if current_session_id is None:
                current_session_id = session_id
                current_session_in_scope = session_start_time < training_cutoff
                if current_session_in_scope:
                    align_split_row(session_id)
                    current_session_rows.append(row)
                continue

            if session_id != current_session_id:
                if current_session_in_scope:
                    flush_current_session()
                else:
                    current_session_id = None
                    current_session_rows = []
                    current_session_in_scope = False

                current_session_id = session_id
                current_session_in_scope = session_start_time < training_cutoff
                if current_session_in_scope:
                    align_split_row(session_id)
                    current_session_rows.append(row)
                continue

            if current_session_in_scope:
                current_session_rows.append(row)

        if current_session_in_scope:
            flush_current_session()

        extra_split_row = next(split_iter, None)
        if extra_split_row is not None:
            raise ValueError(
                "split map contains a session that does not exist in silver"
            )
        logger.info("Processed %d total sessions", total_sessions)

        for split in ("train", "val", "test"):
            if not split_written[split]:
                writers[split].write_table(pa.Table.from_batches([], schema=schema))
                logger.warning("Split '%s' has zero rows (no sessions assigned)", split)
    finally:
        for split in ("train", "val", "test"):
            writers[split].close()

    logger.info(
        "Gold data written: train=%d, val=%d, test=%d rows",
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Materialize Sprint 2a gold snapshots")
    parser.add_argument("--input", default=Config.SILVER_DATA_PATH)
    parser.add_argument(
        "--split-map", default=f"{Config.GOLD_DATA_DIR}/session_split_map.parquet"
    )
    parser.add_argument("--output", default=Config.GOLD_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=Config.GOLD_BATCH_SIZE)
    args = parser.parse_args()

    build_gold_snapshots(
        args.input,
        args.split_map,
        args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
