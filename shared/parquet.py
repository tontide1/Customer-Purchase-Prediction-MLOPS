"""Shared helpers for reading parquet files and dataset directories."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def read_parquet_dataset(path: str | Path) -> pl.DataFrame:
    """Read a parquet file or parquet dataset directory into Polars."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Parquet input not found: {path}")
    if source_path.is_file():
        table = pq.read_table(source_path)
    else:
        table = ds.dataset(source_path, format="parquet").to_table()
    return pl.from_arrow(table)


def iter_parquet_batches(path: str | Path, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Iterate parquet file or dataset batches without full materialization."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Parquet input not found: {path}")
    if source_path.is_file():
        yield from pq.ParquetFile(source_path).iter_batches(batch_size=batch_size)
        return

    yield from ds.dataset(source_path, format="parquet").to_batches(batch_size=batch_size)
