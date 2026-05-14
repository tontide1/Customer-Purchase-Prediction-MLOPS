"""Lightweight data lineage metadata for MLflow logging."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone


def compute_manifest_hash(data_path: str) -> str:
    """Compute a lightweight manifest hash of parquet files in a data directory."""
    base_path = Path(data_path)
    manifest_hasher = hashlib.md5()

    for f in sorted(base_path.rglob("*.parquet")):
        stat = f.stat()
        relative_path = f.relative_to(base_path).as_posix()
        manifest_entry = f"{relative_path}|{stat.st_size}|{stat.st_mtime_ns}\n"
        manifest_hasher.update(manifest_entry.encode("utf-8"))

    return manifest_hasher.hexdigest()


def gather_lineage_metadata(
    train_path: str,
    val_path: str,
    test_path: str,
    session_split_map_path: str,
    window_start_utc: Optional[str] = None,
    window_end_utc: Optional[str] = None,
    dvc_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gather lightweight metadata from sprint inputs for traceability.

    Args:
        train_path: Path to train.parquet
        val_path: Path to val.parquet
        test_path: Path to test.parquet
        session_split_map_path: Path to session_split_map.parquet
        window_start_utc: Data collection window start
        window_end_utc: Data collection window end
        dvc_revision: DVC pipeline revision

    Returns:
        Dictionary of lineage metadata
    """
    gold_root = os.path.dirname(os.path.abspath(train_path))
    return {
        "gold_input_manifest_hash": compute_manifest_hash(gold_root),
        "gold_input_file_count": len(list(Path(gold_root).glob("*.parquet"))),
        "window_start_utc": window_start_utc or "",
        "window_end_utc": window_end_utc or "",
        "row_count_gold_train": _count_rows(train_path),
        "row_count_gold_val": _count_rows(val_path),
        "row_count_gold_test": _count_rows(test_path),
        "dvc_data_revision": dvc_revision or "",
        "input_train_path": os.path.abspath(train_path),
        "input_val_path": os.path.abspath(val_path),
        "input_test_path": os.path.abspath(test_path),
        "input_session_split_map_path": os.path.abspath(session_split_map_path),
        "metadata_timestamp_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _count_rows(parquet_path: str) -> int:
    """Count rows in a parquet file without loading all data."""
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        return parquet_file.metadata.num_rows
    except Exception:
        return 0
