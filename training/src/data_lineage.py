"""Lightweight data lineage metadata for MLflow logging."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def compute_manifest_hash(data_path: str) -> str:
    """Compute hash of all files in data directory."""
    file_hashes = []
    for f in sorted(Path(data_path).rglob("*.parquet")):
        hasher = hashlib.md5()
        with open(f, "rb") as fp:
            while chunk := fp.read(8192):
                hasher.update(chunk)
        file_hashes.append(hasher.hexdigest())

    combined = "".join(file_hashes)
    return hashlib.md5(combined.encode()).hexdigest()


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
    return {
        "raw_input_manifest_hash": compute_manifest_hash("data/gold"),
        "raw_input_file_count": len(list(Path("data/gold").glob("*.parquet"))),
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
        "metadata_timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }


def _count_rows(parquet_path: str) -> int:
    """Count rows in a parquet file without loading all data."""
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(parquet_path)
        return parquet_file.metadata.num_rows
    except Exception:
        return 0
