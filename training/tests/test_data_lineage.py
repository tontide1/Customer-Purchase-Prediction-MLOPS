"""Tests for lightweight data lineage helpers."""

from __future__ import annotations

import pandas as pd

from training.src.data_lineage import compute_manifest_hash, gather_lineage_metadata


def _write_parquet(path, data):
    pd.DataFrame(data).to_parquet(path)


def test_compute_manifest_hash_changes_when_manifest_changes(tmp_path):
    data_dir = tmp_path / "gold"
    data_dir.mkdir()

    _write_parquet(data_dir / "train.parquet", {"a": [1, 2]})
    first_hash = compute_manifest_hash(str(data_dir))

    _write_parquet(data_dir / "val.parquet", {"a": [3, 4]})
    second_hash = compute_manifest_hash(str(data_dir))

    assert first_hash != second_hash


def test_gather_lineage_metadata_counts_rows_and_resolves_paths(tmp_path):
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    split_map_path = tmp_path / "session_split_map.parquet"

    _write_parquet(train_path, {"a": [1, 2, 3]})
    _write_parquet(val_path, {"a": [4, 5]})
    _write_parquet(test_path, {"a": [6]})
    _write_parquet(split_map_path, {"b": [1, 2]})

    metadata = gather_lineage_metadata(
        str(train_path),
        str(val_path),
        str(test_path),
        str(split_map_path),
        window_start_utc="2019-10-01T00:00:00Z",
        window_end_utc="2019-10-31T23:59:59Z",
        dvc_revision="abc123",
    )

    assert metadata["row_count_gold_train"] == 3
    assert metadata["row_count_gold_val"] == 2
    assert metadata["row_count_gold_test"] == 1
    assert metadata["window_start_utc"] == "2019-10-01T00:00:00Z"
    assert metadata["window_end_utc"] == "2019-10-31T23:59:59Z"
    assert metadata["dvc_data_revision"] == "abc123"
    assert metadata["input_train_path"] == str(train_path.resolve())
    assert metadata["input_session_split_map_path"] == str(split_map_path.resolve())
    assert isinstance(metadata.get("gold_input_manifest_hash"), str)
    assert metadata.get("gold_input_file_count") == 4
    assert "raw_input_manifest_hash" not in metadata
    assert "raw_input_file_count" not in metadata
