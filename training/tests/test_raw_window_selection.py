"""Tests for raw window selection in the bronze pipeline."""

from pathlib import Path
import importlib

import pytest


@pytest.fixture
def bronze_module(monkeypatch):
    monkeypatch.setenv("DATA_WINDOW_PROFILE", "dev_smoke")
    monkeypatch.setenv("TRAINING_WINDOW_START", "2019-10")
    monkeypatch.setenv("TRAINING_WINDOW_END", "2019-10")
    monkeypatch.setenv("DEV_SMOKE_WINDOW_START", "2019-10")
    monkeypatch.setenv("DEV_SMOKE_WINDOW_END", "2019-10")
    monkeypatch.setenv("REPLAY_WINDOW_START", "2019-11")
    monkeypatch.setenv("REPLAY_WINDOW_END", "2019-11")

    config_module = importlib.import_module("training.src.config")
    bronze_module = importlib.import_module("training.src.bronze")

    importlib.reload(config_module)
    return importlib.reload(bronze_module)


def _create_raw_files(base_dir: Path, filenames: list[str]) -> None:
    for filename in filenames:
        (base_dir / filename).touch()


def test_select_raw_files_training_window(tmp_path, bronze_module):
    _create_raw_files(
        tmp_path,
        [
            "2019-Sep.csv.gz",
            "2019-Oct.csv.gz",
            "2019-Nov.csv.gz",
            "2019-Dec.csv.gz",
            "2020-Jan.csv.gz",
            "2020-Feb.csv.gz",
            "2020-Mar.csv.gz",
            "notes.txt",
        ],
    )

    selected = bronze_module.select_raw_files(str(tmp_path), window_profile="training")

    assert [path.name for path in selected] == [
        "2019-Oct.csv.gz",
    ]


def test_select_raw_files_replay_window(tmp_path, bronze_module):
    _create_raw_files(
        tmp_path,
            [
                "2019-Oct.csv.gz",
                "2019-Nov.csv.gz",
                "2019-Dec.csv.gz",
                "2020-Jan.csv.gz",
                "2020-Feb.csv.gz",
                "2020-Mar.csv.gz",
                "2020-Apr.csv.gz",
        ],
    )

    selected = bronze_module.select_raw_files(str(tmp_path), window_profile="replay")

    assert [path.name for path in selected] == ["2019-Nov.csv.gz"]


def test_select_raw_files_dev_smoke_window(tmp_path, bronze_module):
    _create_raw_files(
        tmp_path,
        [
            "2019-Sep.csv.gz",
            "2019-Oct.csv.gz",
            "2019-Nov.csv.gz",
            "2019-Dec.csv.gz",
            "2020-Mar.csv.gz",
        ],
    )

    selected = bronze_module.select_raw_files(str(tmp_path), window_profile="dev_smoke")

    assert [path.name for path in selected] == ["2019-Oct.csv.gz"]


def test_custom_window_selection_is_inclusive(tmp_path, bronze_module):
    _create_raw_files(
        tmp_path,
        [
            "2019-Oct.csv.gz",
            "2019-Nov.csv.gz",
            "2019-Dec.csv.gz",
            "2020-Jan.csv.gz",
            "2020-Feb.csv.gz",
        ],
    )

    selected = bronze_module.select_raw_files(
        str(tmp_path),
        window_profile="all",
        window_start="2019-11",
        window_end="2020-01",
    )

    assert [path.name for path in selected] == [
        "2019-Nov.csv.gz",
        "2019-Dec.csv.gz",
        "2020-Jan.csv.gz",
    ]


def test_select_raw_files_raises_when_no_match(tmp_path, bronze_module):
    _create_raw_files(
        tmp_path,
        [
            "2018-Jan.csv.gz",
            "2018-Feb.csv.gz",
        ],
    )

    with pytest.raises(FileNotFoundError):
        bronze_module.select_raw_files(str(tmp_path), window_profile="training")


def test_select_raw_files_all_profile_skips_unsupported_names(tmp_path, bronze_module):
    _create_raw_files(
        tmp_path,
        [
            "2020-Jan.csv.gz",
            "bad_name.csv",
            "2020-Feb.csv",
            "manifest.json",
        ],
    )

    selected = bronze_module.select_raw_files(str(tmp_path), window_profile="all")

    assert [path.name for path in selected] == ["2020-Jan.csv.gz", "2020-Feb.csv"]


def test_resolve_raw_window_bounds_requires_both_bounds(bronze_module):
    with pytest.raises(ValueError):
        bronze_module.resolve_raw_window_bounds(window_start="2019-10")


def test_resolve_raw_window_bounds_rejects_inverted_custom_bounds(bronze_module):
    with pytest.raises(ValueError):
        bronze_module.resolve_raw_window_bounds(
            window_start="2020-02", window_end="2019-10"
        )


def test_extract_raw_file_month_handles_supported_and_unsupported_names(
    tmp_path, bronze_module
):
    supported = tmp_path / "2020-Apr.csv.gz"
    unsupported = tmp_path / "manifest.json"
    supported.touch()
    unsupported.touch()

    assert bronze_module.extract_raw_file_month(supported) is not None
    assert bronze_module.extract_raw_file_month(unsupported) is None
