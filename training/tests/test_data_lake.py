"""
Foundation tests for data lake pipeline (Week 1).

Tests core transformations:
- Raw → Bronze: event_time renamed to source_event_time
- Bronze → Silver: cleaning and sorting
- Timestamp contracts preserved
- Deterministic ordering
- Invalid records rejected
"""

import datetime as dt
import importlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from shared import constants, schemas
from training.src.bronze import (
    discover_raw_files,
    ensure_not_simulation_raw_input,
    parse_event_time,
    transform_to_bronze,
    validate_event_type,
)
import training.src.config as config_module
from training.src.config import Config
from training.src.silver import (
    check_price_validity,
    check_required_fields,
    deduplicate_events,
    sort_deterministic,
)


@pytest.fixture
def sample_raw_df() -> pl.DataFrame:
    """Minimal valid raw frame for bronze tests."""
    return pl.DataFrame(
        {
            "event_time": [
                "2019-10-01 10:00:00 UTC",
                "2019-10-01 10:01:00 UTC",
                "2019-10-01 10:02:00 UTC",
            ],
            "event_type": ["view", "cart", "purchase"],
            "product_id": ["1", "2", "3"],
            "category_id": ["cat1", "cat2", "cat3"],
            "category_code": ["code1", "code2", "code3"],
            "brand": ["brand1", "brand2", "brand3"],
            "price": [10.0, 20.0, 30.0],
            "user_id": ["u1", "u2", "u3"],
            "user_session": ["session1", "session2", "session3"],
        },
    )


@pytest.fixture
def sample_bronze_df() -> pl.DataFrame:
    """Minimal bronze-like frame."""
    times = [
        dt.datetime(2019, 10, 1, 10, 0, 0),
        dt.datetime(2019, 10, 1, 10, 1, 0),
        dt.datetime(2019, 10, 1, 10, 2, 0),
    ]
    return pl.DataFrame(
        {
            constants.FIELD_SOURCE_EVENT_TIME: times,
            "event_type": ["view", "cart", "purchase"],
            "product_id": ["1", "2", "3"],
            constants.FIELD_CATEGORY_ID: ["cat1", "cat2", "cat3"],
            "category_code": ["code1", "code2", "code3"],
            "brand": ["brand1", "brand2", "brand3"],
            "price": [10.0, 20.0, 30.0],
            "user_id": ["u1", "u2", "u3"],
            "user_session": ["session1", "session2", "session3"],
        },
    )


class TestBronzeLayer:
    def test_event_time_parsing(self, sample_raw_df: pl.DataFrame) -> None:
        df = parse_event_time(sample_raw_df.clone())
        assert df.schema[constants.FIELD_EVENT_TIME] == pl.Datetime("us")
        ts = df[constants.FIELD_EVENT_TIME][0]
        assert isinstance(ts, dt.datetime)
        assert ts.replace(tzinfo=None) == dt.datetime(
            2019, 10, 1, 10, 0, 0,
        )

    def test_event_time_to_source_event_time_rename(self, sample_raw_df: pl.DataFrame) -> None:
        parsed = parse_event_time(sample_raw_df.clone())
        df_bronze = transform_to_bronze(parsed)
        assert constants.FIELD_SOURCE_EVENT_TIME in df_bronze.columns
        assert constants.FIELD_EVENT_TIME not in df_bronze.columns

    def test_category_id_preserved_in_bronze(self, sample_raw_df: pl.DataFrame) -> None:
        parsed = parse_event_time(sample_raw_df.clone())
        df_bronze = transform_to_bronze(parsed)

        assert constants.FIELD_CATEGORY_ID in df_bronze.columns
        assert df_bronze[constants.FIELD_CATEGORY_ID].to_list() == ["cat1", "cat2", "cat3"]

    def test_valid_event_type_kept(self, sample_raw_df: pl.DataFrame) -> None:
        parsed = parse_event_time(sample_raw_df.clone())
        df_valid, num_rejected = validate_event_type(parsed)
        assert df_valid.height == 3
        assert num_rejected == 0

    def test_invalid_event_type_rejected(self) -> None:
        df = pl.DataFrame(
            {"event_type": ["view", "invalid_type", "cart"]},
        )

        df_valid, num_rejected = validate_event_type(df)

        assert df_valid.height == 2
        assert num_rejected == 1
        assert "invalid_type" not in df_valid.select("event_type").to_series().to_list()

    def test_bronze_schema_applied(self, sample_raw_df: pl.DataFrame) -> None:
        parsed = parse_event_time(sample_raw_df.clone())
        df_bronze = transform_to_bronze(parsed)
        expected_fields = set(schemas.get_bronze_fields())
        actual_fields = set(df_bronze.columns)
        assert expected_fields == actual_fields


class TestSilverLayer:
    def test_required_fields_check(self, sample_bronze_df: pl.DataFrame) -> None:
        uids = sample_bronze_df["user_id"].to_list()
        uids[0] = None
        df_modified = sample_bronze_df.with_columns(pl.Series(name="user_id", values=uids))

        df_valid, num_rejected = check_required_fields(df_modified)

        assert df_valid.height == 2
        assert num_rejected == 1

    def test_price_validity_check(self, sample_bronze_df: pl.DataFrame) -> None:
        prices = sample_bronze_df["price"].to_list()
        prices[0] = 0.0
        prices[1] = -5.0
        prices[2] = 1.0
        df_modified = sample_bronze_df.with_columns(
            pl.Series(name="price", values=prices),
        )

        df_valid, num_rejected = check_price_validity(df_modified)

        assert df_valid.height == 1
        assert num_rejected == 2

    def test_null_price_is_allowed(self) -> None:
        df = pl.DataFrame(
            {
                constants.FIELD_SOURCE_EVENT_TIME: [
                    dt.datetime(2019, 10, 1, 10, 0, 0),
                    dt.datetime(2019, 10, 1, 10, 1, 0),
                ],
                "event_type": ["view", "cart"],
                "product_id": ["1", "2"],
                constants.FIELD_CATEGORY_ID: ["cat1", "cat2"],
                "category_code": ["code1", "code2"],
                "brand": ["brand1", "brand2"],
                "price": [None, 10.0],
                "user_id": ["u1", "u2"],
                "user_session": ["s1", "s1"],
            },
            schema_overrides={"price": pl.Float64},
        )

        df_valid, num_rejected = check_price_validity(df)

        assert df_valid.height == 2
        assert num_rejected == 0

    def test_deterministic_sort(self, sample_bronze_df: pl.DataFrame) -> None:
        df_unsorted = pl.DataFrame(
            {
                constants.FIELD_SOURCE_EVENT_TIME: [
                    dt.datetime(2019, 10, 1, 10, 2, 0),
                    dt.datetime(2019, 10, 1, 10, 0, 0),
                    dt.datetime(2019, 10, 1, 10, 1, 0),
                ],
                "event_type": ["view", "cart", "purchase"],
                "product_id": ["1", "2", "3"],
                constants.FIELD_CATEGORY_ID: ["cat1", "cat2", "cat3"],
                "category_code": ["code1", "code2", "code3"],
                "brand": ["brand1", "brand2", "brand3"],
                "price": [10.0, 20.0, 30.0],
                "user_id": ["u1", "u2", "u3"],
                "user_session": ["session2", "session1", "session1"],
            },
        )

        df_sorted = sort_deterministic(df_unsorted)

        sessions = df_sorted.select("user_session").to_series().to_list()
        assert sessions == ["session1", "session1", "session2"]

        session1 = df_sorted.filter(pl.col("user_session") == "session1").select(
            constants.FIELD_SOURCE_EVENT_TIME,
        )
        vals = session1.to_series().to_list()
        assert vals == sorted(vals)

    def test_silver_deterministic_when_repeated(self) -> None:
        df1 = pl.DataFrame(
            {
                constants.FIELD_SOURCE_EVENT_TIME: [
                    dt.datetime(2019, 10, 1, 10, 1, 0),
                    dt.datetime(2019, 10, 1, 10, 0, 0),
                ],
                "event_type": ["view", "cart"],
                "product_id": ["1", "2"],
                constants.FIELD_CATEGORY_ID: ["cat1", "cat2"],
                "category_code": ["code1", "code2"],
                "brand": ["brand1", "brand2"],
                "price": [10.0, 20.0],
                "user_id": ["u1", "u2"],
                "user_session": ["s1", "s1"],
            },
        )

        s1 = sort_deterministic(df1.clone())
        s2 = sort_deterministic(df1.clone())

        assert_frame_equal(s1, s2)

    def test_canonical_dedup_removes_duplicate_events(self) -> None:
        df = pl.DataFrame(
            {
                constants.FIELD_SOURCE_EVENT_TIME: [
                    dt.datetime(2019, 10, 1, 10, 0, 0),
                    dt.datetime(2019, 10, 1, 10, 0, 0),
                    dt.datetime(2019, 10, 1, 10, 1, 0),
                ],
                "event_type": ["view", "view", "cart"],
                "product_id": ["1", "1", "2"],
                constants.FIELD_CATEGORY_ID: ["cat1", "cat1", "cat2"],
                "category_code": ["code1", "code1", "code2"],
                "brand": ["brand1", "brand1", "brand2"],
                "price": [10.0, 10.0, 20.0],
                "user_id": ["u1", "u1", "u2"],
                "user_session": ["s1", "s1", "s1"],
            },
        )

        df_deduped, num_duplicates = deduplicate_events(df)

        assert df_deduped.height == 2
        assert num_duplicates == 1
        assert df_deduped.select("event_type").to_series().to_list() == ["view", "cart"]


class TestTimestampContract:
    def test_raw_uses_event_time(self, sample_raw_df: pl.DataFrame) -> None:
        assert constants.FIELD_EVENT_TIME in sample_raw_df.columns

    def test_bronze_uses_source_event_time(self, sample_bronze_df: pl.DataFrame) -> None:
        assert constants.FIELD_SOURCE_EVENT_TIME in sample_bronze_df.columns

    def test_timestamp_preserved_through_layers(self, sample_raw_df: pl.DataFrame) -> None:
        parsed = parse_event_time(sample_raw_df.clone())
        df_bronze = transform_to_bronze(parsed)
        first = df_bronze.item(0, constants.FIELD_SOURCE_EVENT_TIME)
        assert first == dt.datetime(2019, 10, 1, 10, 0, 0)


class TestPipelineIntegration:
    def test_end_to_end_clean_data(self, sample_raw_df: pl.DataFrame) -> None:
        parsed = parse_event_time(sample_raw_df.clone())

        df_valid, _ = validate_event_type(parsed)
        df_bronze = transform_to_bronze(df_valid)

        assert df_bronze.height == 3
        assert constants.FIELD_SOURCE_EVENT_TIME in df_bronze.columns

        df_silver, _ = check_required_fields(df_bronze)
        assert df_silver.height == 3

        df_silver, _ = check_price_validity(df_silver)
        assert df_silver.height == 3

    def test_pipeline_with_mixed_invalid_data(self) -> None:
        raw_df = pl.DataFrame(
            {
                "event_time": [
                    "2019-10-01 10:00:00 UTC",
                    "2019-10-01 10:01:00 UTC",
                    "2019-10-01 10:02:00 UTC",
                    "2019-10-01 10:03:00 UTC",
                ],
                "event_type": ["view", "invalid", "cart", "purchase"],
                "product_id": ["1", "2", "3", "4"],
                constants.FIELD_CATEGORY_ID: ["cat1", "cat2", "cat3", "cat4"],
                "category_code": ["code1", "code2", "code3", "code4"],
                "brand": ["b1", "b2", "b3", "b4"],
                "price": [10.0, 20.0, 0.0, 30.0],
                "user_id": ["u1", "u2", "u3", None],
                "user_session": ["s1", "s2", "s1", "s3"],
            },
        )

        parsed = parse_event_time(raw_df)
        df_valid, invalid_event_types = validate_event_type(parsed)

        assert df_valid.height == 3
        assert invalid_event_types == 1

        df_bronze = transform_to_bronze(df_valid)

        df_silver, missing_fields = check_required_fields(df_bronze)

        assert missing_fields == 1
        assert df_silver.height == 2

        df_silver, invalid_prices = check_price_validity(df_silver)

        assert invalid_prices == 1
        assert df_silver.height == 1


class TestDataPathConfig:
    def test_data_strategy_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRAIN_RAW_DATA_PATH", raising=False)
        monkeypatch.delenv("SIMULATION_RAW_DATA_PATH", raising=False)
        monkeypatch.delenv("RETRAIN_RAW_DATA_DIR", raising=False)
        monkeypatch.delenv("RETRAIN_DATA_DIR", raising=False)
        monkeypatch.delenv("RETRAIN_WINDOW_DAYS", raising=False)

        fresh_config_module = importlib.reload(config_module)

        assert fresh_config_module.Config.TRAIN_RAW_DATA_PATH == "data/train_raw"
        assert (
            fresh_config_module.Config.SIMULATION_RAW_DATA_PATH
            == "data/simulation_raw/2019-Nov.csv.gz"
        )
        assert fresh_config_module.Config.RETRAIN_RAW_DATA_DIR == "data/retrain_raw"
        assert fresh_config_module.Config.RETRAIN_DATA_DIR == "data/retrain"
        assert fresh_config_module.Config.RETRAIN_WINDOW_DAYS == 7

        settings = fresh_config_module.Config.get_all_settings()
        assert "raw_data_path" not in settings
        assert settings["train_raw_data_path"] == "data/train_raw"
        assert (
            settings["simulation_raw_data_path"]
            == "data/simulation_raw/2019-Nov.csv.gz"
        )

    @pytest.mark.parametrize(
        "input_path",
        [
            "data/simulation_raw",
            Config.SIMULATION_RAW_DATA_PATH,
        ],
    )
    def test_bronze_rejects_simulation_raw_input(self, input_path: str) -> None:
        with pytest.raises(ValueError, match="simulation raw data"):
            ensure_not_simulation_raw_input(input_path)

    def test_bronze_discovery_only_reads_train_raw_directory(self, tmp_path) -> None:
        train_raw = tmp_path / "train_raw"
        simulation_raw = tmp_path / "simulation_raw"
        train_raw.mkdir()
        simulation_raw.mkdir()

        oct_file = train_raw / "2019-Oct.csv.gz"
        nov_file = simulation_raw / "2019-Nov.csv.gz"
        oct_file.write_text("event_time,event_type\n")
        nov_file.write_text("event_time,event_type\n")

        discovered = discover_raw_files(str(train_raw))

        assert discovered == [oct_file]
