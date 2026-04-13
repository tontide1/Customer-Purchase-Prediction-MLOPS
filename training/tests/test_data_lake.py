"""
Foundation tests for data lake pipeline (Week 1).

Tests core transformations:
- Raw → Bronze: event_time renamed to source_event_time
- Bronze → Silver: cleaning and sorting
- Timestamp contracts preserved
- Deterministic ordering
- Invalid records rejected
"""

import pytest
import pandas as pd
import pyarrow as pa
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import constants, schemas
from training.src.bronze import (
    validate_event_type,
    transform_to_bronze,
    parse_event_time,
)
from training.src.silver import (
    check_required_fields,
    check_price_validity,
    sort_deterministic,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_raw_df():
    """Create a minimal valid raw DataFrame for testing."""
    return pd.DataFrame(
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
        }
    )


@pytest.fixture
def sample_bronze_df():
    """Create a minimal valid bronze DataFrame for testing."""
    return pd.DataFrame(
        {
            constants.FIELD_SOURCE_EVENT_TIME: pd.to_datetime(
                [
                    "2019-10-01 10:00:00",
                    "2019-10-01 10:01:00",
                    "2019-10-01 10:02:00",
                ]
            ),
            "event_type": ["view", "cart", "purchase"],
            "product_id": ["1", "2", "3"],
            "category_code": ["code1", "code2", "code3"],
            "brand": ["brand1", "brand2", "brand3"],
            "price": [10.0, 20.0, 30.0],
            "user_id": ["u1", "u2", "u3"],
            "user_session": ["session1", "session2", "session3"],
        }
    )


# ============================================================================
# Bronze Layer Tests
# ============================================================================


class TestBronzeLayer:
    """Tests for raw → bronze transformation."""

    def test_event_time_parsing(self, sample_raw_df):
        """Test that event_time string is parsed to timestamp."""
        df = parse_event_time(sample_raw_df.copy())

        # Check that event_time is now datetime
        assert pd.api.types.is_datetime64_any_dtype(df[constants.FIELD_EVENT_TIME])

        # Check specific timestamp value
        assert df[constants.FIELD_EVENT_TIME].iloc[0] == pd.Timestamp(
            "2019-10-01 10:00:00"
        )

    def test_event_time_to_source_event_time_rename(self, sample_raw_df):
        """Test that event_time is renamed to source_event_time in bronze."""
        df = parse_event_time(sample_raw_df.copy())
        df_bronze = transform_to_bronze(df)

        # Check that source_event_time exists
        assert constants.FIELD_SOURCE_EVENT_TIME in df_bronze.columns

        # Check that old event_time field is gone
        assert constants.FIELD_EVENT_TIME not in df_bronze.columns

    def test_valid_event_type_kept(self, sample_raw_df):
        """Test that records with valid event_type are kept."""
        df_valid, num_rejected = validate_event_type(sample_raw_df)

        assert len(df_valid) == 3
        assert num_rejected == 0

    def test_invalid_event_type_rejected(self):
        """Test that records with invalid event_type are rejected."""
        df = pd.DataFrame(
            {
                "event_type": ["view", "invalid_type", "cart"],
            }
        )

        df_valid, num_rejected = validate_event_type(df)

        assert len(df_valid) == 2
        assert num_rejected == 1
        assert "invalid_type" not in df_valid["event_type"].values

    def test_bronze_schema_applied(self, sample_raw_df):
        """Test that bronze transformation applies correct schema."""
        df = parse_event_time(sample_raw_df.copy())
        df_bronze = transform_to_bronze(df)

        # Check all required fields exist
        expected_fields = set(schemas.get_bronze_fields())
        actual_fields = set(df_bronze.columns)

        assert expected_fields == actual_fields


# ============================================================================
# Silver Layer Tests
# ============================================================================


class TestSilverLayer:
    """Tests for bronze → silver transformation."""

    def test_required_fields_check(self, sample_bronze_df):
        """Test that records missing required fields are rejected."""
        df_modified = sample_bronze_df.copy()
        # Make one user_id missing
        df_modified.loc[0, "user_id"] = None

        df_valid, num_rejected = check_required_fields(df_modified)

        assert len(df_valid) == 2
        assert num_rejected == 1

    def test_price_validity_check(self, sample_bronze_df):
        """Test that records with invalid prices are rejected."""
        df_modified = sample_bronze_df.copy()
        # Set some prices to invalid values
        df_modified.loc[0, "price"] = 0.0  # Invalid: <= 0
        df_modified.loc[1, "price"] = -5.0  # Invalid: <= 0
        df_modified.loc[2, "price"] = None  # Invalid: missing

        df_valid, num_rejected = check_price_validity(df_modified)

        assert len(df_valid) == 0
        assert num_rejected == 3

    def test_deterministic_sort(self, sample_bronze_df):
        """Test that records are sorted deterministically."""
        # Create unsorted data with multiple sessions
        df_unsorted = pd.DataFrame(
            {
                constants.FIELD_SOURCE_EVENT_TIME: pd.to_datetime(
                    [
                        "2019-10-01 10:02:00",  # session2, later time
                        "2019-10-01 10:00:00",  # session1, earlier time
                        "2019-10-01 10:01:00",  # session1, middle time
                    ]
                ),
                "event_type": ["view", "cart", "purchase"],
                "product_id": ["1", "2", "3"],
                "category_code": ["code1", "code2", "code3"],
                "brand": ["brand1", "brand2", "brand3"],
                "price": [10.0, 20.0, 30.0],
                "user_id": ["u1", "u2", "u3"],
                "user_session": ["session2", "session1", "session1"],
            }
        )

        df_sorted = sort_deterministic(df_unsorted)

        # Check that session1 comes before session2
        sessions = df_sorted["user_session"].tolist()
        assert sessions == ["session1", "session1", "session2"]

        # Check that within session1, times are sorted
        session1_times = df_sorted[df_sorted["user_session"] == "session1"][
            constants.FIELD_SOURCE_EVENT_TIME
        ].tolist()
        assert session1_times == sorted(session1_times)

    def test_silver_deterministic_when_repeated(self):
        """Test that silver transformation is deterministic."""
        # Create test data
        df1 = pd.DataFrame(
            {
                constants.FIELD_SOURCE_EVENT_TIME: pd.to_datetime(
                    [
                        "2019-10-01 10:01:00",
                        "2019-10-01 10:00:00",
                    ]
                ),
                "event_type": ["view", "cart"],
                "product_id": ["1", "2"],
                "category_code": ["code1", "code2"],
                "brand": ["brand1", "brand2"],
                "price": [10.0, 20.0],
                "user_id": ["u1", "u2"],
                "user_session": ["s1", "s1"],
            }
        )

        # Sort twice and check results are identical
        df1_sorted1 = sort_deterministic(df1.copy())
        df1_sorted2 = sort_deterministic(df1.copy())

        pd.testing.assert_frame_equal(df1_sorted1, df1_sorted2)


# ============================================================================
# Timestamp Contract Tests
# ============================================================================


class TestTimestampContract:
    """Tests for timestamp field naming contract."""

    def test_raw_uses_event_time(self, sample_raw_df):
        """Test that raw layer uses event_time field."""
        assert constants.FIELD_EVENT_TIME in sample_raw_df.columns

    def test_bronze_uses_source_event_time(self, sample_bronze_df):
        """Test that bronze layer uses source_event_time field."""
        assert constants.FIELD_SOURCE_EVENT_TIME in sample_bronze_df.columns

    def test_timestamp_preserved_through_layers(self, sample_raw_df):
        """Test that timestamp value is preserved raw → bronze."""
        original_time = "2019-10-01 10:00:00 UTC"

        # Parse and transform
        df = parse_event_time(sample_raw_df.copy())
        df_bronze = transform_to_bronze(df)

        # Check that first timestamp is preserved
        bronze_time = df_bronze[constants.FIELD_SOURCE_EVENT_TIME].iloc[0]
        assert bronze_time == pd.Timestamp("2019-10-01 10:00:00")


# ============================================================================
# Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Integration tests for full pipeline flow."""

    def test_end_to_end_clean_data(self, sample_raw_df):
        """Test complete raw → bronze → silver flow on clean data."""
        # Parse event_time
        df = parse_event_time(sample_raw_df.copy())

        # Validate and transform to bronze
        df_valid, _ = validate_event_type(df)
        df_bronze = transform_to_bronze(df_valid)

        # Check bronze output
        assert len(df_bronze) == 3
        assert constants.FIELD_SOURCE_EVENT_TIME in df_bronze.columns

        # Check required fields
        df_silver, _ = check_required_fields(df_bronze)
        assert len(df_silver) == 3

        # Check price validity
        df_silver, _ = check_price_validity(df_silver)
        assert len(df_silver) == 3

    def test_pipeline_with_mixed_invalid_data(self):
        """Test pipeline with mixture of valid and invalid data."""
        raw_df = pd.DataFrame(
            {
                "event_time": [
                    "2019-10-01 10:00:00 UTC",
                    "2019-10-01 10:01:00 UTC",
                    "2019-10-01 10:02:00 UTC",
                    "2019-10-01 10:03:00 UTC",
                ],
                "event_type": ["view", "invalid", "cart", "purchase"],
                "product_id": ["1", "2", "3", "4"],
                "category_code": ["code1", "code2", "code3", "code4"],
                "brand": ["b1", "b2", "b3", "b4"],
                "price": [10.0, 20.0, 0.0, 30.0],  # price[2] is invalid
                "user_id": ["u1", "u2", "u3", None],  # user_id[3] is missing
                "user_session": ["s1", "s2", "s1", "s3"],
            }
        )

        # Bronze stage
        df = parse_event_time(raw_df.copy())
        df_valid, invalid_event_types = validate_event_type(df)

        # Only 3 valid event types
        assert len(df_valid) == 3
        assert invalid_event_types == 1

        df_bronze = transform_to_bronze(df_valid)

        # Silver stage
        df_silver, missing_fields = check_required_fields(df_bronze)

        # One record has missing user_id
        assert missing_fields == 1
        assert len(df_silver) == 2

        df_silver, invalid_prices = check_price_validity(df_silver)

        # One record has price = 0
        assert invalid_prices == 1
        assert len(df_silver) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
