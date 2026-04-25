"""
Silver layer pipeline: Bronze Parquet → Silver Parquet.

Cleans and prepares bronze data for modeling:
- Removes records with missing required fields
- Keeps null prices but removes records with price <= 0
- Removes canonical duplicate events
- Sorts deterministically by user_session + source_event_time
- Outputs clean, ready-for-modeling artifact
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import constants, schemas
from training.src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SILVER_STRING_COLUMNS = [
    "event_type",
    "product_id",
    constants.FIELD_CATEGORY_ID,
    "user_id",
    "user_session",
    "category_code",
    "brand",
]


def read_bronze_parquet(bronze_path: str) -> pd.DataFrame:
    """
    Read bronze parquet artifact.

    Args:
        bronze_path: Path to bronze parquet file

    Returns:
        DataFrame with bronze data
    """
    bronze_path = Path(bronze_path)

    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze parquet not found: {bronze_path}")

    logger.info(f"Reading bronze artifact: {bronze_path}")
    table = pq.read_table(bronze_path)
    df = table.to_pandas()
    logger.info(f"  ✓ Read {len(df)} rows")

    return df


def enforce_silver_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce dtypes compatible with SILVER_SCHEMA.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with normalized dtypes
    """
    if constants.FIELD_SOURCE_EVENT_TIME in df.columns:
        df[constants.FIELD_SOURCE_EVENT_TIME] = pd.to_datetime(
            df[constants.FIELD_SOURCE_EVENT_TIME], errors="coerce"
        )

    for column in SILVER_STRING_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


def check_required_fields(df: pd.DataFrame) -> tuple:
    """
    Check and remove records with missing required fields.

    Args:
        df: Input DataFrame

    Returns:
        (valid_df, num_rejected): Filtered DataFrame and count of rejected records
    """
    logger.info("Checking required fields...")

    # Required fields in silver layer use source_event_time, not event_time
    required_fields = constants.REQUIRED_FIELDS.copy()
    required_fields.discard(constants.FIELD_EVENT_TIME)
    required_fields.add(constants.FIELD_SOURCE_EVENT_TIME)

    mask = ~df[list(required_fields)].isna().any(axis=1)
    num_rejected = (~mask).sum()

    if num_rejected > 0:
        logger.warning(
            f"  Rejected {num_rejected} records with missing required fields"
        )

    return df[mask].copy(), num_rejected


def check_price_validity(df: pd.DataFrame) -> tuple:
    """
    Remove records with invalid non-null prices (price <= 0).

    Args:
        df: Input DataFrame

    Returns:
        (valid_df, num_rejected): Filtered DataFrame and count of rejected records
    """
    logger.info("Checking price validity...")

    # Null price is allowed; only non-null non-positive prices are rejected.
    mask = df["price"].isna() | (df["price"] > constants.DEFAULT_PRICE_THRESHOLD)
    num_rejected = (~mask).sum()

    if num_rejected > 0:
        logger.warning(f"  Rejected {num_rejected} records with invalid price")

    return df[mask].copy(), num_rejected


def deduplicate_events(df: pd.DataFrame) -> tuple:
    """
    Remove duplicate events using the canonical event key.

    The canonical key matches the planned event_id contract:
    user_session | source_event_time | event_type | product_id | user_id
    """
    logger.info("Deduplicating events by canonical key...")
    before_count = len(df)
    deduplicated = df.drop_duplicates(
        subset=list(constants.DEDUP_KEY_FIELDS),
        keep="first",
    ).copy()
    num_duplicates = before_count - len(deduplicated)

    if num_duplicates > 0:
        logger.warning(f"  Removed {num_duplicates} duplicate record(s)")

    return deduplicated, num_duplicates


def sort_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort records deterministically by user_session + source_event_time.

    Args:
        df: Input DataFrame

    Returns:
        Sorted DataFrame
    """
    logger.info("Sorting deterministically by user_session + source_event_time...")
    df = df.sort_values(
        by=["user_session", constants.FIELD_SOURCE_EVENT_TIME],
        ascending=True,
    ).reset_index(drop=True)
    logger.info(f"  ✓ Sorted {len(df)} records")

    return df


def write_silver_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Write DataFrame to Silver Parquet file.

    Args:
        df: Input DataFrame
        output_path: Path to output parquet file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reorder columns to match schema order
    silver_fields = [field.name for field in schemas.SILVER_SCHEMA]
    df = df[silver_fields]

    # Convert to PyArrow table with schema
    table = pa.Table.from_pandas(df, schema=schemas.SILVER_SCHEMA)

    # Write parquet
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"✓ Wrote silver artifact: {output_path}")


def main():
    """Main entry point for silver pipeline."""
    parser = argparse.ArgumentParser(
        description="Clean bronze data and produce silver artifact"
    )
    parser.add_argument(
        "--input",
        default=Config.BRONZE_DATA_PATH,
        help=f"Path to bronze parquet file (default: {Config.BRONZE_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        default=Config.SILVER_DATA_PATH,
        help=f"Path to silver output parquet file (default: {Config.SILVER_DATA_PATH})",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SILVER PIPELINE: Bronze Parquet → Silver Parquet")
    logger.info("=" * 70)

    try:
        # Read bronze artifact
        logger.info(f"\n1. Reading bronze artifact from {args.input}...")
        df = read_bronze_parquet(args.input)
        initial_count = len(df)
        total_rejected = 0
        total_duplicates = 0

        # Normalize dtypes before validation and write
        df = enforce_silver_dtypes(df)

        # Check required fields
        logger.info("\n2. Checking required fields...")
        df, num_rejected = check_required_fields(df)
        total_rejected += num_rejected
        logger.info(f"   Valid records after field check: {len(df)}")

        # Check price validity
        logger.info("\n3. Checking price validity...")
        df, num_rejected = check_price_validity(df)
        total_rejected += num_rejected
        logger.info(f"   Valid records after price check: {len(df)}")

        # Deduplicate canonical events before final sort/write
        logger.info("\n4. Deduplicating canonical events...")
        df, num_duplicates = deduplicate_events(df)
        total_duplicates += num_duplicates
        logger.info(f"   Valid records after dedup: {len(df)}")

        # Sort deterministically
        logger.info("\n5. Sorting deterministically...")
        df = sort_deterministic(df)

        # Write output
        logger.info(f"\n6. Writing silver artifact...")
        write_silver_parquet(df, args.output)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("SILVER PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Input rows:     {initial_count}")
        logger.info(f"Rejected:       {total_rejected}")
        logger.info(f"Duplicates:     {total_duplicates}")
        logger.info(f"Output rows:    {len(df)}")
        logger.info(f"Output file:    {args.output}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
