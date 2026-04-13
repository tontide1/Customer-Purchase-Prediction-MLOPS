"""
Bronze layer pipeline: Raw CSV → Bronze Parquet.

Transforms raw event data from CSV to standardized Parquet format.
- Renames event_time → source_event_time
- Validates event_type
- Rejects invalid records
- Outputs immutable parquet artifact
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

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


def read_raw_csvs(raw_dir: str) -> pd.DataFrame:
    """
    Read all CSV files from raw directory and concatenate.

    Args:
        raw_dir: Path to raw data directory

    Returns:
        Concatenated DataFrame with all raw events
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    csv_files = list(raw_path.glob("*.csv"))
    gz_files = list(raw_path.glob("*.csv.gz"))

    all_files = csv_files + gz_files

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    logger.info(f"Found {len(all_files)} raw file(s) to process")

    dfs = []
    for file_path in sorted(all_files):
        logger.info(f"Reading {file_path.name}...")
        try:
            if str(file_path).endswith(".gz"):
                df = pd.read_csv(file_path, compression="gzip")
            else:
                df = pd.read_csv(file_path)
            dfs.append(df)
            logger.info(f"  ✓ Read {len(df)} rows from {file_path.name}")
        except Exception as e:
            logger.error(f"  ✗ Error reading {file_path.name}: {e}")
            raise

    return pd.concat(dfs, ignore_index=True)


def parse_event_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse event_time string to timestamp.

    Input format: "2019-10-01 00:00:00 UTC"

    Args:
        df: Input DataFrame with event_time column

    Returns:
        DataFrame with parsed event_time as timestamp
    """
    # Remove ' UTC' suffix if present, then parse
    df[constants.FIELD_EVENT_TIME] = pd.to_datetime(
        df[constants.FIELD_EVENT_TIME].str.replace(" UTC", "")
    )
    return df


def validate_event_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Filter records with valid event_type.

    Args:
        df: Input DataFrame

    Returns:
        (valid_df, num_rejected): Filtered DataFrame and count of rejected records
    """
    mask = df["event_type"].isin(constants.ALLOWED_EVENT_TYPES)
    num_rejected = (~mask).sum()

    if num_rejected > 0:
        logger.warning(f"Rejecting {num_rejected} records with invalid event_type")
        invalid_types = df.loc[~mask, "event_type"].unique()
        logger.warning(f"  Invalid types found: {invalid_types}")

    return df[mask].copy(), num_rejected


def transform_to_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw DataFrame to bronze schema.

    - Rename event_time → source_event_time
    - Select only schema fields
    - Ensure correct dtypes

    Args:
        df: Input raw DataFrame (with parsed event_time)

    Returns:
        Bronze-formatted DataFrame
    """
    # Rename event_time to source_event_time
    df = df.rename(
        columns={constants.FIELD_EVENT_TIME: constants.FIELD_SOURCE_EVENT_TIME}
    )

    # Select only schema fields (in order)
    bronze_fields = list(schemas.get_bronze_fields())
    df = df[bronze_fields]

    return df


def write_bronze_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Write DataFrame to Parquet file with bronze schema.

    Args:
        df: Input DataFrame
        output_path: Path to output parquet file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PyArrow table with schema
    table = pa.Table.from_pandas(df, schema=schemas.BRONZE_SCHEMA)

    # Write parquet
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"✓ Wrote bronze artifact: {output_path}")


def main():
    """Main entry point for bronze pipeline."""
    parser = argparse.ArgumentParser(
        description="Transform raw CSV data to bronze parquet format"
    )
    parser.add_argument(
        "--input",
        default=Config.RAW_DATA_PATH,
        help=f"Path to raw data directory (default: {Config.RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        default=Config.BRONZE_DATA_PATH,
        help=f"Path to bronze output parquet file (default: {Config.BRONZE_DATA_PATH})",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("BRONZE PIPELINE: Raw CSV → Bronze Parquet")
    logger.info("=" * 70)

    try:
        # Read raw CSVs
        logger.info(f"\n1. Reading raw data from {args.input}...")
        df_raw = read_raw_csvs(args.input)
        initial_count = len(df_raw)
        logger.info(f"   Total rows read: {initial_count}")

        # Parse event_time
        logger.info("\n2. Parsing event_time...")
        df_raw = parse_event_time(df_raw)
        logger.info("   ✓ Event time parsed")

        # Validate event_type
        logger.info("\n3. Validating event_type...")
        df_valid, num_rejected = validate_event_type(df_raw)
        logger.info(f"   Valid records: {len(df_valid)}")
        logger.info(f"   Rejected records: {num_rejected}")

        # Transform to bronze schema
        logger.info("\n4. Transforming to bronze schema...")
        df_bronze = transform_to_bronze(df_valid)
        logger.info(f"   ✓ Renamed event_time → {constants.FIELD_SOURCE_EVENT_TIME}")

        # Write output
        logger.info(f"\n5. Writing bronze artifact...")
        write_bronze_parquet(df_bronze, args.output)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("BRONZE PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Input rows:     {initial_count}")
        logger.info(f"Rejected:       {num_rejected}")
        logger.info(f"Output rows:    {len(df_bronze)}")
        logger.info(f"Output file:    {args.output}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
