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
from collections.abc import Iterator
from dataclasses import dataclass
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
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


DEFAULT_SILVER_BATCH_SIZE = 200000

SILVER_STRING_COLUMNS = [
    "event_type",
    "product_id",
    constants.FIELD_CATEGORY_ID,
    "user_id",
    "user_session",
    "category_code",
    "brand",
]


@dataclass
class SilverPipelineStats:
    """Counters emitted by the silver streaming pipeline."""

    input_rows: int = 0
    rejected_rows: int = 0
    duplicate_rows: int = 0
    output_rows: int = 0


def get_silver_sort_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the deterministic silver sort columns in canonical order.

    Args:
        df: Input DataFrame. Kept in the signature so callers can pair this API
            with validate_silver_sort_columns(df).

    Returns:
        Ordered column names used for deterministic sorting.
    """
    return list(constants.DEDUP_KEY_FIELDS)


def validate_silver_sort_columns(df: pd.DataFrame) -> None:
    """
    Fail fast if the DataFrame cannot be sorted deterministically.

    Args:
        df: Input DataFrame

    Raises:
        ValueError: If one or more required sort columns are missing.
    """
    for column in get_silver_sort_columns(df):
        if column not in df.columns:
            raise ValueError(f"missing: {column}")


def normalize_category_code(
    df: pd.DataFrame,
    policy: str = "keep",
    fill_value: str = "unknown",
) -> pd.DataFrame:
    """
    Normalize nullable category_code values according to the selected policy.

    Args:
        df: Input DataFrame
        policy: "keep" preserves nulls, "fill" replaces nulls with fill_value
        fill_value: Value used when policy="fill"

    Returns:
        DataFrame with category_code normalized.

    Raises:
        ValueError: If policy is not supported.
    """
    if policy == "keep":
        return df
    if policy == "fill":
        if "category_code" in df.columns:
            df["category_code"] = df["category_code"].fillna(fill_value)
        return df

    raise ValueError(f"Invalid category_code policy: {policy}")


def read_bronze_parquet(bronze_path: str) -> pd.DataFrame:
    """
    Read bronze parquet artifact.

    Args:
        bronze_path: Path to bronze parquet file or parquet dataset directory

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


def iter_bronze_batches(
    bronze_path: str,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """
    Yield record batches from a bronze parquet file or dataset directory.

    Args:
        bronze_path: Path to bronze parquet file or dataset directory
        batch_size: Maximum rows per yielded batch

    Yields:
        PyArrow record batches
    """
    bronze_path = Path(bronze_path)

    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze parquet not found: {bronze_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    logger.info(f"Reading bronze artifact in batches: {bronze_path}")
    if bronze_path.is_file():
        parquet_file = pq.ParquetFile(bronze_path)
        yield from parquet_file.iter_batches(batch_size=batch_size)
        return

    dataset = ds.dataset(bronze_path, format="parquet")
    yield from dataset.to_batches(batch_size=batch_size)


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


def clean_silver_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Apply per-batch silver dtype normalization and row-level validations.

    Args:
        df: Input bronze batch converted to pandas

    Returns:
        (clean_df, rejected_rows): Filtered batch and number of rejected rows
    """
    df = enforce_silver_dtypes(df)
    df = normalize_category_code(df, policy="keep")

    total_rejected = 0
    df, rejected = check_required_fields(df)
    total_rejected += rejected

    df, rejected = check_price_validity(df)
    total_rejected += rejected

    return df, total_rejected


def sort_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort records deterministically by canonical event key columns.

    Args:
        df: Input DataFrame

    Returns:
        Sorted DataFrame
    """
    validate_silver_sort_columns(df)
    sort_columns = get_silver_sort_columns(df)
    logger.info(f"Sorting deterministically by {sort_columns}...")
    df = df.sort_values(
        by=sort_columns,
        ascending=True,
    ).reset_index(drop=True)
    logger.info(f"  ✓ Sorted {len(df)} records")

    return df


def empty_silver_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with columns ordered by SILVER_SCHEMA."""
    table = pa.Table.from_batches([], schema=schemas.SILVER_SCHEMA)
    return table.to_pandas()


def write_cleaned_silver_batch_parts(
    bronze_path: str,
    parts_dir: str,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> SilverPipelineStats:
    """
    Stream bronze batches through per-row silver validations into parquet parts.

    Global operations such as canonical deduplication and deterministic sorting
    are intentionally deferred until all parts have been written.
    """
    parts_path = Path(parts_dir)
    parts_path.mkdir(parents=True, exist_ok=True)

    stats = SilverPipelineStats()
    silver_fields = [field.name for field in schemas.SILVER_SCHEMA]
    part_index = 0

    for batch in iter_bronze_batches(bronze_path, batch_size=batch_size):
        stats.input_rows += batch.num_rows
        df = batch.to_pandas()
        df, rejected = clean_silver_batch(df)
        stats.rejected_rows += rejected

        if df.empty:
            continue

        df = df[silver_fields]
        table = pa.Table.from_pandas(df, schema=schemas.SILVER_SCHEMA)
        part_path = parts_path / f"part-{part_index:05d}.parquet"
        pq.write_table(table, part_path, compression="snappy")
        logger.info(f"  Wrote cleaned silver batch part: {part_path}")
        part_index += 1

    return stats


def finalize_silver_parts(
    parts_dir: str,
    output_path: str,
    stats: SilverPipelineStats,
) -> pd.DataFrame:
    """
    Apply global silver operations to cleaned parts and write final output.

    Deduplication and sorting are global because correctness depends on seeing
    every event key and all tie-breaker columns across batch boundaries.
    """
    parts_path = Path(parts_dir)
    if any(parts_path.glob("*.parquet")):
        df = read_bronze_parquet(str(parts_path))
        df = enforce_silver_dtypes(df)
    else:
        df = empty_silver_dataframe()

    df, duplicates = deduplicate_events(df)
    stats.duplicate_rows += duplicates
    df = sort_deterministic(df)
    write_silver_parquet(df, output_path)

    stats.output_rows = len(df)
    return df


def run_silver_pipeline(
    bronze_path: str,
    output_path: str,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> SilverPipelineStats:
    """
    Run the silver pipeline using batch-oriented I/O and a global finalization.
    """
    with TemporaryDirectory(prefix="silver-cleaned-") as temp_dir:
        parts_dir = Path(temp_dir) / "parts"
        stats = write_cleaned_silver_batch_parts(
            bronze_path,
            str(parts_dir),
            batch_size=batch_size,
        )
        finalize_silver_parts(str(parts_dir), output_path, stats)
        return stats


def write_silver_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Write DataFrame to Silver Parquet file or dataset directory.

    Args:
        df: Input DataFrame
        output_path: Path to output parquet file, or dataset directory when the
            path does not end in .parquet
    """
    output_path = Path(output_path)

    if (
        output_path.suffix != ".parquet"
        and output_path.exists()
        and any(output_path.iterdir())
    ):
        raise FileExistsError(f"Silver dataset directory is not empty: {output_path}")

    # Reorder columns to match schema order
    silver_fields = [field.name for field in schemas.SILVER_SCHEMA]
    df = df[silver_fields]

    # Convert to PyArrow table with schema
    table = pa.Table.from_pandas(df, schema=schemas.SILVER_SCHEMA)

    if output_path.suffix == ".parquet":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="snappy")
        logger.info(f"✓ Wrote silver artifact: {output_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_file = output_path / "part-000.parquet"
    pq.write_table(table, dataset_file, compression="snappy")
    logger.info(f"✓ Wrote silver dataset artifact: {dataset_file}")


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SILVER_BATCH_SIZE,
        help=f"Rows per bronze read batch (default: {DEFAULT_SILVER_BATCH_SIZE})",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SILVER PIPELINE: Bronze Parquet → Silver Parquet")
    logger.info("=" * 70)

    try:
        logger.info("\n1. Streaming bronze artifact into cleaned silver parts...")
        stats = run_silver_pipeline(
            args.input,
            args.output,
            batch_size=args.batch_size,
        )

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("SILVER PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Input rows:     {stats.input_rows}")
        logger.info(f"Rejected:       {stats.rejected_rows}")
        logger.info(f"Duplicates:     {stats.duplicate_rows}")
        logger.info(f"Output rows:    {stats.output_rows}")
        logger.info(f"Output file:    {args.output}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
