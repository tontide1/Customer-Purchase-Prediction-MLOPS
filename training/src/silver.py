"""
Silver layer pipeline: Bronze Parquet → Silver Parquet.

Cleans and prepares bronze data for modeling:
- Removes records with missing required fields
- Removes records with price <= 0
- Sorts deterministically by session/time plus stable tie-breakers
- Writes either a single parquet file or a parquet dataset directory
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
    "user_id",
    "user_session",
    "category_code",
    "brand",
]

SILVER_SORT_TIE_BREAKERS = ["event_type", "product_id", "user_id"]


def is_parquet_target(path: Path) -> bool:
    """Return True when the path is intended to be a parquet file."""
    return path.suffix == ".parquet"


def is_nonempty_directory(path: Path) -> bool:
    """Return True when a directory already contains files."""
    return path.exists() and path.is_dir() and any(path.iterdir())


def get_silver_sort_columns(df: pd.DataFrame) -> list[str]:
    """Build the deterministic sort key list used by silver."""
    return [
        "user_session",
        constants.FIELD_SOURCE_EVENT_TIME,
        *SILVER_SORT_TIE_BREAKERS,
    ]


def validate_silver_sort_columns(df: pd.DataFrame) -> None:
    """Fail fast when deterministic sort columns are missing."""
    required_sort_columns = get_silver_sort_columns(df)
    missing_columns = [
        column for column in required_sort_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Silver sort requires columns: "
            f"{', '.join(required_sort_columns)}; missing: {', '.join(missing_columns)}"
        )


def read_bronze_parquet(bronze_path: str) -> pd.DataFrame:
    """
    Read bronze parquet artifact.

    Args:
        bronze_path: Path to a bronze parquet file or parquet dataset directory

    Returns:
        DataFrame with bronze data
    """
    bronze_path = Path(bronze_path)

    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze parquet not found: {bronze_path}")

    if bronze_path.is_dir():
        logger.info(f"Reading bronze dataset directory: {bronze_path}")
    else:
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


def normalize_category_code(
    df: pd.DataFrame,
    policy: str = "keep",
    fill_value: str = "unknown",
) -> pd.DataFrame:
    """Normalize category_code according to the selected policy."""
    if "category_code" not in df.columns:
        return df

    normalized_policy = policy.strip().lower()
    if normalized_policy == "keep":
        return df

    if normalized_policy == "fill":
        df["category_code"] = df["category_code"].fillna(fill_value).astype("string")
        return df

    raise ValueError("category_code_policy must be 'keep' or 'fill'")


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
    Remove records with invalid prices (price <= 0 or NaN).

    Args:
        df: Input DataFrame

    Returns:
        (valid_df, num_rejected): Filtered DataFrame and count of rejected records
    """
    logger.info("Checking price validity...")

    # Records with missing or invalid price are rejected
    mask = (df["price"].notna()) & (df["price"] > constants.DEFAULT_PRICE_THRESHOLD)
    num_rejected = (~mask).sum()

    if num_rejected > 0:
        logger.warning(f"  Rejected {num_rejected} records with invalid price")

    return df[mask].copy(), num_rejected


def sort_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort records deterministically by session/time plus stable tie-breakers.

    Args:
        df: Input DataFrame

    Returns:
        Sorted DataFrame
    """
    validate_silver_sort_columns(df)
    sort_columns = get_silver_sort_columns(df)
    logger.info("Sorting deterministically by %s...", "+".join(sort_columns))
    df = df.sort_values(
        by=sort_columns,
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)
    logger.info(f"  ✓ Sorted {len(df)} records")

    return df


def write_silver_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Write a silver parquet file or parquet dataset.

    Args:
        df: Input DataFrame
        output_path: Path to output parquet file
    """
    output_path = Path(output_path)

    if is_parquet_target(output_path):
        if output_path.exists() and output_path.is_dir():
            raise FileExistsError(
                f"Output path is a directory but file target was requested: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if is_nonempty_directory(output_path):
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_path}"
            )
        output_path.mkdir(parents=True, exist_ok=True)

    # Reorder columns to match schema order
    silver_fields = [field.name for field in schemas.SILVER_SCHEMA]
    df = df[silver_fields]

    # Convert to PyArrow table with schema
    table = pa.Table.from_pandas(df, schema=schemas.SILVER_SCHEMA)

    if is_parquet_target(output_path):
        pq.write_table(table, output_path, compression="snappy")
        logger.info(f"✓ Wrote silver artifact: {output_path}")
    else:
        pq.write_to_dataset(table, root_path=output_path, compression="snappy")
        logger.info(f"✓ Wrote silver dataset: {output_path}")


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
        help=(
            f"Path to silver output parquet file or dataset directory "
            f"(default: {Config.SILVER_DATA_PATH})"
        ),
    )
    parser.add_argument(
        "--category-code-policy",
        choices=["keep", "fill"],
        default="keep",
        help="Policy for null category_code values (default: keep)",
    )
    parser.add_argument(
        "--category-code-fill-value",
        default="unknown",
        help="Value used when category_code-policy=fill (default: unknown)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SILVER PIPELINE: Bronze Parquet → Silver Parquet")
    logger.info("=" * 70)
    logger.info(f"Category code policy: {args.category_code_policy}")

    try:
        # Read bronze artifact
        logger.info(f"\n1. Reading bronze artifact from {args.input}...")
        df = read_bronze_parquet(args.input)
        initial_count = len(df)
        total_rejected = 0

        # Normalize dtypes before validation and write
        df = enforce_silver_dtypes(df)

        # Normalize nullable category semantics explicitly
        df = normalize_category_code(
            df,
            policy=args.category_code_policy,
            fill_value=args.category_code_fill_value,
        )

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

        # Sort deterministically
        logger.info("\n4. Sorting deterministically...")
        df = sort_deterministic(df)

        # Write output
        logger.info(f"\n5. Writing silver artifact...")
        write_silver_parquet(df, args.output)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("SILVER PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Input rows:     {initial_count}")
        logger.info(f"Rejected:       {total_rejected}")
        logger.info(f"Output rows:    {len(df)}")
        logger.info(f"Output file:    {args.output}")
        logger.info(f"Category code policy: {args.category_code_policy}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
