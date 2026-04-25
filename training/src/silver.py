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
import heapq
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
SILVER_INPUT_ORDER_COLUMN = "_silver_input_order"

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
    next_input_order: int = 0


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


def get_silver_output_fields() -> list[str]:
    """Return silver output columns ordered exactly as SILVER_SCHEMA."""
    return [field.name for field in schemas.SILVER_SCHEMA]


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


def get_silver_working_fields() -> list[str]:
    """Return temporary working columns used during external sort/dedup."""
    return get_silver_output_fields() + [SILVER_INPUT_ORDER_COLUMN]


def add_input_order(df: pd.DataFrame, stats: SilverPipelineStats) -> pd.DataFrame:
    """
    Add a monotonic source-order column for deterministic global deduplication.

    The final output is sorted by canonical event key, but duplicate keys must
    keep the first surviving record from bronze input order. This temporary
    column makes that possible during the external merge step.
    """
    start_order = stats.next_input_order
    end_order = start_order + len(df)
    df[SILVER_INPUT_ORDER_COLUMN] = range(start_order, end_order)
    stats.next_input_order = end_order
    return df


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
    part_index = 0

    for batch in iter_bronze_batches(bronze_path, batch_size=batch_size):
        stats.input_rows += batch.num_rows
        df = batch.to_pandas()
        df, rejected = clean_silver_batch(df)
        stats.rejected_rows += rejected

        if df.empty:
            continue

        df = add_input_order(df, stats)
        df = df[get_silver_working_fields()]
        table = pa.Table.from_pandas(df, preserve_index=False)
        part_path = parts_path / f"part-{part_index:05d}.parquet"
        pq.write_table(table, part_path, compression="snappy")
        logger.info(f"  Wrote cleaned silver batch part: {part_path}")
        part_index += 1

    return stats


def get_silver_merge_columns() -> list[str]:
    """Return columns used for sorted runs and merge ordering."""
    return list(constants.DEDUP_KEY_FIELDS) + [SILVER_INPUT_ORDER_COLUMN]


def sort_silver_working_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a working silver batch by canonical key plus input order."""
    return df.sort_values(
        by=get_silver_merge_columns(),
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)


def write_sorted_silver_runs(
    parts_dir: str,
    runs_dir: str,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> list[Path]:
    """
    Sort cleaned working parts into bounded parquet runs.

    Each run is at most one configured batch in memory. Runs are later merged
    with a heap so finalize memory is O(batch size * run count), not O(total
    cleaned rows).
    """
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    run_paths: list[Path] = []
    source = ds.dataset(parts_dir, format="parquet")
    for run_index, batch in enumerate(source.to_batches(batch_size=batch_size)):
        df = batch.to_pandas()
        if df.empty:
            continue

        df = enforce_silver_dtypes(df)
        df[SILVER_INPUT_ORDER_COLUMN] = pd.to_numeric(
            df[SILVER_INPUT_ORDER_COLUMN],
            errors="raise",
        ).astype("int64")
        df = sort_silver_working_frame(df[get_silver_working_fields()])

        run_path = runs_path / f"run-{run_index:05d}.parquet"
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, run_path, compression="snappy")
        run_paths.append(run_path)
        logger.info(f"  Wrote sorted silver run: {run_path}")

    return run_paths


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict]:
    """Yield rows from a parquet file without loading the full file."""
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        for row in df.to_dict(orient="records"):
            yield row


def row_merge_key(row: dict) -> tuple:
    """Return the heap key for a working silver row."""
    return tuple(row[column] for column in get_silver_merge_columns())


def row_dedup_key(row: dict) -> tuple:
    """Return the canonical deduplication key for a working silver row."""
    return tuple(row[column] for column in constants.DEDUP_KEY_FIELDS)


def iter_merged_unique_rows(
    run_paths: list[Path],
    stats: SilverPipelineStats,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> Iterator[dict]:
    """
    Merge sorted runs and yield globally sorted, deduplicated silver rows.

    Duplicate keys are adjacent during the merge. Because input order is part of
    the merge key, the first duplicate popped is the first surviving record from
    the bronze stream.
    """
    iterators = [iter_parquet_rows(path, batch_size=batch_size) for path in run_paths]
    heap: list[tuple[tuple, int, dict]] = []

    for run_index, iterator in enumerate(iterators):
        try:
            row = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (row_merge_key(row), run_index, row))

    previous_key = None
    while heap:
        _, run_index, row = heapq.heappop(heap)
        current_key = row_dedup_key(row)

        if current_key == previous_key:
            stats.duplicate_rows += 1
        else:
            previous_key = current_key
            yield {field: row[field] for field in get_silver_output_fields()}

        try:
            next_row = next(iterators[run_index])
        except StopIteration:
            continue
        heapq.heappush(heap, (row_merge_key(next_row), run_index, next_row))


def prepare_silver_output_target(output_path: str) -> Path:
    """
    Resolve final parquet target for file or dataset-directory output.

    A path ending in .parquet is treated as a single parquet file. Any other
    path is treated as a dataset directory containing part-000.parquet.
    """
    output_path = Path(output_path)

    if output_path.suffix == ".parquet":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Silver dataset directory is not empty: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / "part-000.parquet"


def write_silver_rows(
    rows: Iterator[dict],
    output_path: str,
    stats: SilverPipelineStats,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> None:
    """Stream final silver rows to parquet without materializing all rows."""
    target_path = prepare_silver_output_target(output_path)
    output_fields = get_silver_output_fields()
    writer = None
    buffer: list[dict] = []

    def flush_buffer() -> None:
        nonlocal writer, buffer
        if not buffer:
            return

        df = pd.DataFrame.from_records(buffer, columns=output_fields)
        df = enforce_silver_dtypes(df)
        table = pa.Table.from_pandas(
            df[output_fields],
            schema=schemas.SILVER_SCHEMA,
            preserve_index=False,
        )
        if writer is None:
            writer = pq.ParquetWriter(
                target_path,
                schemas.SILVER_SCHEMA,
                compression="snappy",
            )
        writer.write_table(table)
        buffer = []

    try:
        for row in rows:
            buffer.append(row)
            stats.output_rows += 1
            if len(buffer) >= batch_size:
                flush_buffer()

        flush_buffer()
    finally:
        if writer is not None:
            writer.close()

    if stats.output_rows == 0:
        table = pa.Table.from_batches([], schema=schemas.SILVER_SCHEMA)
        pq.write_table(table, target_path, compression="snappy")

    logger.info(f"✓ Wrote silver artifact: {target_path}")


def finalize_silver_parts(
    parts_dir: str,
    output_path: str,
    stats: SilverPipelineStats,
) -> None:
    """
    Apply global silver operations to cleaned parts and write final output.

    This is an external sort + k-way merge. It preserves global ordering and
    deduplication semantics without loading all cleaned rows into pandas.
    """
    parts_path = Path(parts_dir)
    if not any(parts_path.glob("*.parquet")):
        write_silver_rows(iter([]), output_path, stats)
        return

    with TemporaryDirectory(prefix="silver-runs-") as runs_dir:
        run_paths = write_sorted_silver_runs(str(parts_path), runs_dir)
        rows = iter_merged_unique_rows(run_paths, stats)
        write_silver_rows(rows, output_path, stats)


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
    target_path = prepare_silver_output_target(output_path)

    # Reorder columns to match schema order
    silver_fields = get_silver_output_fields()
    df = df[silver_fields]

    # Convert to PyArrow table with schema
    table = pa.Table.from_pandas(df, schema=schemas.SILVER_SCHEMA)

    pq.write_table(table, target_path, compression="snappy")
    logger.info(f"✓ Wrote silver artifact: {target_path}")


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
