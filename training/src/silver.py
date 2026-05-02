"""
Silver layer pipeline: Bronze Parquet → Silver Parquet.

Cleans and prepares bronze data for modeling:
- Removes records with missing required fields
- Keeps null prices but removes records with price <= 0
- Removes canonical duplicate events (global dedup via external merge)
- Sorts deterministically by user_session + source_event_time + tie-break columns
"""

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
import heapq
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from shared import constants, schemas
from training.src.config import Config

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

_SILVER_FIELD_ORDER = [field.name for field in schemas.SILVER_SCHEMA]


@dataclass
class SilverPipelineStats:
    """Counters emitted by the silver streaming pipeline."""

    input_rows: int = 0
    rejected_rows: int = 0
    duplicate_rows: int = 0
    output_rows: int = 0
    next_input_order: int = 0


def silver_polars_to_arrow_table(chunk: pl.DataFrame) -> pa.Table:
    """Cast a silver Polars frame to SILVER_SCHEMA for Parquet writes."""
    t = chunk.select(_SILVER_FIELD_ORDER).to_arrow()
    arrays: list[pa.Array] = []
    for field in schemas.SILVER_SCHEMA:
        col = t.column(field.name).combine_chunks()
        if col.type != field.type:
            col = col.cast(field.type, safe=False)
        arrays.append(col)
    return pa.Table.from_arrays(arrays, schema=schemas.SILVER_SCHEMA)


def get_silver_sort_columns(df: pl.DataFrame) -> list[str]:
    """
    Return the deterministic silver sort columns in canonical order.

    Args:
        df: Input frame (signature kept so callers validate columns first).

    Returns:
        Ordered column names used for deterministic sorting.
    """
    _ = df.columns
    return list(constants.DEDUP_KEY_FIELDS)

def get_silver_output_fields() -> list[str]:
    """Return silver output columns ordered exactly as SILVER_SCHEMA."""
    return [field.name for field in schemas.SILVER_SCHEMA]


def validate_silver_sort_columns(df: pl.DataFrame) -> None:
    """
    Fail fast if the frame cannot be sorted deterministically.

    Raises:
        ValueError: If one or more required sort columns are missing.
    """
    for column in get_silver_sort_columns(df):
        if column not in df.columns:
            raise ValueError(f"missing: {column}")


def normalize_category_code(
    df: pl.DataFrame,
    policy: str = "keep",
    fill_value: str = "unknown",
) -> pl.DataFrame:
    """
    Normalize nullable category_code values according to the selected policy.

    policy: "keep" preserves nulls, "fill" replaces nulls with fill_value

    Raises:
        ValueError: If policy is not supported.
    """
    if policy == "keep":
        return df
    if policy == "fill":
        if "category_code" in df.columns:
            return df.with_columns(pl.col("category_code").fill_null(fill_value))
        return df

    raise ValueError(f"Invalid category_code policy: {policy}")


def read_bronze_parquet(bronze_path: str) -> pl.DataFrame:
    """
    Read bronze parquet artifact into a Polars frame.

    Args:
        bronze_path: Path to bronze parquet file or parquet dataset directory

    Returns:
        Polars frame with bronze data
    """
    bronze_path_p = Path(bronze_path)

    if not bronze_path_p.exists():
        raise FileNotFoundError(f"Bronze parquet not found: {bronze_path}")

    logger.info(f"Reading bronze artifact: {bronze_path_p}")
    table = pq.read_table(bronze_path_p)
    df = pl.from_arrow(table)
    logger.info(f"  ✓ Read {df.height} rows")

    return df


def iter_bronze_batches(
    bronze_path: str,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """
    Yield record batches from a bronze parquet file or dataset directory.

    Args:
        bronze_path: Path to bronze parquet file or parquet dataset directory
        batch_size: Maximum rows per yielded batch

    Yields:
        PyArrow record batches
    """
    bronze_path_p = Path(bronze_path)

    if not bronze_path_p.exists():
        raise FileNotFoundError(f"Bronze parquet not found: {bronze_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    logger.info(f"Reading bronze artifact in batches: {bronze_path_p}")
    if bronze_path_p.is_file():
        parquet_file = pq.ParquetFile(bronze_path_p)
        yield from parquet_file.iter_batches(batch_size=batch_size)
        return

    dataset = ds.dataset(bronze_path_p, format="parquet")
    yield from dataset.to_batches(batch_size=batch_size)


def enforce_silver_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize dtypes compatible with SILVER_SCHEMA."""
    exprs: list[pl.Expr] = []

    if constants.FIELD_SOURCE_EVENT_TIME in df.columns:
        exprs.append(
            pl.col(constants.FIELD_SOURCE_EVENT_TIME).cast(pl.Datetime("us"), strict=False),
        )

    for column in SILVER_STRING_COLUMNS:
        if column in df.columns:
            exprs.append(pl.col(column).cast(pl.Utf8, strict=False))

    if "price" in df.columns:
        exprs.append(pl.col("price").cast(pl.Float64, strict=False))

    return df.with_columns(*exprs) if exprs else df


def _required_silver_columns() -> set[str]:
    required = constants.REQUIRED_FIELDS.copy()
    required.discard(constants.FIELD_EVENT_TIME)
    required.add(constants.FIELD_SOURCE_EVENT_TIME)
    return required


def check_required_fields(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """
    Remove records with missing required fields.

    Returns:
        (valid_df, num_rejected): Filtered frame and count of rejected records
    """
    logger.info("Checking required fields...")
    cols = sorted(_required_silver_columns())
    null_exprs = [pl.col(c).is_null() for c in cols]
    combined = ~pl.any_horizontal(*null_exprs) if null_exprs else pl.lit(True)
    valid = df.filter(combined)
    rejected = df.height - valid.height
    if rejected > 0:
        logger.warning(
            f"  Rejected {rejected} records with missing required fields",
        )
    return valid, rejected


def check_price_validity(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """
    Remove records with invalid non-null prices (price <= 0).

    Returns:
        (valid_df, num_rejected): Filtered frame and count of rejected records
    """
    logger.info("Checking price validity...")
    mask = pl.col("price").is_null() | (
        pl.col("price") > constants.DEFAULT_PRICE_THRESHOLD
    )
    valid = df.filter(mask)
    rejected = df.height - valid.height
    if rejected > 0:
        logger.warning(f"  Rejected {rejected} records with invalid price")
    return valid, rejected


def deduplicate_events(df: pl.DataFrame) -> tuple:
    """Remove duplicates by canonical dedup keys; preserves first occurrence order."""
    logger.info("Deduplicating events by canonical key...")
    before_count = df.height
    deduplicated = df.unique(
        subset=list(constants.DEDUP_KEY_FIELDS),
        maintain_order=True,
        keep="first",
    )
    num_duplicates = before_count - deduplicated.height

    if num_duplicates > 0:
        logger.warning(f"  Removed {num_duplicates} duplicate record(s)")

    return deduplicated, num_duplicates


def clean_silver_batch(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Apply dtype normalization + row validations for one bronze-derived batch."""
    df = enforce_silver_dtypes(df)
    df = normalize_category_code(df, policy="keep")

    total_rejected = 0
    df, rejected = check_required_fields(df)
    total_rejected += rejected

    df, rejected = check_price_validity(df)
    total_rejected += rejected

    return df, total_rejected


def sort_deterministic(df: pl.DataFrame) -> pl.DataFrame:
    """Sort deterministically using canonical tie-break ordering."""
    validate_silver_sort_columns(df)
    sort_columns = get_silver_sort_columns(df)
    logger.info(f"Sorting deterministically by {sort_columns}...")
    out = df.sort(sort_columns, maintain_order=True)
    logger.info(f"  ✓ Sorted {out.height} records")
    return out


def empty_silver_dataframe() -> pl.DataFrame:
    """Return an empty frame with SILVER_SCHEMA columns dtypes."""
    table = pa.Table.from_batches([], schema=schemas.SILVER_SCHEMA)
    return pl.from_arrow(table)


def get_silver_working_fields() -> list[str]:
    """Return temporary working columns used during external sort/dedup."""
    return get_silver_output_fields() + [SILVER_INPUT_ORDER_COLUMN]


def add_input_order(df: pl.DataFrame, stats: SilverPipelineStats) -> pl.DataFrame:
    """Attach monotonic input order for deterministic global duplicate resolution."""
    start_order = stats.next_input_order
    end_order = start_order + df.height
    orders = range(start_order, end_order)
    stats.next_input_order = end_order
    return df.with_columns(
        pl.Series(SILVER_INPUT_ORDER_COLUMN, list(orders), dtype=pl.Int64),
    )


def write_cleaned_silver_batch_parts(
    bronze_path: str,
    parts_dir: str,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> SilverPipelineStats:
    """
    Stream bronze batches through validations into parquet parts prior to finalize.
    """
    parts_path = Path(parts_dir)
    parts_path.mkdir(parents=True, exist_ok=True)

    stats = SilverPipelineStats()
    part_index = 0

    for batch in iter_bronze_batches(bronze_path, batch_size=batch_size):
        stats.input_rows += batch.num_rows
        df = pl.from_arrow(batch)
        df, rejected = clean_silver_batch(df)
        stats.rejected_rows += rejected

        if df.height == 0:
            continue

        df = add_input_order(df, stats)
        df = df.select(get_silver_working_fields())
        table = df.to_arrow()
        part_path = parts_path / f"part-{part_index:05d}.parquet"
        pq.write_table(table, part_path, compression="snappy")
        logger.info(f"  Wrote cleaned silver batch part: {part_path}")
        part_index += 1

    return stats


def get_silver_merge_columns() -> list[str]:
    """Columns used for sorted runs + merge ordering."""
    return list(constants.DEDUP_KEY_FIELDS) + [SILVER_INPUT_ORDER_COLUMN]


def sort_silver_working_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by canonical dedup columns then input-order (stable)."""
    return df.sort(
        get_silver_merge_columns(),
        maintain_order=True,
        multithreaded=False,
    )


def write_sorted_silver_runs(
    parts_dir: str,
    runs_dir: str,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> list[Path]:
    """
    Sort cleaned working parts into bounded parquet runs suitable for heap merge.

    Peak memory bounded by configured batch sizes.
    """
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    run_paths: list[Path] = []
    source = ds.dataset(parts_dir, format="parquet")
    for run_index, batch in enumerate(source.to_batches(batch_size=batch_size)):
        df = pl.from_arrow(batch)
        if df.height == 0:
            continue

        df = enforce_silver_dtypes(df)
        df = df.with_columns(pl.col(SILVER_INPUT_ORDER_COLUMN).cast(pl.Int64, strict=True))
        df = sort_silver_working_frame(df.select(get_silver_working_fields()))

        run_path = runs_path / f"run-{run_index:05d}.parquet"
        pq.write_table(df.to_arrow(), run_path, compression="snappy")
        run_paths.append(run_path)
        logger.info(f"  Wrote sorted silver run: {run_path}")

    return run_paths


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict]:
    """Yield rows from a parquet file without loading the entire file."""
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = pl.from_arrow(batch)
        yield from df.iter_rows(named=True)


def row_merge_key(row: dict) -> tuple:
    """Heap key preserving global merge order."""
    return tuple(row[column] for column in get_silver_merge_columns())


def row_dedup_key(row: dict) -> tuple:
    """Canonical dedupe key."""

    return tuple(row[column] for column in constants.DEDUP_KEY_FIELDS)


def iter_merged_unique_rows(
    run_paths: list[Path],
    stats: SilverPipelineStats,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> Iterator[dict]:
    """
    Merge sorted runs and emit globally deduplicated silver rows preserving:
    deterministic sort ordering and first-survivor duplicates by bronze stream order.
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
    output_path_p = Path(output_path)

    if output_path_p.suffix == ".parquet":
        output_path_p.parent.mkdir(parents=True, exist_ok=True)
        return output_path_p

    if output_path_p.exists() and any(output_path_p.iterdir()):
        raise FileExistsError(f"Silver dataset directory is not empty: {output_path}")

    output_path_p.mkdir(parents=True, exist_ok=True)
    return output_path_p / "part-000.parquet"


def write_silver_rows(
    rows: Iterator[dict],
    output_path: str,
    stats: SilverPipelineStats,
    batch_size: int = DEFAULT_SILVER_BATCH_SIZE,
) -> None:
    """Append silver rows via ParquetWriter using bounded Polars batches."""
    target_path = prepare_silver_output_target(output_path)
    output_fields = get_silver_output_fields()
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []

    def flush_buffer() -> None:
        nonlocal writer, buffer
        if not buffer:
            return

        df_batch = enforce_silver_dtypes(pl.DataFrame(buffer))
        table = silver_polars_to_arrow_table(df_batch.select(output_fields))

        if writer is None:
            writer = pq.ParquetWriter(
                target_path,
                schemas.SILVER_SCHEMA,
                compression="snappy",
            )
        writer.write_table(table)
        buffer.clear()

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

    External sort + k-way merge; bounded memory versus full materialization.
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
    """Streaming silver pipeline with finalize external merge."""
    with TemporaryDirectory(prefix="silver-cleaned-") as temp_dir:
        parts_dir = Path(temp_dir) / "parts"
        stats = write_cleaned_silver_batch_parts(
            bronze_path,
            str(parts_dir),
            batch_size=batch_size,
        )
        finalize_silver_parts(str(parts_dir), output_path, stats)
        return stats


def write_silver_parquet(df: pl.DataFrame, output_path: str) -> None:
    """Write silver Polars frame as Parquet single file or dataset directory."""
    target_path = prepare_silver_output_target(output_path)
    silver_fields = get_silver_output_fields()
    table = silver_polars_to_arrow_table(df.select(silver_fields))
    pq.write_table(table, target_path, compression="snappy")
    logger.info(f"✓ Wrote silver artifact: {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean bronze data and produce silver artifact",
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
