"""
Bronze layer pipeline: Raw CSV → Bronze Parquet (chunked streaming).

Transforms raw event data from CSV to standardized Parquet format.
- Reads files in batches via Polars scan_csv → collect_batches (bounded memory)
- Renames event_time → source_event_time
- Validates event_type
- Rejects invalid records
- Outputs immutable parquet artifact via ParquetWriter append

Memory-efficient: peak RAM ~O(chunksize) instead of O(total_data).
"""

import argparse
import gc
import logging
import re
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from shared import constants, schemas
from training.src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_RAW_CSV_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "event_time": pl.Utf8,
    "event_type": pl.Utf8,
    "product_id": pl.Utf8,
    constants.FIELD_CATEGORY_ID: pl.Utf8,
    "category_code": pl.Utf8,
    "brand": pl.Utf8,
    "user_id": pl.Utf8,
    "user_session": pl.Utf8,
    "price": pl.Float64,
}

BRONZE_STRING_COLUMNS = [
    "event_type",
    "product_id",
    constants.FIELD_CATEGORY_ID,
    "user_id",
    "user_session",
    "category_code",
    "brand",
]

RAW_FILE_MONTH_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>[A-Za-z]{3})\.csv(?:\.gz)?$"
)
MONTH_ABBREVIATIONS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
VALID_WINDOW_PROFILES = {"training", "dev_smoke", "replay", "all"}

_BRONZE_FIELD_ORDER = [field.name for field in schemas.BRONZE_SCHEMA]


def finalize_bronze_polars_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Cast columns to dtypes compatible with BRONZE_SCHEMA ordering."""
    str_casts = [
        pl.col(column).cast(pl.Utf8, strict=False)
        for column in BRONZE_STRING_COLUMNS
        if column in df.columns
    ]
    return df.with_columns(
        *str_casts,
        pl.col("price").cast(pl.Float64, strict=False),
        pl.col(constants.FIELD_SOURCE_EVENT_TIME).cast(pl.Datetime("us")),
    ).select(_BRONZE_FIELD_ORDER)


def get_current_memory_mb() -> float:
    """Get current process memory in MB. Returns 0 if psutil unavailable."""
    if not HAS_PSUTIL:
        return 0.0
    try:
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_polars_estimated_mb(df: pl.DataFrame) -> float:
    """Best-effort DataFrame footprint in MB (Polars estimator)."""
    if df.height == 0:
        return 0.0
    try:
        return float(df.estimated_size(unit="mb"))
    except Exception:
        return 0.0


def bronze_polars_to_arrow_table(chunk: pl.DataFrame) -> pa.Table:
    """Cast a bronze Polars frame to BRONZE_SCHEMA for Parquet writes."""
    t = chunk.select(_BRONZE_FIELD_ORDER).to_arrow()
    arrays: list[pa.Array] = []
    for field in schemas.BRONZE_SCHEMA:
        col = t.column(field.name).combine_chunks()
        if col.type != field.type:
            col = col.cast(field.type, safe=False)
        arrays.append(col)
    return pa.Table.from_arrays(arrays, schema=schemas.BRONZE_SCHEMA)


def parse_window_month(value: str) -> tuple[int, int]:
    """Parse a YYYY-MM window bound into a comparable (year, month) tuple."""
    try:
        year_text, month_text = value.split("-", maxsplit=1)
        year = int(year_text)
        month = int(month_text)
    except ValueError as exc:
        raise ValueError(f"Invalid window month '{value}', expected YYYY-MM") from exc

    if month < 1 or month > 12:
        raise ValueError(f"Invalid window month '{value}', expected month 01-12")

    return year, month


def extract_raw_file_month(file_path: Path) -> Optional[tuple[int, int]]:
    """
    Extract month from supported raw filenames: YYYY-Mon.csv or YYYY-Mon.csv.gz.

    Unsupported filenames return None so callers can skip manifests/notes.
    """
    match = RAW_FILE_MONTH_PATTERN.match(file_path.name)
    if not match:
        return None

    month = MONTH_ABBREVIATIONS.get(match.group("month").lower())
    if month is None:
        return None

    return int(match.group("year")), month


def resolve_raw_window_bounds(
    window_profile: Optional[str] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
    """
    Resolve profile/custom window bounds into inclusive month tuples.

    Custom bounds override the profile only when both start and end are provided.
    The all profile without custom bounds selects every supported raw file.
    """
    if bool(window_start) != bool(window_end):
        raise ValueError("Both window_start and window_end must be provided together")

    if window_start and window_end:
        start_month = parse_window_month(window_start)
        end_month = parse_window_month(window_end)
        if start_month > end_month:
            raise ValueError(
                f"window_start must be <= window_end: {window_start} > {window_end}"
            )
        return start_month, end_month

    profile = (window_profile or Config.DATA_WINDOW_PROFILE).lower()
    if profile not in VALID_WINDOW_PROFILES:
        raise ValueError(
            f"Invalid window_profile '{profile}'. "
            f"Expected one of: {', '.join(sorted(VALID_WINDOW_PROFILES))}"
        )

    if profile == "all":
        return None, None

    if profile == "training":
        start_value = Config.TRAINING_WINDOW_START
        end_value = Config.TRAINING_WINDOW_END
    elif profile == "dev_smoke":
        start_value = Config.DEV_SMOKE_WINDOW_START
        end_value = Config.DEV_SMOKE_WINDOW_END
    else:
        start_value = Config.REPLAY_WINDOW_START
        end_value = Config.REPLAY_WINDOW_END

    start_month = parse_window_month(start_value)
    end_month = parse_window_month(end_value)
    if start_month > end_month:
        raise ValueError(f"Invalid {profile} window: {start_value} > {end_value}")

    return start_month, end_month


def select_raw_files(
    raw_dir: str,
    window_profile: Optional[str] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
) -> list[Path]:
    """
    Select supported raw CSV files for a profile/custom inclusive month window.
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    start_month, end_month = resolve_raw_window_bounds(
        window_profile=window_profile,
        window_start=window_start,
        window_end=window_end,
    )

    candidates = list(raw_path.glob("*.csv")) + list(raw_path.glob("*.csv.gz"))
    selected_with_month: list[tuple[tuple[int, int], Path]] = []

    for file_path in candidates:
        file_month = extract_raw_file_month(file_path)
        if file_month is None:
            logger.info(f"Skipping unsupported raw filename: {file_path.name}")
            continue

        if start_month is not None and file_month < start_month:
            continue
        if end_month is not None and file_month > end_month:
            continue

        selected_with_month.append((file_month, file_path))

    selected = [
        path
        for _, path in sorted(
            selected_with_month, key=lambda item: (item[0], item[1].name)
        )
    ]

    if not selected:
        profile = window_profile or Config.DATA_WINDOW_PROFILE
        raise FileNotFoundError(
            f"No supported raw CSV files found in {raw_dir} for profile/window "
            f"{profile}"
        )

    return selected


def discover_raw_files(
    raw_dir: str,
    window_profile: Optional[str] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
) -> list[Path]:
    """
    Discover supported raw CSV files in raw directory for the selected window.

    Args:
        raw_dir: Path to raw data directory

    Returns:
        Sorted list of Path objects for .csv and .csv.gz files
    """
    all_files = select_raw_files(
        raw_dir,
        window_profile=window_profile,
        window_start=window_start,
        window_end=window_end,
    )

    logger.info(f"Discovered {len(all_files)} raw file(s) to process")
    for f in all_files:
        logger.info(f"  - {f.name}")

    return all_files


def ensure_not_simulation_raw_input(raw_dir: str) -> None:
    """
    Reject simulation raw input for the bronze baseline pipeline.

    Baseline bronze should only ingest train_raw. Simulation raw is reserved for
    Online Simulation and retraining export flows.
    """
    input_path = Path(raw_dir).resolve()
    simulation_path = Path(Config.SIMULATION_RAW_DATA_PATH).resolve()
    simulation_dir = simulation_path.parent

    if (
        input_path == simulation_path
        or input_path == simulation_dir
        or simulation_dir in input_path.parents
    ):
        raise ValueError(
            f"Bronze baseline input must not read from simulation raw data: {raw_dir}"
        )


def read_raw_chunks(
    raw_dir: str,
    chunksize: int = 200000,
    window_profile: Optional[str] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
) -> Generator[Tuple[Path, pl.DataFrame], None, None]:
    """
    Generator: yield chunks from each raw CSV file.

    Minimizes memory by processing one chunk at a time across all files.

    Args:
        raw_dir: Path to raw data directory
        chunksize: Number of rows per chunk

    Yields:
        (file_path, chunk_df) tuples
    """
    ensure_not_simulation_raw_input(raw_dir)
    all_files = discover_raw_files(
        raw_dir,
        window_profile=window_profile,
        window_start=window_start,
        window_end=window_end,
    )

    for file_path in all_files:
        logger.info(f"\nProcessing {file_path.name}...")

        try:
            lf = pl.scan_csv(
                str(file_path),
                schema_overrides=_RAW_CSV_SCHEMA_OVERRIDES,
            )
            chunk_idx = 0
            for chunk in lf.collect_batches(chunk_size=chunksize):
                chunk_idx += 1
                logger.debug(
                    f"  Chunk {chunk_idx}: {chunk.height} rows | "
                    f"RAM: {get_current_memory_mb():.1f} MB | "
                    f"DF: {get_polars_estimated_mb(chunk):.1f} MB"
                )
                yield file_path, chunk

        except Exception as e:
            logger.error(f"  ✗ Error reading {file_path.name}: {e}")
            raise


def parse_event_time_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse event_time string to timestamp on chunk.

    Input format: "2019-10-01 00:00:00 UTC"
    """
    return df.with_columns(
        pl.col(constants.FIELD_EVENT_TIME)
        .str.replace(" UTC", "")
        .str.strptime(
            pl.Datetime("us"),
            format="%Y-%m-%d %H:%M:%S",
            strict=False,
        )
        .alias(constants.FIELD_EVENT_TIME),
    )


def parse_event_time(df: pl.DataFrame) -> pl.DataFrame:
    """
    Backward-compatible wrapper for parse_event_time_chunk.
    Used in tests and standalone script calls.
    """
    return parse_event_time_chunk(df)


def validate_event_type(df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
    """
    Filter records with valid event_type.

    Args:
        df: Input Polars frame

    Returns:
        (valid_df, num_rejected): Filtered frame and count of rejected records
    """
    n_in = df.height
    allowed = sorted(constants.ALLOWED_EVENT_TYPES)
    valid_df = df.filter(pl.col("event_type").is_in(allowed))
    num_rejected = n_in - valid_df.height

    if num_rejected > 0:
        logger.warning(f"Rejecting {num_rejected} records with invalid event_type")

    return valid_df, num_rejected


def transform_to_bronze_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform raw Polars chunk to bronze schema order and dtypes.

    - Rename event_time → source_event_time
    - Cast string columns / price coherently with BRONZE_SCHEMA
    """
    renamed = df.rename(
        {constants.FIELD_EVENT_TIME: constants.FIELD_SOURCE_EVENT_TIME},
    )
    return finalize_bronze_polars_dtypes(renamed)


def transform_to_bronze(df: pl.DataFrame) -> pl.DataFrame:
    """
    Backward-compatible wrapper for transform_to_bronze_chunk.
    Used in tests and standalone script calls.
    """
    return transform_to_bronze_chunk(df)


def write_bronze_parquet_chunked(
    raw_dir: str,
    output_path: str,
    chunksize: int = 200000,
    memory_log: bool = True,
    window_profile: Optional[str] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
) -> dict:
    """
    Write bronze artifact from raw CSVs using chunked streaming + ParquetWriter.
    """
    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows_in = 0
    total_rows_valid = 0
    total_rows_rejected = 0
    chunk_count = 0
    start_time = time.time()
    start_memory_mb = get_current_memory_mb()

    try:
        for file_path, chunk in read_raw_chunks(
            raw_dir,
            chunksize=chunksize,
            window_profile=window_profile,
            window_start=window_start,
            window_end=window_end,
        ):
            chunk_count += 1
            rows_in_chunk = chunk.height
            total_rows_in += rows_in_chunk

            chunk_parsed = parse_event_time_chunk(chunk)
            chunk_valid, chunk_rejected = validate_event_type(chunk_parsed)
            total_rows_valid += chunk_valid.height
            total_rows_rejected += chunk_rejected

            if chunk_valid.height == 0:
                logger.info(f"  Chunk {chunk_count}: 0 valid rows (all rejected)")
                del chunk, chunk_parsed, chunk_valid
                if memory_log:
                    gc.collect()
                continue

            chunk_bronze = transform_to_bronze_chunk(chunk_valid)

            try:
                table = bronze_polars_to_arrow_table(chunk_bronze)
            except Exception as e:
                logger.error(f"  Failed to convert chunk {chunk_count} to Arrow: {e}")
                raise

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path_p,
                    schema=schemas.BRONZE_SCHEMA,
                    compression="snappy",
                )
                logger.info(f"Initialized ParquetWriter: {output_path_p}")

            writer.write_table(table)

            if memory_log:
                elapsed = time.time() - start_time
                current_memory_mb = get_current_memory_mb()
                throughput = total_rows_in / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Chunk {chunk_count}: {rows_in_chunk} rows | "
                    f"Valid: {chunk_valid.height} | Rejected: {chunk_rejected} | "
                    f"RAM: {current_memory_mb:.1f}MB (Δ{current_memory_mb - start_memory_mb:+.1f}MB) | "
                    f"Throughput: {throughput:.0f} rows/s"
                )

            del chunk, chunk_parsed, chunk_valid, chunk_bronze, table
            if memory_log:
                gc.collect()

    finally:
        if writer is not None:
            writer.close()
            logger.info(f"✓ Closed ParquetWriter: {output_path_p}")

    elapsed_total = time.time() - start_time
    final_memory_mb = get_current_memory_mb()
    avg_throughput = total_rows_in / elapsed_total if elapsed_total > 0 else 0

    return {
        "total_rows_in": total_rows_in,
        "total_rows_valid": total_rows_valid,
        "total_rows_rejected": total_rows_rejected,
        "chunk_count": chunk_count,
        "elapsed_seconds": elapsed_total,
        "throughput_rows_per_sec": avg_throughput,
        "start_memory_mb": start_memory_mb,
        "final_memory_mb": final_memory_mb,
        "peak_memory_delta_mb": final_memory_mb - start_memory_mb,
    }


def write_bronze_parquet(df: pl.DataFrame, output_path: str) -> None:
    """Write single Polars bronze frame to Parquet."""
    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)
    table = bronze_polars_to_arrow_table(finalize_bronze_polars_dtypes(df))
    pq.write_table(table, output_path_p, compression="snappy")
    logger.info(f"✓ Wrote bronze artifact: {output_path_p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform raw CSV data to bronze parquet format (chunked streaming)"
    )
    parser.add_argument(
        "--input",
        default=Config.TRAIN_RAW_DATA_PATH,
        help=(
            "Path to baseline training raw data directory "
            f"(default: {Config.TRAIN_RAW_DATA_PATH})"
        ),
    )
    parser.add_argument(
        "--output",
        default=Config.BRONZE_DATA_PATH,
        help=f"Path to bronze output parquet file (default: {Config.BRONZE_DATA_PATH})",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Rows per chunk (default: 200000). Lower for tight RAM, higher for speed.",
    )
    parser.add_argument(
        "--window-profile",
        choices=sorted(VALID_WINDOW_PROFILES),
        default=Config.DATA_WINDOW_PROFILE,
        help=(
            "Raw data usage window to materialize "
            f"(default: {Config.DATA_WINDOW_PROFILE})"
        ),
    )
    parser.add_argument(
        "--window-start",
        help="Optional custom inclusive window start month in YYYY-MM format",
    )
    parser.add_argument(
        "--window-end",
        help="Optional custom inclusive window end month in YYYY-MM format",
    )
    parser.add_argument(
        "--memory-log",
        action="store_true",
        default=True,
        help="Enable memory telemetry logging (default: enabled)",
    )
    parser.add_argument(
        "--no-memory-log",
        action="store_false",
        dest="memory_log",
        help="Disable memory telemetry logging",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("BRONZE PIPELINE: Raw CSV → Bronze Parquet (Chunked Streaming)")
    logger.info("=" * 70)
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output file:     {args.output}")
    logger.info(f"Window profile:  {args.window_profile}")
    if args.window_start or args.window_end:
        logger.info(f"Custom window:   {args.window_start} -> {args.window_end}")
    logger.info(f"Chunksize:       {args.chunksize:,} rows")
    logger.info(f"Memory telemetry: {'enabled' if args.memory_log else 'disabled'}")
    if not HAS_PSUTIL and args.memory_log:
        logger.warning(
            "  (psutil not available; memory telemetry limited to DataFrame estimates)"
        )
    logger.info("=" * 70)

    try:
        logger.info("\n1. Reading and transforming raw data in chunks...")
        stats = write_bronze_parquet_chunked(
            args.input,
            args.output,
            chunksize=args.chunksize,
            memory_log=args.memory_log,
            window_profile=args.window_profile,
            window_start=args.window_start,
            window_end=args.window_end,
        )

        logger.info("\n" + "=" * 70)
        logger.info("BRONZE PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Input rows:      {stats['total_rows_in']:,}")
        logger.info(f"Valid rows:      {stats['total_rows_valid']:,}")
        logger.info(f"Rejected rows:   {stats['total_rows_rejected']:,}")
        logger.info(
            f"Reject rate:     {stats['total_rows_rejected'] / max(stats['total_rows_in'], 1) * 100:.2f}%"
        )
        logger.info(f"Chunks processed: {stats['chunk_count']}")
        logger.info(f"Total time:      {stats['elapsed_seconds']:.2f}s")
        logger.info(f"Throughput:      {stats['throughput_rows_per_sec']:.0f} rows/s")
        if args.memory_log:
            logger.info(f"Initial RAM:     {stats['start_memory_mb']:.1f} MB")
            logger.info(f"Final RAM:       {stats['final_memory_mb']:.1f} MB")
            logger.info(f"RAM delta:       {stats['peak_memory_delta_mb']:+.1f} MB")
        logger.info(f"Output file:     {args.output}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
