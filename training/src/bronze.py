"""
Bronze layer pipeline: Raw CSV → Bronze Parquet (chunked streaming).

Transforms raw event data from CSV to standardized Parquet format.
- Reads files in chunks to minimize memory footprint
- Renames event_time → source_event_time
- Validates event_type
- Rejects invalid records
- Outputs immutable parquet artifact via ParquetWriter append

Memory-efficient: peak RAM ~O(chunksize) instead of O(total_data).
"""

import argparse
import logging
from pathlib import Path
from typing import Generator, Tuple
import time
import gc

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

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


RAW_CSV_DTYPES = {
    "event_type": "string",
    "product_id": "string",
    "user_id": "string",
    "user_session": "string",
    "category_code": "string",
    "brand": "string",
}

BRONZE_STRING_COLUMNS = [
    "event_type",
    "product_id",
    "user_id",
    "user_session",
    "category_code",
    "brand",
]


def get_current_memory_mb() -> float:
    """Get current process memory in MB. Returns 0 if psutil unavailable."""
    if not HAS_PSUTIL:
        return 0.0
    try:
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_dataframe_memory_mb(df: pd.DataFrame) -> float:
    """Get DataFrame memory in MB (deep count)."""
    if df.empty:
        return 0.0
    try:
        return df.memory_usage(deep=True).sum() / (1024 * 1024)
    except Exception:
        return 0.0


def discover_raw_files(raw_dir: str) -> list:
    """
    Discover all CSV files in raw directory.

    Args:
        raw_dir: Path to raw data directory

    Returns:
        Sorted list of Path objects for .csv and .csv.gz files
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    csv_files = list(raw_path.glob("*.csv"))
    gz_files = list(raw_path.glob("*.csv.gz"))

    all_files = sorted(csv_files + gz_files)

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    logger.info(f"Discovered {len(all_files)} raw file(s) to process")
    for f in all_files:
        logger.info(f"  - {f.name}")

    return all_files


def read_raw_chunks(
    raw_dir: str, chunksize: int = 200000
) -> Generator[Tuple[Path, pd.DataFrame], None, None]:
    """
    Generator: yield chunks from each raw CSV file.

    Minimizes memory by processing one chunk at a time across all files.

    Args:
        raw_dir: Path to raw data directory
        chunksize: Number of rows per chunk

    Yields:
        (file_path, chunk_df) tuples
    """
    all_files = discover_raw_files(raw_dir)

    for file_path in all_files:
        logger.info(f"\nProcessing {file_path.name}...")

        try:
            if str(file_path).endswith(".gz"):
                reader = pd.read_csv(
                    file_path,
                    compression="gzip",
                    dtype=RAW_CSV_DTYPES,
                    chunksize=chunksize,
                )
            else:
                reader = pd.read_csv(
                    file_path,
                    dtype=RAW_CSV_DTYPES,
                    chunksize=chunksize,
                )

            for chunk_idx, chunk in enumerate(reader, start=1):
                logger.debug(
                    f"  Chunk {chunk_idx}: {len(chunk)} rows | "
                    f"RAM: {get_current_memory_mb():.1f} MB | "
                    f"DF: {get_dataframe_memory_mb(chunk):.1f} MB"
                )
                yield file_path, chunk

        except Exception as e:
            logger.error(f"  ✗ Error reading {file_path.name}: {e}")
            raise


def parse_event_time_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse event_time string to timestamp in-place on chunk.

    Input format: "2019-10-01 00:00:00 UTC"

    Args:
        df: Input DataFrame with event_time column (modified in-place)

    Returns:
        DataFrame with parsed event_time as timestamp
    """
    # Remove ' UTC' suffix if present, then parse
    df[constants.FIELD_EVENT_TIME] = pd.to_datetime(
        df[constants.FIELD_EVENT_TIME].str.replace(" UTC", "")
    )
    return df


def parse_event_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper for parse_event_time_chunk.
    Used in tests and standalone script calls.
    """
    return parse_event_time_chunk(df)


def validate_event_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Filter records with valid event_type.

    Returns view (not copy) to save memory; caller must copy if needed.

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

    return df[mask], num_rejected


def transform_to_bronze_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw DataFrame chunk to bronze schema.

    - Rename event_time → source_event_time
    - Select only schema fields in correct order
    - Minimal dtype casting (only when needed)

    Args:
        df: Input raw DataFrame (with parsed event_time)

    Returns:
        Bronze-formatted DataFrame ready for Arrow conversion
    """
    # Rename event_time to source_event_time
    df = df.rename(
        columns={constants.FIELD_EVENT_TIME: constants.FIELD_SOURCE_EVENT_TIME}
    )

    # Cast only if needed: check current dtype before conversion
    for column in BRONZE_STRING_COLUMNS:
        if column in df.columns and df[column].dtype != "string":
            df[column] = df[column].astype("string")

    # Cast price only if it exists and is not already float64
    if "price" in df.columns and df["price"].dtype != "float64":
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Select only schema fields (in order), avoiding unnecessary copy
    bronze_fields = [field.name for field in schemas.BRONZE_SCHEMA]
    df = df[bronze_fields]

    return df


def transform_to_bronze(df: pd.DataFrame) -> pd.DataFrame:
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
) -> dict:
    """
    Write bronze artifact from raw CSVs using chunked streaming + ParquetWriter.

    This is the main chunked pipeline: reads chunks → validates → transforms → appends to parquet.

    Args:
        raw_dir: Path to raw data directory
        output_path: Path to output parquet file
        chunksize: Rows per chunk (default 200000)
        memory_log: Enable memory telemetry logging

    Returns:
        dict with pipeline stats: total_rows_in, total_rows_valid, total_rows_rejected, etc.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows_in = 0
    total_rows_valid = 0
    total_rows_rejected = 0
    chunk_count = 0
    start_time = time.time()
    start_memory_mb = get_current_memory_mb()

    try:
        for file_path, chunk in read_raw_chunks(raw_dir, chunksize=chunksize):
            chunk_count += 1
            rows_in_chunk = len(chunk)
            total_rows_in += rows_in_chunk

            # Parse event_time
            chunk = parse_event_time_chunk(chunk)

            # Validate event_type
            chunk_valid, chunk_rejected = validate_event_type(chunk)
            total_rows_valid += len(chunk_valid)
            total_rows_rejected += chunk_rejected

            if len(chunk_valid) == 0:
                logger.info(f"  Chunk {chunk_count}: 0 valid rows (all rejected)")
                del chunk, chunk_valid
                if memory_log:
                    gc.collect()
                continue

            # Transform to bronze schema
            chunk_bronze = transform_to_bronze_chunk(chunk_valid)

            # Convert to Arrow table
            try:
                table = pa.Table.from_pandas(
                    chunk_bronze,
                    schema=schemas.BRONZE_SCHEMA,
                    preserve_index=False,
                )
            except Exception as e:
                logger.error(f"  Failed to convert chunk {chunk_count} to Arrow: {e}")
                raise

            # Lazy-init writer on first valid chunk
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path, schema=schemas.BRONZE_SCHEMA, compression="snappy"
                )
                logger.info(f"Initialized ParquetWriter: {output_path}")

            # Append table to parquet
            writer.write_table(table)

            if memory_log:
                elapsed = time.time() - start_time
                current_memory_mb = get_current_memory_mb()
                throughput = total_rows_in / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Chunk {chunk_count}: {rows_in_chunk} rows | "
                    f"Valid: {len(chunk_valid)} | Rejected: {chunk_rejected} | "
                    f"RAM: {current_memory_mb:.1f}MB (Δ{current_memory_mb - start_memory_mb:+.1f}MB) | "
                    f"Throughput: {throughput:.0f} rows/s"
                )

            # Clean up chunk memory
            del chunk, chunk_valid, chunk_bronze, table
            if memory_log:
                gc.collect()

    finally:
        # Close writer
        if writer is not None:
            writer.close()
            logger.info(f"✓ Closed ParquetWriter: {output_path}")

    elapsed_total = time.time() - start_time
    final_memory_mb = get_current_memory_mb()
    avg_throughput = total_rows_in / elapsed_total if elapsed_total > 0 else 0

    stats = {
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

    return stats


def write_bronze_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Backward-compatible wrapper: write single DataFrame to parquet.
    Used in tests and full-load scenarios (not chunked).
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
        description="Transform raw CSV data to bronze parquet format (chunked streaming)"
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
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Number of rows per chunk (default: 200000). Lower for tight RAM, higher for speed.",
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
    logger.info(f"Chunksize:       {args.chunksize:,} rows")
    logger.info(f"Memory telemetry: {'enabled' if args.memory_log else 'disabled'}")
    if not HAS_PSUTIL and args.memory_log:
        logger.warning(
            "  (psutil not available; memory logging limited to DataFrame stats)"
        )
    logger.info("=" * 70)

    try:
        # Run chunked pipeline
        logger.info(f"\n1. Reading and transforming raw data in chunks...")
        stats = write_bronze_parquet_chunked(
            args.input,
            args.output,
            chunksize=args.chunksize,
            memory_log=args.memory_log,
        )

        # Summary
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
