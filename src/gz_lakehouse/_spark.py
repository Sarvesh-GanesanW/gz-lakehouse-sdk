"""Spark conversion helpers for the gz-lakehouse SDK.

Spark is a popular destination for query results, but the naive
``spark.createDataFrame(arrow_table.to_pandas())`` path that earlier
versions of this SDK shipped is roughly an order of magnitude slower
than necessary. The helpers in this module pick the fastest available
path at runtime:

#. **PySpark 3.4+**: ``spark.createDataFrame`` accepts a
   :class:`pyarrow.Table` directly and serialises it as Arrow record
   batches over the JVM bridge. ~100–200 MB/s, no pandas hop.
#. **PySpark 3.0–3.3**: enable ``spark.sql.execution.arrow.pyspark.enabled``
   for the duration of the conversion and pass a pandas DataFrame.
   ~50–100 MB/s, single pandas materialisation.
#. **PySpark < 3.0** (and the rare case where Arrow is forcibly
   disabled): plain pandas conversion. ~5–10 MB/s. Warned about so
   callers know they are on the slow path.

For very large results (multi-GB) the Python driver becomes the
bottleneck no matter how fast the conversion is — write the Arrow
table to Parquet (locally or on S3) and let the Spark executors load
it in parallel. :func:`write_arrow_to_parquet_for_spark` exists for
that pattern.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from gz_lakehouse._logging import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

_logger = get_logger("spark")

_ARROW_CONFIG = "spark.sql.execution.arrow.pyspark.enabled"


def parse_pyspark_version(version: str) -> tuple[int, int, int]:
    """Parse a PySpark version string into a comparable tuple.

    Args:
        version: A version string like ``"3.4.1"`` or ``"3.5.0.dev0"``.

    Returns:
        ``(major, minor, patch)`` as integers. Non-numeric suffixes
        (``dev0``, ``rc1``) are stripped. Missing components default
        to zero so callers can compare with simple ``>=`` semantics.
    """
    cleaned: list[int] = []
    for part in version.split(".")[:3]:
        prefix_digits = ""
        for ch in part:
            if ch.isdigit():
                prefix_digits += ch
            else:
                break
        cleaned.append(int(prefix_digits) if prefix_digits else 0)
    while len(cleaned) < 3:
        cleaned.append(0)
    return cleaned[0], cleaned[1], cleaned[2]


def _supports_arrow_table_input(spark: SparkSession) -> bool:
    """Return True when ``createDataFrame`` accepts a :class:`pa.Table`."""
    pyspark = _import_pyspark()
    if pyspark is None:
        return False
    return parse_pyspark_version(pyspark.__version__) >= (3, 4, 0)


def _import_pyspark() -> Any:
    """Best-effort import of the optional ``pyspark`` module."""
    try:
        import pyspark  # noqa: PLC0415

        return pyspark
    except ImportError:
        return None


@contextmanager
def _arrow_config(spark: SparkSession) -> Iterator[None]:
    """Temporarily enable Spark's Arrow-based pandas conversion."""
    previous: str | None
    try:
        previous = spark.conf.get(_ARROW_CONFIG)
    except Exception:  # noqa: BLE001
        previous = None
    spark.conf.set(_ARROW_CONFIG, "true")
    try:
        yield
    finally:
        if previous is None:
            try:
                spark.conf.unset(_ARROW_CONFIG)
            except Exception:  # noqa: BLE001
                spark.conf.set(_ARROW_CONFIG, "false")
        else:
            spark.conf.set(_ARROW_CONFIG, previous)


def arrow_to_spark(
    table: pa.Table,
    spark: SparkSession,
) -> SparkDataFrame:
    """Convert a :class:`pyarrow.Table` to a Spark DataFrame, fast.

    The function picks the fastest available conversion path based on
    the running PySpark version. Empty tables retain their schema so
    downstream Spark code keeps the column metadata.

    Args:
        table: Arrow table holding the materialised result.
        spark: Active :class:`pyspark.sql.SparkSession`.

    Returns:
        A Spark DataFrame with the same schema as ``table``.
    """
    if table.num_rows == 0:
        return _empty_spark_dataframe(table, spark)

    if _supports_arrow_table_input(spark):
        _logger.debug("Spark conversion path: Arrow Table direct (>=3.4)")
        try:
            return spark.createDataFrame(table)
        except (TypeError, ValueError) as ex:
            _logger.warning(
                "spark.createDataFrame(pa.Table) failed; "
                "falling back to arrow-enabled pandas: %s",
                ex,
            )

    pandas_df = table.to_pandas()
    try:
        with _arrow_config(spark):
            _logger.debug("Spark conversion path: arrow-enabled pandas")
            return spark.createDataFrame(pandas_df)
    except Exception as ex:  # noqa: BLE001
        warnings.warn(
            f"Falling back to non-Arrow Spark conversion (slow). Reason: {ex}",
            stacklevel=3,
        )
        _logger.warning(
            "Spark conversion path: plain pandas (slow); reason=%s", ex
        )
        return spark.createDataFrame(pandas_df)


def _empty_spark_dataframe(
    table: pa.Table, spark: SparkSession
) -> SparkDataFrame:
    """Build an empty Spark DataFrame that preserves the Arrow schema."""
    schema_string = ", ".join(
        f"{field.name} {_arrow_to_spark_type(field.type)}"
        for field in table.schema
    )
    if not schema_string:
        return spark.createDataFrame([], schema="value string")
    return spark.createDataFrame([], schema=schema_string)


def _is_binary_or_fixed(arrow_type: pa.DataType) -> bool:
    """True for both ``binary`` and ``fixed_size_binary`` Arrow types."""
    return pa.types.is_binary(arrow_type) or pa.types.is_fixed_size_binary(
        arrow_type
    )


_ARROW_TO_SPARK_TYPE_CHECKS: tuple[tuple[str, Any], ...] = (
    ("boolean", pa.types.is_boolean),
    ("tinyint", pa.types.is_int8),
    ("smallint", pa.types.is_int16),
    ("int", pa.types.is_int32),
    ("bigint", pa.types.is_int64),
    ("float", pa.types.is_float32),
    ("double", pa.types.is_float64),
    ("date", pa.types.is_date),
    ("timestamp", pa.types.is_timestamp),
    ("binary", _is_binary_or_fixed),
)


def _arrow_to_spark_type(arrow_type: pa.DataType) -> str:
    """Translate an Arrow type to a Spark DDL type string.

    Used only for empty-table conversions where Spark cannot infer the
    schema from rows. The mapping covers the type families
    :func:`gz_lakehouse._arrow_build.arrow_type_for` produces.
    """
    for ddl_name, predicate in _ARROW_TO_SPARK_TYPE_CHECKS:
        if predicate(arrow_type):
            return ddl_name
    return "string"


def write_arrow_to_parquet_for_spark(
    table: pa.Table,
    path: str,
    compression: str = "zstd",
    row_group_size: int = 1_000_000,
    **pyarrow_kwargs: Any,
) -> str:
    """Write an Arrow table to Parquet for distributed Spark loading.

    For results that fit on the driver, :func:`arrow_to_spark` is
    fastest. For results that do not fit comfortably (multi-GB) the
    pattern is to land the Arrow table as Parquet on a path Spark can
    read in parallel (local FS, S3, HDFS) and let the Spark executors
    do the heavy lifting:

    .. code-block:: python

        path = write_arrow_to_parquet_for_spark(table, "s3://bucket/x")
        df = spark.read.parquet(path)

    Args:
        table: Arrow table to land.
        path: Destination path. Anything :func:`pyarrow.parquet.write_table`
            accepts works (local, S3 via ``pyarrow.fs``, etc.).
        compression: Parquet compression codec. ``zstd`` matches the
            wire format used elsewhere in this SDK.
        row_group_size: Rows per Parquet row group. Larger groups
            improve scan throughput at the cost of memory during write.
        **pyarrow_kwargs: Forwarded to
            :func:`pyarrow.parquet.write_table`.

    Returns:
        The ``path`` argument, for chained ``spark.read.parquet`` calls.
    """
    pq.write_table(
        table,
        path,
        compression=compression,
        row_group_size=row_group_size,
        **pyarrow_kwargs,
    )
    return path
