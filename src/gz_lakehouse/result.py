"""Materialised query result returned by :class:`LakehouseClient.query`.

Wraps a :class:`pyarrow.Table` and exposes ergonomic conversions to the
common in-memory representations: pandas, plain Python, and Spark. The
Arrow table is the canonical form — it is the same byte layout the
provider streams to the client over presigned S3 URLs — so callers
never pay a Python-object conversion cost unless they explicitly ask
for one via :meth:`to_pandas`, :meth:`to_list`, :meth:`to_spark`, or
:meth:`to_spark_via_parquet`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from gz_lakehouse._spark import (
    arrow_to_spark,
    write_arrow_to_parquet_for_spark,
)

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

    from gz_lakehouse._transport import TransportTimings


class QueryResult:
    """Holds the Arrow table and metadata returned by the provider."""

    def __init__(
        self,
        table: pa.Table,
        schema: list[Mapping[str, Any]],
        truncated: bool,
        total_rows: int,
        timings: TransportTimings | None = None,
    ) -> None:
        """Initialise the result wrapper.

        Args:
            table: Arrow table holding the materialised rows.
            schema: Provider-supplied column descriptors. Each entry
                exposes ``columnName`` and ``dataType``. The Arrow
                table's own schema is the authoritative type source;
                this list is preserved for round-trip use cases.
            truncated: True when the provider capped the result and
                more rows would be available with a different read path.
            total_rows: Number of rows the provider materialised.
            timings: Wall-clock breakdown for the execution. ``None``
                only when the result is constructed by callers that
                bypass the transport (tests, fanned-out subqueries
                whose individual timings are merged elsewhere).
        """
        self._table = table
        self._schema = schema
        self._truncated = truncated
        self._total_rows = total_rows
        self._timings = timings

    @property
    def schema(self) -> list[Mapping[str, Any]]:
        """Provider-supplied column metadata."""
        return list(self._schema)

    @property
    def arrow_schema(self) -> pa.Schema:
        """Authoritative Arrow schema for the materialised table."""
        return self._table.schema

    @property
    def total_rows(self) -> int:
        """Number of rows in this result."""
        return self._total_rows

    @property
    def truncated(self) -> bool:
        """True when the provider reported ``hasMore`` on this result."""
        return self._truncated

    @property
    def timings(self) -> TransportTimings | None:
        """Wall-clock breakdown for the execution that produced this result.

        ``None`` for results constructed without a transport (test
        fixtures, manually-merged fan-outs). Otherwise exposes
        ``submit_seconds`` (HTTP submit + pod execution + presigning),
        ``download_seconds`` (parallel chunk fetch + Arrow parse),
        ``chunk_count``, and on-the-wire / decoded byte counts.
        """
        return self._timings

    def to_arrow(self) -> pa.Table:
        """Return the underlying :class:`pyarrow.Table`."""
        return self._table

    def to_pandas(self) -> pd.DataFrame:
        """Convert the rows to a :class:`pandas.DataFrame`.

        Requires the optional ``pandas`` extra to be installed.
        """
        return self._table.to_pandas()

    def to_list(self) -> list[Mapping[str, Any]]:
        """Return the rows as a list of dicts (one per row)."""
        return self._table.to_pylist()

    def to_spark(self, spark: SparkSession) -> SparkDataFrame:
        """Convert the rows to a Spark :class:`DataFrame`, fast.

        Picks the fastest available conversion path based on the
        running PySpark version: Arrow Table direct on 3.4+, Arrow-
        enabled pandas on 3.0–3.3, plain pandas with a warning on
        anything older. Empty tables retain their schema so downstream
        Spark code keeps the column metadata.

        For results larger than a few hundred MB consider
        :meth:`to_spark_via_parquet`, which lets Spark executors load
        the result in parallel instead of pushing every row through
        the Python driver.

        Args:
            spark: An active :class:`pyspark.sql.SparkSession`.

        Returns:
            A Spark DataFrame whose schema matches the Arrow table's.
        """
        return arrow_to_spark(self._table, spark)

    def to_spark_via_parquet(
        self,
        spark: SparkSession,
        path: str,
        compression: str = "zstd",
        row_group_size: int = 1_000_000,
    ) -> SparkDataFrame:
        """Land the result as Parquet, then read it via Spark.

        Recommended for multi-GB results where the Python driver would
        otherwise be the bottleneck. The Arrow table is written to
        ``path`` (local FS, S3, HDFS — anything pyarrow can write) and
        the returned Spark DataFrame loads it through Spark's native
        Parquet reader, which fans out across executors.

        Args:
            spark: An active :class:`pyspark.sql.SparkSession`.
            path: Destination path for the Parquet write.
            compression: Parquet compression codec.
            row_group_size: Rows per Parquet row group.

        Returns:
            ``spark.read.parquet(path)`` for the freshly-written file.
        """
        write_arrow_to_parquet_for_spark(
            self._table,
            path,
            compression=compression,
            row_group_size=row_group_size,
        )
        return spark.read.parquet(path)

    def iter_batches(
        self,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Yield :class:`pyarrow.RecordBatch` slices of the materialised table.

        Streaming directly from the network requires
        :meth:`LakehouseClient.iter_batches`; this method is a
        convenience for callers that already have a materialised result
        and want bounded-memory iteration over it.
        """
        return iter(self._table.to_batches(max_chunksize=batch_size))

    def __len__(self) -> int:
        """Return ``total_rows`` so :func:`len` reports the row count."""
        return self._total_rows

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Iterate over rows as dicts."""
        return iter(self._table.to_pylist())

    def __repr__(self) -> str:
        """Return a short, side-effect-free representation."""
        columns = ", ".join(
            f"{field.name}:{field.type}" for field in self._table.schema
        )
        return (
            f"QueryResult(rows={self._total_rows}, "
            f"truncated={self._truncated}, columns=[{columns}])"
        )
