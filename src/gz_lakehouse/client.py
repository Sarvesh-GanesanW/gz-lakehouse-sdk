"""High-level client for the GroundZero Lakehouse provider.

The :class:`LakehouseClient` wraps the provider behind a small,
ergonomic surface. Internally it delegates to a :class:`Transport`
that picks the fastest available data plane (Arrow IPC, chunked
Parquet on S3, or the legacy JSON envelope) via HTTP content
negotiation, so callers get the same API regardless of which path the
provider supports today.

Three execution modes are exposed:

* :meth:`query` returns a fully materialised :class:`QueryResult`.
* :meth:`iter_batches` streams the result as
  :class:`pyarrow.RecordBatch` instances; memory stays bounded
  regardless of result size.
* :meth:`query_parallel` fans out range-partitioned subqueries over a
  thread pool and concatenates their tables, working around the
  legacy 10K row cap and using bandwidth on multi-chunk responses.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any

import pyarrow as pa

from gz_lakehouse._http import HttpClient
from gz_lakehouse._logging import get_logger
from gz_lakehouse._transport import Transport
from gz_lakehouse.config import LakehouseConfig
from gz_lakehouse.exceptions import (
    ConfigurationError,
    QueryValidationError,
)
from gz_lakehouse.result import QueryResult

_logger = get_logger("client")


class LakehouseClient:
    """Client for interacting with a GroundZero Lakehouse provider."""

    def __init__(self, config: LakehouseConfig) -> None:
        """Initialise the client with the supplied configuration.

        Args:
            config: A validated :class:`LakehouseConfig`.
        """
        if not isinstance(config, LakehouseConfig):
            raise ConfigurationError(
                "LakehouseClient requires a LakehouseConfig instance"
            )
        self._config = config
        self._http = HttpClient(
            base_url=config.lakehouse_url,
            site=config.derived_site,
            connect_timeout_seconds=config.connect_timeout_seconds,
            max_retries=config.max_retries,
            backoff_seconds=config.backoff_seconds,
            pool_connections=config.pool_connections,
            pool_maxsize=config.pool_maxsize,
            enable_compression=config.enable_compression,
        )
        self._transport = Transport(http=self._http, config=config)
        self._verified = False

    @classmethod
    def from_kwargs(
        cls,
        lakehouse_url: str,
        warehouse: str,
        database: str,
        username: str,
        password: str,
        site: str | None = None,
        compute_size: str = "small",
        compute_id: int | None = None,
        **extra: Any,
    ) -> LakehouseClient:
        """Build a client without explicitly constructing a config."""
        config = LakehouseConfig(
            lakehouse_url=lakehouse_url,
            warehouse=warehouse,
            database=database,
            username=username,
            password=password,
            site=site,
            compute_size=compute_size,
            compute_id=compute_id,
            **extra,
        )
        return cls(config)

    @classmethod
    def from_env(
        cls,
        prefix: str = "GZ_LAKEHOUSE_",
        **overrides: Any,
    ) -> LakehouseClient:
        """Build a client from environment variables (see
        :meth:`LakehouseConfig.from_env`).
        """
        return cls(LakehouseConfig.from_env(prefix=prefix, **overrides))

    def test_connection(self) -> bool:
        """Verify the provider is reachable and the credentials are valid.

        Returns:
            ``True`` when the provider returns HTTP 200.

        Raises:
            AuthenticationError: When credentials are rejected.
            QueryError: When the warehouse or database does not exist.
            TransportError: When the provider URL is unreachable.
        """
        self._transport.verify()
        self._verified = True
        return True

    def query(self, sql: str) -> QueryResult:
        """Execute a SQL query and return the materialised result.

        On the first call the client verifies the provider via
        ``/iceberg/testconnection`` to surface auth and warehouse errors
        early. Subsequent calls reuse that verification.

        Args:
            sql: SQL statement to execute on the lakehouse.

        Returns:
            A :class:`QueryResult` wrapping the rows and schema.

        Raises:
            AuthenticationError: When credentials are rejected.
            QueryError: When the provider returns an execution error.
            TransportError: When the provider URL is unreachable.
            QueryValidationError: When ``sql`` is empty or not a string.
        """
        self._validate_sql(sql)
        self._ensure_verified()

        outcome = self._transport.execute(sql)
        return QueryResult(
            table=outcome.table,
            schema=outcome.schema,
            truncated=outcome.truncated,
            total_rows=outcome.total_rows,
        )

    def iter_batches(
        self,
        sql: str,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Stream the result of ``sql`` as :class:`pyarrow.RecordBatch`.

        Memory stays bounded regardless of total result size: on the
        Arrow IPC path each batch is yielded as it arrives, on the
        Parquet-chunked path each chunk is downloaded sequentially, and
        on the JSON path the materialised table is sliced into
        ``batch_size`` row chunks before being yielded.
        """
        self._validate_sql(sql)
        self._ensure_verified()
        return self._transport.iter_batches(sql, batch_size=batch_size)

    def query_parallel(
        self,
        sql_template: str,
        partition_column: str,
        bounds: Sequence[tuple[Any, Any]],
        max_workers: int | None = None,
    ) -> QueryResult:
        """Fan out range-partitioned subqueries and concatenate the results.

        Each entry in ``bounds`` is a ``(low, high)`` pair that becomes a
        ``WHERE partition_column BETWEEN low AND high`` clause appended
        to ``sql_template``. The subqueries run concurrently on a
        :class:`ThreadPoolExecutor` and the resulting Arrow tables are
        concatenated in submission order.

        This is the practical workaround while the provider still
        enforces a per-query row cap: the caller picks a numeric or
        date column with a known distribution, splits its range into N
        bounds, and gets the full result in roughly ``1/N`` of the
        wall-clock time.

        Args:
            sql_template: SQL with no trailing semicolon. The ``WHERE``
                clause is appended (or ``AND``-joined if one exists).
            partition_column: Column to partition on (case sensitive,
                must be quoted in the template if your dialect requires
                it).
            bounds: Sequence of inclusive ``(low, high)`` pairs.
            max_workers: Override for the worker count. Defaults to
                ``LakehouseConfig.parallel_workers``.

        Returns:
            A :class:`QueryResult` whose Arrow table is the
            concatenation of the per-partition tables.

        Raises:
            QueryValidationError: When the inputs are malformed.
        """
        if not bounds:
            raise QueryValidationError(
                "query_parallel requires at least one (low, high) bound"
            )
        if not partition_column or not isinstance(partition_column, str):
            raise QueryValidationError(
                "query_parallel requires a partition_column string"
            )
        self._validate_sql(sql_template)
        self._ensure_verified()

        statements = [
            self._compose_partitioned_sql(
                sql_template, partition_column, low, high
            )
            for low, high in bounds
        ]
        workers = max(1, max_workers or self._config.parallel_workers)
        _logger.info(
            "query_parallel partitions=%s workers=%s",
            len(statements),
            workers,
        )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(self._transport.execute, statements))

        tables = [outcome.table for outcome in outcomes]
        combined = pa.concat_tables(tables, promote_options="default")
        truncated = any(outcome.truncated for outcome in outcomes)
        total_rows = sum(outcome.total_rows for outcome in outcomes)
        schema = outcomes[0].schema if outcomes else []
        return QueryResult(
            table=combined,
            schema=schema,
            truncated=truncated,
            total_rows=total_rows,
        )

    def _ensure_verified(self) -> None:
        """Run the verification handshake the first time it's needed."""
        if not self._verified:
            self.test_connection()

    @staticmethod
    def _validate_sql(sql: str) -> None:
        """Reject empty or non-string SQL early with a typed exception."""
        if not sql or not isinstance(sql, str):
            raise QueryValidationError(
                "query() requires a non-empty SQL string"
            )

    @staticmethod
    def _compose_partitioned_sql(
        sql_template: str,
        partition_column: str,
        low: Any,
        high: Any,
    ) -> str:
        """Compose a partitioned SQL by appending a BETWEEN clause."""
        rendered_low = LakehouseClient._render_literal(low)
        rendered_high = LakehouseClient._render_literal(high)
        clause = (
            f"{partition_column} BETWEEN {rendered_low} AND {rendered_high}"
        )
        upper = sql_template.upper()
        if " WHERE " in f" {upper} ":
            return f"{sql_template} AND {clause}"
        return f"{sql_template} WHERE {clause}"

    @staticmethod
    def _render_literal(value: Any) -> str:
        """Render a Python value as a SQL literal for partition bounds."""
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return "NULL"
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    def close(self) -> None:
        """Release the underlying HTTP connection pools."""
        self._transport.close()
        self._http.close()

    def __enter__(self) -> LakehouseClient:
        """Enter a context-manager block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the HTTP session when leaving a ``with`` block."""
        self.close()
