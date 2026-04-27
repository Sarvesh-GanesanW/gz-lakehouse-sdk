"""Warm compute session for the gz-lakehouse SDK.

A :class:`Session` represents a long-lived compute pod on the
provider side: workers are registered, Spark is warm, and statements
execute fast. Sessions are created by :meth:`LakehouseClient.start_session`
and explicitly stopped by :meth:`Session.stop` (or implicitly via the
context-manager protocol). Multiple statements run on the same session
to amortise the ~17-second pod-boot cost across queries.

Per-statement APIs mirror the convenience methods on
:class:`LakehouseClient`:

* :meth:`query` — materialise a result as :class:`QueryResult`.
* :meth:`iter_batches` — stream record batches with bounded memory.
* :meth:`query_parallel` — fan-out range-partitioned subqueries onto
  the same session pool.

The Session class is the recommended entry point for any caller that
expects to issue more than one statement; the convenience wrappers on
:class:`LakehouseClient` (``client.query`` etc.) auto-create and stop a
session per call, which is plug-and-play but pays the boot cost every
time.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from gz_lakehouse._logging import get_logger
from gz_lakehouse.exceptions import QueryValidationError
from gz_lakehouse.result import QueryResult

if TYPE_CHECKING:
    from gz_lakehouse._transport import Transport
    from gz_lakehouse.config import LakehouseConfig

_logger = get_logger("session")


class Session:
    """Warm session bound to a running compute pod on the provider.

    Construct via :meth:`LakehouseClient.start_session`. Use as a
    context manager when possible so ``stop()`` is guaranteed to run
    even on exceptions::

        with client.start_session() as session:
            result = session.query("SELECT * FROM orders LIMIT 1000")
    """

    def __init__(
        self,
        session_id: str,
        transport: Transport,
        config: LakehouseConfig,
    ) -> None:
        """Hold the sessionId and transport handle for the session."""
        self._session_id = session_id
        self._transport = transport
        self._config = config
        self._closed = False

    @property
    def session_id(self) -> str:
        """The provider-issued sessionId for this session."""
        return self._session_id

    @property
    def closed(self) -> bool:
        """True after :meth:`stop` has run successfully."""
        return self._closed

    def query(
        self,
        sql: str,
    ) -> QueryResult:
        """Execute ``sql`` and return the materialised :class:`QueryResult`."""
        self._validate_sql(sql)
        self._ensure_open()
        outcome = self._transport.execute(
            self._session_id,
            sql,
        )
        return QueryResult(
            table=outcome.table,
            schema=outcome.schema,
            truncated=outcome.truncated,
            total_rows=outcome.total_rows,
            timings=outcome.timings,
        )

    def iter_batches(
        self,
        sql: str,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Stream the result of ``sql`` as :class:`pyarrow.RecordBatch`."""
        self._validate_sql(sql)
        self._ensure_open()
        return self._transport.iter_batches(
            self._session_id,
            sql,
            batch_size=batch_size,
        )

    def query_parallel(
        self,
        sql_template: str,
        partition_column: str,
        bounds: Sequence[tuple[Any, Any]],
        max_workers: int | None = None,
    ) -> QueryResult:
        """Fan range-partitioned subqueries out across the same session.

        See :meth:`LakehouseClient.query_parallel` for the partitioning
        semantics. All subqueries share this session, so worker
        registration cost is paid once.
        """
        if not bounds:
            raise QueryValidationError(
                "query_parallel requires at least one (low, high) bound",
            )
        if not partition_column or not isinstance(partition_column, str):
            raise QueryValidationError(
                "query_parallel requires a partition_column string",
            )
        self._validate_sql(sql_template)
        self._ensure_open()

        statements = [
            _compose_partitioned_sql(sql_template, partition_column, low, high)
            for low, high in bounds
        ]
        workers = max(1, max_workers or self._config.parallel_workers)
        _logger.info(
            "query_parallel partitions=%s workers=%s session=%s",
            len(statements),
            workers,
            self._session_id,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(
                pool.map(
                    lambda sql: self._transport.execute(
                        self._session_id,
                        sql,
                    ),
                    statements,
                ),
            )

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

    def stop(self) -> None:
        """Tear down the session pod. Safe to call multiple times."""
        if self._closed:
            return
        try:
            self._transport.stop_session(self._session_id)
        finally:
            self._closed = True

    def __enter__(self) -> Session:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the session when the ``with`` block exits."""
        self.stop()

    def _ensure_open(self) -> None:
        """Reject calls made after :meth:`stop`."""
        if self._closed:
            raise QueryValidationError(
                "Session is stopped; create a new one to run more queries",
            )

    @staticmethod
    def _validate_sql(sql: str) -> None:
        """Reject empty or non-string SQL early."""
        if not sql or not isinstance(sql, str):
            raise QueryValidationError(
                "query() requires a non-empty SQL string",
            )


def _compose_partitioned_sql(
    sql_template: str,
    partition_column: str,
    low: Any,
    high: Any,
) -> str:
    """Append ``WHERE col BETWEEN low AND high`` to ``sql_template``."""
    rendered_low = _render_literal(low)
    rendered_high = _render_literal(high)
    clause = f"{partition_column} BETWEEN {rendered_low} AND {rendered_high}"
    upper = sql_template.upper()
    if " WHERE " in f" {upper} ":
        return f"{sql_template} AND {clause}"
    return f"{sql_template} WHERE {clause}"


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
