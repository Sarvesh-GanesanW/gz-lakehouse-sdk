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

import queue
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from gz_lakehouse._logging import get_logger
from gz_lakehouse.exceptions import (
    QueryExecutionError,
    QueryValidationError,
)
from gz_lakehouse.result import QueryResult

if TYPE_CHECKING:
    from gz_lakehouse._transport import ExecutorChoice, Transport
    from gz_lakehouse.config import LakehouseConfig
    from gz_lakehouse.pipeline_config import PipelineConfig

_SPLIT_QUEUE_DEPTH_PER_THREAD = 4
_SPLIT_THREAD_JOIN_TIMEOUT = 5.0

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
        executor: ExecutorChoice = "auto",
        pipeline: PipelineConfig | None = None,
    ) -> QueryResult:
        """Execute ``sql`` and return the materialised :class:`QueryResult`.

        ``executor`` overrides server-side path selection. ``"auto"``
        leaves it to the pod (PyIceberg fast path for simple SELECTs,
        Spark for everything else). ``"fast"`` forces the fast path
        and errors if not eligible. ``"spark"`` forces Spark even on
        eligible queries — useful for A/B comparison.

        ``pipeline`` is an optional :class:`PipelineConfig` that
        tunes the fast-path pod-side pipeline (encoder count, upload
        concurrency, batch sizing, compression level, etc.). Pass it
        when you know the workload shape and want to override server
        defaults — e.g. ``zstd_level=0`` for same-region clients,
        ``num_encoders=12`` for a large pod.
        """
        self._validate_sql(sql)
        self._ensure_open()
        outcome = self._transport.execute(
            self._session_id,
            sql,
            executor=executor,
            pipeline=pipeline,
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
        executor: ExecutorChoice = "auto",
        pipeline: PipelineConfig | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Stream the result of ``sql`` as :class:`pyarrow.RecordBatch`.

        ``executor`` and ``pipeline`` have the same meaning as on
        :meth:`query`.
        """
        self._validate_sql(sql)
        self._ensure_open()
        return self._transport.iter_batches(
            self._session_id,
            sql,
            batch_size=batch_size,
            executor=executor,
            pipeline=pipeline,
        )

    def iter_batches_split(
        self,
        sql: str,
        split_by: str,
        splits: int = 4,
        bounds: tuple[Any, Any] | None = None,
        batch_size: int = 65_536,
        executor: ExecutorChoice = "auto",
        pipeline: PipelineConfig | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Stream a query split across N parallel range queries.

        Splits ``[low, high]`` of ``split_by`` into ``splits`` uniform
        sub-ranges, runs each as a parallel statement on this session,
        and interleaves the resulting :class:`pyarrow.RecordBatch`
        objects to the caller. Useful when one statement would exceed
        the per-request timeout cap or saturate a single pod's
        encoder pool.

        ``bounds`` is auto-probed via ``SELECT MIN(col), MAX(col)
        FROM (sql) AS sub`` if not supplied. Pass it explicitly when
        you already know the range — a single ``MIN/MAX`` over a
        billion-row table is itself expensive.

        Trade-offs:

        * ``split_by`` must be an integer or timestamp column with
          reasonable distribution across the range. A skewed column
          (90% in one sub-range) will tail-bottleneck on one split.
        * Splits run concurrently against the same session pod, so
          encoder + upload pools are shared across them. Pick
          ``splits`` to roughly match the pod's ``parallel_workers``
          for best balance.
        * Row order across splits is undefined. Within a split,
          order is preserved as in :meth:`iter_batches`.
        """
        self._validate_sql(sql)
        self._ensure_open()
        if not isinstance(splits, int) or splits < 1:
            raise QueryValidationError(
                "iter_batches_split requires splits >= 1",
            )
        if not split_by or not isinstance(split_by, str):
            raise QueryValidationError(
                "iter_batches_split requires a non-empty split_by column",
            )

        resolved_bounds = bounds or self._probe_split_bounds(sql, split_by)
        low, high = resolved_bounds
        if low is None or high is None:
            raise QueryExecutionError(
                f"split_by={split_by!r} bounds resolved to NULL; "
                f"pass bounds explicitly or pick another column",
            )

        range_sqls = _build_split_sqls(sql, split_by, low, high, splits)
        _logger.info(
            "iter_batches_split sql=%s split_by=%s splits=%d bounds=(%s, %s)",
            _truncate_sql(sql),
            split_by,
            splits,
            low,
            high,
        )

        if splits == 1:
            yield from self._transport.iter_batches(
                self._session_id,
                range_sqls[0],
                batch_size=batch_size,
                executor=executor,
                pipeline=pipeline,
            )
            return

        yield from _interleave_iter_batches(
            transport=self._transport,
            session_id=self._session_id,
            sqls=range_sqls,
            batch_size=batch_size,
            executor=executor,
            pipeline=pipeline,
        )

    def _probe_split_bounds(
        self,
        sql: str,
        split_by: str,
    ) -> tuple[Any, Any]:
        """Run MIN/MAX over the source query to find split bounds."""
        probe_sql = (
            f"SELECT MIN({split_by}) AS gz_split_lo, "
            f"MAX({split_by}) AS gz_split_hi "
            f"FROM ({sql}) AS gz_split_src"
        )
        outcome = self._transport.execute(self._session_id, probe_sql)
        if outcome.table.num_rows == 0:
            raise QueryExecutionError(
                "split bounds probe returned an empty result",
            )
        lo = outcome.table.column("gz_split_lo").to_pylist()[0]
        hi = outcome.table.column("gz_split_hi").to_pylist()[0]
        return (lo, hi)

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


def _build_split_sqls(
    sql: str,
    split_by: str,
    low: Any,
    high: Any,
    splits: int,
) -> list[str]:
    """Divide ``[low, high]`` into ``splits`` uniform sub-ranges.

    Currently supports integer and float bounds; non-numeric bounds
    fall back to a single un-split range so callers don't need to
    special-case timestamps or strings client-side. The last split
    uses the original ``high`` to avoid floating-point drift losing
    rows at the upper edge.
    """
    is_numeric = isinstance(low, (int, float)) and not isinstance(low, bool)
    if splits == 1 or not is_numeric:
        return [_compose_partitioned_sql(sql, split_by, low, high)]

    span = high - low
    step = span / splits
    statements: list[str] = []
    for i in range(splits):
        sub_low = low + i * step
        sub_high = high if i == splits - 1 else low + (i + 1) * step
        if isinstance(low, int) and isinstance(high, int):
            sub_low = int(round(sub_low))
            sub_high = int(round(sub_high))
        statements.append(
            _compose_partitioned_sql(sql, split_by, sub_low, sub_high),
        )
    return statements


def _interleave_iter_batches(
    transport: Transport,
    session_id: str,
    sqls: Sequence[str],
    batch_size: int,
    executor: ExecutorChoice,
    pipeline: PipelineConfig | None,
) -> Iterator[pa.RecordBatch]:
    """Run ``sqls`` concurrently and interleave their batches.

    Each SQL gets its own worker thread that pushes batches into a
    bounded shared queue; the main generator yields from the queue
    in arrival order. A shutdown event is set on the first error so
    in-flight workers can exit before the queue stays half-drained.
    """
    out_queue: queue.Queue[Any] = queue.Queue(
        maxsize=max(len(sqls) * _SPLIT_QUEUE_DEPTH_PER_THREAD, len(sqls)),
    )
    sentinel = object()
    error_box: list[BaseException] = []
    shutdown = threading.Event()

    def _drain_one(split_sql: str) -> None:
        try:
            for batch in transport.iter_batches(
                session_id,
                split_sql,
                batch_size=batch_size,
                executor=executor,
                pipeline=pipeline,
            ):
                if shutdown.is_set():
                    return
                out_queue.put(batch)
        except BaseException as ex:
            if not error_box:
                error_box.append(ex)
            shutdown.set()
        finally:
            out_queue.put(sentinel)

    threads = [
        threading.Thread(
            target=_drain_one,
            args=(split_sql,),
            name=f"sdk-split-{i}",
            daemon=True,
        )
        for i, split_sql in enumerate(sqls)
    ]
    for thread in threads:
        thread.start()

    try:
        remaining = len(sqls)
        while remaining > 0:
            item = out_queue.get()
            if item is sentinel:
                remaining -= 1
                continue
            yield item
        if error_box:
            raise error_box[0]
    finally:
        shutdown.set()
        for thread in threads:
            thread.join(timeout=_SPLIT_THREAD_JOIN_TIMEOUT)


def _truncate_sql(sql: str, limit: int = 80) -> str:
    """Return a short preview of an SQL statement for log messages."""
    flat = " ".join(sql.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."
