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

import contextlib
import queue
import re
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import date, datetime
from decimal import Decimal
from types import TracebackType
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from gz_lakehouse._logging import get_logger
from gz_lakehouse._transport import TransportTimings
from gz_lakehouse.exceptions import (
    QueryExecutionError,
    QueryValidationError,
)
from gz_lakehouse.result import QueryResult

if TYPE_CHECKING:
    from gz_lakehouse._transport import (
        ExecutorChoice,
        Transport,
        TransportResult,
    )
    from gz_lakehouse.config import LakehouseConfig
    from gz_lakehouse.pipeline_config import PipelineConfig

_SPLIT_QUEUE_DEPTH_PER_THREAD = 4
_SPLIT_THREAD_JOIN_TIMEOUT = 5.0
_SPLIT_QUEUE_PUT_TIMEOUT_SECONDS = 0.5

_SQL_LITERAL_PATTERN = re.compile(r"'(?:''|[^'])*'")

_FORBIDDEN_OUTER_CLAUSES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ORDER BY", re.compile(r"\bORDER\s+BY\b")),
    ("GROUP BY", re.compile(r"\bGROUP\s+BY\b")),
    ("HAVING", re.compile(r"\bHAVING\b")),
    ("LIMIT", re.compile(r"\bLIMIT\b")),
    ("OFFSET", re.compile(r"\bOFFSET\b")),
    ("FETCH", re.compile(r"\bFETCH\b")),
    ("UNION", re.compile(r"\bUNION\b")),
    ("INTERSECT", re.compile(r"\bINTERSECT\b")),
    ("EXCEPT", re.compile(r"\bEXCEPT\b")),
)
_WHERE_PATTERN = re.compile(r"\bWHERE\b")

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
        _validate_partition_template(sql)

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
        _validate_partition_template(sql_template)

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
        outcomes = self._run_parallel_statements(statements, workers)

        tables = [outcome.table for outcome in outcomes]
        combined = pa.concat_tables(tables, promote_options="default")
        truncated = any(outcome.truncated for outcome in outcomes)
        total_rows = sum(outcome.total_rows for outcome in outcomes)
        schema = outcomes[0].schema if outcomes else []
        merged_timings = _merge_timings(
            [outcome.timings for outcome in outcomes],
        )
        return QueryResult(
            table=combined,
            schema=schema,
            truncated=truncated,
            total_rows=total_rows,
            timings=merged_timings,
        )

    def _run_parallel_statements(
        self,
        statements: Sequence[str],
        workers: int,
    ) -> list[TransportResult]:
        """Submit ``statements`` concurrently with fail-fast cancellation.

        Submits all statements up front, then drains in completion order
        so a fast-failing partition surfaces immediately rather than
        waiting for the slowest leg. On the first failure, every
        not-yet-running future is cancelled and the pool is shut down
        with ``cancel_futures=True`` so the user does not pay for
        downloads that will be discarded. Outcomes are returned in
        submission order so concatenation is stable across runs.
        """
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="sdk-parallel",
        ) as pool:
            futures: list[Future[TransportResult]] = [
                pool.submit(
                    self._transport.execute,
                    self._session_id,
                    sql,
                )
                for sql in statements
            ]
            future_index = {f: i for i, f in enumerate(futures)}
            results: list[TransportResult | None] = [None] * len(futures)
            failure: BaseException | None = None
            for completed in as_completed(futures):
                if failure is not None:
                    completed.cancel()
                    continue
                try:
                    results[future_index[completed]] = completed.result()
                except BaseException as ex:
                    failure = ex
                    for pending in futures:
                        if pending is not completed:
                            pending.cancel()
            if failure is not None:
                pool.shutdown(wait=False, cancel_futures=True)
                raise failure
        return [outcome for outcome in results if outcome is not None]

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


def _strip_sql_literals(sql: str) -> str:
    """Replace single-quoted string literals with empty quotes.

    Lets keyword scanning ignore literals that *contain* SQL keywords —
    a column value of ``'WHERE NOT FOUND'`` must not register as a
    real ``WHERE`` clause.
    """
    return _SQL_LITERAL_PATTERN.sub("''", sql)


def _is_at_depth_zero(stripped_upper: str, position: int) -> bool:
    """Return True when ``position`` is outside any parenthesised group."""
    depth = 0
    for ch in stripped_upper[:position]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
    return depth == 0


def _has_outer_match(stripped_upper: str, pattern: re.Pattern[str]) -> bool:
    """Return True when ``pattern`` matches at outermost paren depth."""
    return any(
        _is_at_depth_zero(stripped_upper, m.start())
        for m in pattern.finditer(stripped_upper)
    )


def _validate_partition_template(sql: str) -> None:
    """Reject templates the SDK cannot safely append a WHERE/AND to.

    Outer-level ORDER BY / GROUP BY / HAVING / LIMIT / OFFSET / FETCH
    and set operators (UNION / INTERSECT / EXCEPT) all conflict with
    appending a partition predicate. The SDK rejects these instead of
    silently producing broken SQL — wrap the query as a subquery if
    you need them: ``SELECT * FROM (<your sql>) AS sub``.

    Inside parenthesised subqueries / CTEs these clauses are fine; the
    depth-aware scanner only flags them at the outermost level.
    """
    stripped_upper = _strip_sql_literals(sql).upper()
    for label, pattern in _FORBIDDEN_OUTER_CLAUSES:
        if _has_outer_match(stripped_upper, pattern):
            raise QueryValidationError(
                f"Partition template must not contain a top-level "
                f"{label} clause: the SDK appends a WHERE/AND for the "
                f"partition column. Wrap as a subquery if needed: "
                f"SELECT * FROM (<your sql>) AS sub",
            )


def _compose_partitioned_sql(
    sql_template: str,
    partition_column: str,
    low: Any,
    high: Any,
) -> str:
    """Append ``WHERE col BETWEEN low AND high`` to ``sql_template``.

    Uses depth-aware WHERE detection so a ``WHERE`` inside a CTE or
    subquery doesn't trigger AND-mode for the outer clause, and
    string literals containing the substring ``WHERE`` don't false-match.
    Callers should run :func:`_validate_partition_template` first to
    reject templates with terminal clauses that would break composition.
    """
    rendered_low = _render_literal(low)
    rendered_high = _render_literal(high)
    clause = f"{partition_column} BETWEEN {rendered_low} AND {rendered_high}"
    stripped_upper = _strip_sql_literals(sql_template).upper()
    if _has_outer_match(stripped_upper, _WHERE_PATTERN):
        return f"{sql_template} AND {clause}"
    return f"{sql_template} WHERE {clause}"


def _render_literal(value: Any) -> str:
    """Render a Python value as a SQL literal for partition bounds.

    The wire protocol does not support parameterized queries for
    partition bounds — the SDK has to inline the value into the
    BETWEEN predicate. This is safe for *trusted* call-site bounds
    (programmatic ranges, IDs from a manifest, computed timestamps),
    which is the only intended use of fan-out APIs. Do not pass
    bounds derived from untrusted user input here; the SDK escapes
    single quotes in strings the ANSI-correct way (``'`` → ``''``)
    so a single-quoted string cannot break out of its literal, but
    it cannot defend against type-confusion or the user passing a
    raw SQL fragment as a "bound."

    Native types render with their typed SQL syntax so the provider
    does not have to guess at coercion:

    * ``int`` / ``float`` / ``Decimal`` → unquoted numeric literal
    * ``bool`` → ``TRUE`` / ``FALSE``
    * ``None`` → ``NULL``
    * ``datetime.date`` → ``DATE 'YYYY-MM-DD'``
    * ``datetime.datetime`` → ``TIMESTAMP 'YYYY-MM-DD HH:MM:SS[.ffffff]'``
    * ``str`` → escaped, single-quoted ANSI string

    Bytes / arbitrary objects are rejected outright: there is no
    well-defined SQL representation, and silently calling ``str()`` on
    them would mask a caller-side bug.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    raise QueryValidationError(
        f"Cannot render partition bound of type {type(value).__name__!r} "
        f"as a SQL literal; pass int, float, Decimal, bool, str, "
        f"datetime.date, or datetime.datetime",
    )


def _merge_timings(
    timings: Sequence[TransportTimings],
) -> TransportTimings | None:
    """Merge per-statement timings into one aggregate for fan-out queries.

    Wall-clock fields (``submit_seconds``, ``download_seconds``) take
    the *max* across legs since legs run in parallel — the slowest leg
    bounds total wall time on each phase. Volume fields
    (``chunk_count`` and byte counters) sum across legs, since they
    measure aggregate throughput rather than wall time. ``executor``
    surfaces a single value when every leg used the same engine, or
    ``"mixed"`` when legs split between paths (e.g. some legs landed
    on the fast path and some fell back to Spark).
    """
    if not timings:
        return None
    distinct_executors = {t.executor for t in timings if t.executor}
    if len(distinct_executors) == 0:
        merged_executor: str | None = None
    elif len(distinct_executors) == 1:
        merged_executor = next(iter(distinct_executors))
    else:
        merged_executor = "mixed"
    return TransportTimings(
        submit_seconds=max(t.submit_seconds for t in timings),
        download_seconds=max(t.download_seconds for t in timings),
        chunk_count=sum(t.chunk_count for t in timings),
        compressed_bytes=sum(t.compressed_bytes for t in timings),
        uncompressed_bytes=sum(t.uncompressed_bytes for t in timings),
        executor=merged_executor,
    )


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
    in arrival order. A shutdown event is set on the first error or
    on consumer exit so in-flight workers can exit promptly instead
    of running to completion.

    Cancellation is co-operative across two failure modes:

    * Worker discovers the consumer has stopped (consumer breaks out
      of the ``for batch in iter_batches_split(...)`` loop, the main
      generator's ``finally`` sets ``shutdown``). Workers test
      ``shutdown`` between batches AND use a bounded ``queue.put``
      with a small timeout so a full queue with no consumer doesn't
      pin the worker forever.
    * Worker raises (a leg's transport.iter_batches fails). The
      first exception is captured, ``shutdown`` is set, the main
      generator drains its sentinel count and re-raises.

    The main generator's ``finally`` also drains anything left in
    the queue so a worker blocked on ``put`` can make progress to
    its sentinel and exit.
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
                while not shutdown.is_set():
                    try:
                        out_queue.put(
                            batch,
                            timeout=_SPLIT_QUEUE_PUT_TIMEOUT_SECONDS,
                        )
                        break
                    except queue.Full:
                        continue
                else:
                    return
        except BaseException as ex:
            if not error_box:
                error_box.append(ex)
            shutdown.set()
        finally:
            with contextlib.suppress(queue.Full):
                out_queue.put(
                    sentinel,
                    timeout=_SPLIT_QUEUE_PUT_TIMEOUT_SECONDS,
                )

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
        while True:
            try:
                out_queue.get_nowait()
            except queue.Empty:
                break
        for thread in threads:
            thread.join(timeout=_SPLIT_THREAD_JOIN_TIMEOUT)


def _truncate_sql(sql: str, limit: int = 80) -> str:
    """Return a short preview of an SQL statement for log messages."""
    flat = " ".join(sql.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."
