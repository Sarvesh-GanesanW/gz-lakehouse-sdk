"""Snowflake-style data plane for the gz-lakehouse SDK.

The provider hands the client a small JSON envelope listing presigned
S3 URLs to Arrow IPC result chunks; the client fans the chunk
downloads out across a thread pool, parses each chunk straight into a
:class:`pyarrow.Table` with no Python-object intermediate, and
concatenates the per-chunk tables in submission order.

Wire shape returned by ``POST /iceberg/v1/statements``::

    {
        "schema": [
            {"columnName": "id",   "dataType": "BIGINT"},
            {"columnName": "name", "dataType": "VARCHAR"}
        ],
        "totalRecords": 10500000,
        "hasMore": false,
        "chunks": [
            {
                "url": "https://s3.<region>.amazonaws.com/...&X-Amz-...",
                "rowCount": 200000,
                "compressedSize": 19834234,
                "uncompressedSize": 67234234
            },
            ...
        ]
    }

Each chunk URL is a presigned S3 GET to an Arrow IPC stream
(zstd-compressed at the IPC body level). The client neither needs nor
expects any other wire format — the legacy JSON-row envelope has been
removed.

Throughput scales linearly with ``parallel_workers`` until the S3 /
network ceiling is hit. With 20 MB chunks and 8 workers a 1 GB result
typically lands in 2–3 seconds on a 1 Gb/s link.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import httpx
import pyarrow as pa
import pyarrow.ipc as paipc
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gz_lakehouse._arrow_build import (
    empty_table_for,
    schema_to_descriptors,
)
from gz_lakehouse._http import HttpClient
from gz_lakehouse._logging import get_logger
from gz_lakehouse.exceptions import (
    QueryExecutionError,
    TransportError,
)
from gz_lakehouse.pipeline_config import PipelineConfig

ExecutorChoice = Literal["auto", "fast", "spark"]
_VALID_EXECUTORS = ("auto", "fast", "spark")

if TYPE_CHECKING:
    from gz_lakehouse.config import LakehouseConfig

_STATEMENTS_PATH = "/iceberg/v1/statements"
_START_SESSION_PATH = "/iceberg/startsession"
_STOP_SESSION_PATH = "/iceberg/stopsession"
_VERIFY_PATH = "/iceberg/testconnection"

_DOWNLOAD_BUFFER_BYTES = 1 << 20

_S3_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_CHUNK_BACKOFF_CEILING_SECONDS = 30.0

_DEFER_POLL_MIN_SECONDS = 1.0
_DEFER_POLL_MAX_SECONDS = 30.0


class _IterBytesReader(io.RawIOBase):
    """Adapt an ``Iterator[bytes]`` into a ``read(n)``-style file object.

    pyarrow's :class:`pyarrow.ipc.RecordBatchStreamReader` reads the IPC
    stream by asking for specific byte counts, but httpx streaming
    yields whatever frame size the underlying TCP / h2 layer hands
    back. This adapter maintains a small overflow buffer so a tiny
    ``read(8)`` call after a 64 KB iter chunk does not throw away the
    remainder of that chunk. Used by the HTTP/2 chunk download path
    so the IPC reader can pull bytes off the wire incrementally
    instead of waiting for ``response.content`` to materialise the
    whole body in memory.
    """

    def __init__(self, source: Iterator[bytes]) -> None:
        """Wrap a byte iterator with a residual buffer."""
        super().__init__()
        self._source = source
        self._residual = b""
        self._exhausted = False

    def readable(self) -> bool:
        """Always readable; pyarrow checks this before reading."""
        return True

    def readinto(self, buffer: Any) -> int:
        """Fill ``buffer`` with up to ``len(buffer)`` bytes from source."""
        target = len(buffer)
        if target == 0:
            return 0
        while len(self._residual) < target and not self._exhausted:
            try:
                self._residual += next(self._source)
            except StopIteration:
                self._exhausted = True
                break
        emit = min(target, len(self._residual))
        buffer[:emit] = self._residual[:emit]
        self._residual = self._residual[emit:]
        return emit


class _RetryableChunkError(Exception):
    """Internal sentinel: chunk download failed in a way worth retrying.

    Wraps the underlying exception so the retry loop can preserve the
    original cause for the final TransportError when all attempts run
    out. Never escapes the transport module — callers see
    :class:`TransportError` or :class:`QueryExecutionError` instead.
    """

    def __init__(
        self,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        """Hold a human-readable message plus the original exception."""
        super().__init__(message)
        self.cause = cause


_COMPUTE_SIZE_TO_ID: dict[str, int] = {
    "small": 1003,
    "medium": 1006,
    "large": 1009,
    "xlarge": 1012,
    "2xlarge": 1015,
}

_logger = get_logger("transport")


class TransportTimings:
    """Per-execution wall-clock breakdown captured by :class:`Transport`.

    Useful for benchmarks and ops dashboards: the timings add up to
    total query latency split between server-side compute and
    client-side download. ``executor`` reports which engine the pod
    actually ran the statement on (``"pyiceberg"`` for the fast path,
    ``"spark"`` for full Spark SQL); useful for A/B comparisons and
    for understanding why an "auto" query took the path it did.
    """

    def __init__(
        self,
        submit_seconds: float,
        download_seconds: float,
        chunk_count: int,
        compressed_bytes: int,
        uncompressed_bytes: int,
        executor: str | None = None,
    ) -> None:
        """Hold the timing and size breakdown for a single execution."""
        self.submit_seconds = submit_seconds
        self.download_seconds = download_seconds
        self.chunk_count = chunk_count
        self.compressed_bytes = compressed_bytes
        self.uncompressed_bytes = uncompressed_bytes
        self.executor = executor

    @property
    def total_seconds(self) -> float:
        """Sum of submit + download time."""
        return self.submit_seconds + self.download_seconds


class TransportResult:
    """Outcome of a successful statement execution.

    Always exposes a single :class:`pyarrow.Table` regardless of how
    many chunks the provider returned, alongside metadata other layers
    consume (truncation flag, total row count, descriptor list,
    per-execution timing breakdown).
    """

    def __init__(
        self,
        table: pa.Table,
        truncated: bool,
        total_rows: int,
        schema: list[dict[str, str]],
        timings: TransportTimings,
    ) -> None:
        """Hold the materialised Arrow table and provider metadata."""
        self.table = table
        self.truncated = truncated
        self.total_rows = total_rows
        self.schema = schema
        self.timings = timings


class _Envelope:
    """Parsed metadata returned by ``POST /iceberg/v1/statements``."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        schema: list[dict[str, str]],
        total_rows: int,
        truncated: bool,
    ) -> None:
        """Hold the chunk list and metadata about the result."""
        self.chunks = chunks
        self.schema = schema
        self.total_rows = total_rows
        self.truncated = truncated


class Transport:
    """Submits SQL and downloads Arrow IPC chunks from presigned S3 URLs."""

    def __init__(
        self,
        http: HttpClient,
        config: LakehouseConfig,
    ) -> None:
        """Bind the transport to an HTTP client and configuration."""
        self._http = http
        self._config = config
        self._s3_session = self._build_s3_session()
        self._s3_http2_client: httpx.Client | None = (
            self._build_s3_http2_client() if config.enable_http2 else None
        )
        self._download_pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

    def _submit_chunk(
        self,
        pool: ThreadPoolExecutor,
        event: dict[str, Any],
    ) -> Future[pa.Table]:
        """Submit a chunk decode (inline or S3-presigned) to the pool."""
        if "inline" in event:
            return pool.submit(self._decode_inline_chunk, event)
        return pool.submit(self._download_chunk, event)

    def _get_download_pool(self) -> ThreadPoolExecutor:
        """Return the transport-shared chunk-download thread pool.

        One pool is created lazily and reused across every ``execute``
        and ``iter_batches`` call on this transport. Sharing matters for
        fan-out APIs (``query_parallel``, ``iter_batches_split``): each
        leg used to spawn its own ``parallel_workers``-sized pool, so a
        16-leg fan-out on top of ``parallel_workers=32`` would create
        512 download threads competing for the same network. With one
        shared pool the cap is just ``parallel_workers`` regardless of
        leg count.
        """
        with self._pool_lock:
            if self._download_pool is None:
                self._download_pool = ThreadPoolExecutor(
                    max_workers=self._config.parallel_workers,
                    thread_name_prefix="sdk-download",
                )
            return self._download_pool

    def execute(
        self,
        session_id: str,
        sql: str,
        executor: ExecutorChoice = "auto",
        pipeline: PipelineConfig | None = None,
    ) -> TransportResult:
        """Submit ``sql`` on ``session_id`` and materialise the result.

        Iterates :meth:`_event_stream`, which owns the underlying HTTP
        responses (sync NDJSON, deferred-then-polled NDJSON, and the
        legacy JSON envelope all surface as the same event sequence).
        Every chunk event is dispatched to the shared download pool;
        the resulting Arrow tables are concatenated in submission order
        once the stream terminates with a ``done`` event.
        """
        _validate_executor(executor)
        submit_started = time.monotonic()

        schema: list[dict[str, str]] = []
        pending: list[Future[pa.Table]] = []
        pool = self._get_download_pool()
        first_event_at: float | None = None
        total_rows = 0
        compressed_bytes = 0
        uncompressed_bytes = 0
        used_executor: str | None = None
        download_started: float | None = None
        download_tables: list[pa.Table] = []

        try:
            for event in self._event_stream(
                session_id, sql, executor, pipeline,
            ):
                etype = event.get("type")
                if etype == "heartbeat":
                    continue
                if first_event_at is None:
                    first_event_at = time.monotonic() - submit_started
                if etype == "schema":
                    schema = list(event.get("columns") or [])
                elif etype == "chunk":
                    pending.append(self._submit_chunk(pool, event))
                    chunk_compressed, chunk_uncompressed = (
                        _chunk_byte_estimate(event)
                    )
                    compressed_bytes += chunk_compressed
                    uncompressed_bytes += chunk_uncompressed
                elif etype == "done":
                    total_rows = int(event.get("totalRecords") or 0)
                    used_executor = event.get("executor")
                    break
                elif etype == "error":
                    raise QueryExecutionError(_format_error_event(event))

            download_started = time.monotonic()
            download_tables = [f.result() for f in pending]
        except BaseException:
            _cancel_pending(pending)
            raise
        download_seconds = (
            time.monotonic() - download_started
            if download_started is not None
            else 0.0
        )

        submit_seconds = (
            first_event_at
            if first_event_at is not None
            else (time.monotonic() - submit_started)
        )

        if not download_tables:
            return TransportResult(
                table=empty_table_for(schema),
                truncated=False,
                total_rows=total_rows,
                schema=schema,
                timings=TransportTimings(
                    submit_seconds=submit_seconds,
                    download_seconds=0.0,
                    chunk_count=0,
                    compressed_bytes=0,
                    uncompressed_bytes=0,
                    executor=used_executor,
                ),
            )

        table = pa.concat_tables(download_tables, promote_options="default")
        descriptors = schema if schema else schema_to_descriptors(table.schema)
        return TransportResult(
            table=table,
            truncated=False,
            total_rows=total_rows or table.num_rows,
            schema=descriptors,
            timings=TransportTimings(
                submit_seconds=submit_seconds,
                download_seconds=download_seconds,
                chunk_count=len(download_tables),
                compressed_bytes=compressed_bytes,
                uncompressed_bytes=uncompressed_bytes,
                executor=used_executor,
            ),
        )

    def iter_batches(
        self,
        session_id: str,
        sql: str,
        batch_size: int = 65_536,
        executor: ExecutorChoice = "auto",
        pipeline: PipelineConfig | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Stream the result chunk-by-chunk as :class:`pyarrow.RecordBatch`.

        With a streaming-aware provider, the first batch lands in
        the caller's hands roughly one network round-trip after the
        pod produces its first chunk — no need to wait for the whole
        scan to finish before consuming. Memory stays bounded to
        roughly ``parallel_workers`` chunks in flight at once.
        Yields are in submission order so row order across chunks
        stays stable for downstream code.
        """
        _validate_executor(executor)
        pending: deque[Future[pa.Table]] = deque()
        pool = self._get_download_pool()
        try:
            for event in self._event_stream(
                session_id, sql, executor, pipeline,
            ):
                etype = event.get("type")
                if etype == "chunk":
                    pending.append(self._submit_chunk(pool, event))
                    while pending and pending[0].done():
                        table = pending.popleft().result()
                        yield from table.to_batches(max_chunksize=batch_size)
                elif etype in ("heartbeat", "schema"):
                    continue
                elif etype == "done":
                    break
                elif etype == "error":
                    raise QueryExecutionError(_format_error_event(event))

            while pending:
                table = pending.popleft().result()
                yield from table.to_batches(max_chunksize=batch_size)
        finally:
            _cancel_pending(pending)

    def _event_stream(
        self,
        session_id: str,
        sql: str,
        executor: ExecutorChoice,
        pipeline: PipelineConfig | None,
    ) -> Iterator[dict[str, Any]]:
        """Yield NDJSON events for the statement, owning all responses.

        Three transport shapes converge on a single event sequence:

        1. **Sync NDJSON** — provider streams events; we re-yield until
           ``done`` (or ``error``) and close the response.
        2. **Deferred** — provider emits ``deferred(queryId)`` instead
           of ``done`` when soft-timeout pressure forces it to release
           the request thread. We close the POST response and switch
           to GET ``/iceberg/v1/statements/{queryId}`` polling, then
           re-yield events from the eventual 200 NDJSON response.
        3. **Legacy envelope** — older provider responds with a JSON
           manifest; we synthesise the event sequence in-memory.

        Callers iterate one for-loop and never see an HTTP response
        object — every cleanup path lives in this generator's finally.
        """
        payload = self._build_statement_payload(
            session_id, sql, executor, pipeline,
        )
        post_response = self._http.post(
            path=_STATEMENTS_PATH,
            json_body=payload,
            timeout_seconds=self._config.query_timeout_seconds,
            accept="application/x-ndjson",
            stream=True,
        )
        deferred_query_id: str | None = None
        try:
            if post_response.content_type != "application/x-ndjson":
                envelope_body = post_response.json()
                envelope = self._parse_envelope(envelope_body)
                yield from self._synthesize_events(envelope)
                return

            for event in self._iter_ndjson(post_response):
                if event.get("type") == "deferred":
                    deferred_query_id = self._resolve_deferred_id(event)
                    break
                yield event
            else:
                return
        finally:
            post_response.close()

        if deferred_query_id is not None:
            yield from self._poll_deferred_statement(deferred_query_id)

    def _resolve_deferred_id(self, event: dict[str, Any]) -> str:
        """Extract ``queryId`` from a ``deferred`` event or fail loudly."""
        query_id = event.get("queryId")
        if not isinstance(query_id, str) or not query_id:
            raise TransportError(
                "Provider sent 'deferred' event without a queryId; "
                "cannot resume polling",
            )
        return query_id

    def _poll_deferred_statement(
        self,
        query_id: str,
    ) -> Iterator[dict[str, Any]]:
        """Poll ``GET /iceberg/v1/statements/{queryId}`` until 200.

        The provider returns 202 with a ``Retry-After`` header while
        the pod's manifest has not yet landed in S3, and 200 with the
        same NDJSON wire as :meth:`_event_stream` once it has. We honor
        the server-supplied retry hint when present and fall back to a
        bounded exponential backoff otherwise; the total polling wall
        is capped by ``LakehouseConfig.defer_poll_max_seconds`` so a
        lost queryId never wedges the caller.
        """
        path = f"{_STATEMENTS_PATH}/{query_id}"
        deadline = time.monotonic() + self._config.defer_poll_max_seconds
        delay = _DEFER_POLL_MIN_SECONDS
        while True:
            response = self._http.get(
                path=path,
                timeout_seconds=self._config.query_timeout_seconds,
                accept="application/x-ndjson",
                stream=True,
            )
            try:
                if response.status_code == 200:
                    yield from self._iter_ndjson(response)
                    return
                if response.status_code != 202:
                    raise TransportError(
                        f"Polling {path} returned unexpected status "
                        f"{response.status_code}",
                    )
                wait = _resolve_retry_after(response.headers, delay)
            finally:
                response.close()
            now = time.monotonic()
            if now + wait > deadline:
                raise TransportError(
                    f"Deferred statement {query_id} did not complete "
                    f"within {self._config.defer_poll_max_seconds}s; "
                    f"giving up",
                )
            time.sleep(wait)
            delay = min(delay * 1.5, _DEFER_POLL_MAX_SECONDS)

    def _build_statement_payload(
        self,
        session_id: str,
        sql: str,
        executor: ExecutorChoice,
        pipeline: PipelineConfig | None,
    ) -> dict[str, Any]:
        """Compose the request body for ``POST /v1/statements``."""
        payload: dict[str, Any] = {
            "sessionId": session_id,
            "connectionConfig": {
                "config": {
                    "userName": self._config.username,
                    "password": self._config.password,
                    "warehouseName": self._config.warehouse,
                },
            },
            "query": sql,
            "queryKey": self._compute_query_key(sql, executor, pipeline),
        }
        if executor != "auto":
            payload["executor"] = executor
        if pipeline is not None:
            wire = pipeline.to_wire()
            if wire:
                payload["pipelineConfig"] = wire
        return payload

    def _compute_query_key(
        self,
        sql: str,
        executor: ExecutorChoice,
        pipeline: PipelineConfig | None,
    ) -> str:
        """Return a stable, deterministic hash of the query inputs.

        Same (sql, warehouse, database, user, compute, executor,
        pipeline) → same 32-char hex digest, across SDK versions and
        runs. The server may use this for result caching, request
        deduplication, or observability correlation; it is safe to
        ignore on the server side.

        sessionId is intentionally excluded so repeated runs of an
        identical query against different warm pods share a key.
        """
        parts = [
            sql.strip(),
            self._config.warehouse,
            self._config.database,
            self._config.username,
            self._config.compute_size,
            str(self._config.compute_id or 0),
            str(self._config.minimum_workers),
            executor,
        ]
        if pipeline is not None:
            wire = pipeline.to_wire()
            for key in sorted(wire):
                parts.append(f"{key}={wire[key]}")
        raw = "\x00".join(parts).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    @staticmethod
    def _iter_ndjson(response: Any) -> Iterator[dict[str, Any]]:
        """Yield parsed JSON objects, one per line of the streaming body.

        A malformed NDJSON line means the wire is corrupted (truncated
        frame, partial body, broken upstream). Silently dropping it
        risks losing schema/chunk/done events and leaving the consumer
        wedged or with a partial result. Surface as
        :class:`TransportError` so the caller sees the failure
        immediately rather than getting a mysteriously-truncated table.
        """
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as ex:
                raise TransportError(
                    f"Provider sent malformed NDJSON line "
                    f"(first 120 chars: {line[:120]!r}): {ex}",
                ) from ex
            if isinstance(event, dict):
                yield event

    @staticmethod
    def _synthesize_events(envelope: _Envelope) -> Iterator[dict[str, Any]]:
        """Render a legacy JSON envelope as a sequence of streaming events.

        Lets the consumer code in :meth:`execute` and
        :meth:`iter_batches` treat both wire formats identically:
        one schema, then one chunk per descriptor, then a done.
        """
        yield {"type": "schema", "columns": envelope.schema}
        for chunk in envelope.chunks:
            event = {"type": "chunk"}
            event.update(chunk)
            yield event
        yield {
            "type": "done",
            "totalRecords": envelope.total_rows,
            "hasMore": envelope.truncated,
        }

    def start_session(self) -> str:
        """Create a warm compute session and return its sessionId.

        The session pod boots with the configured ``compute_size``
        (resolved to a concrete ``computeId`` server-side identifier
        on the wire) and remains alive until :meth:`stop_session` is
        called. Spark workers register with the master during the
        provider's wait — by the time this call returns the cluster
        is ready for statement execution.
        """
        compute_id = self._resolve_compute_id()
        payload: dict[str, Any] = {
            "computeId": compute_id,
            "minimumWorkers": self._config.minimum_workers,
            "connectionConfig": {
                "userName": self._config.username,
                "warehouseName": self._config.warehouse,
                "databaseName": self._config.database,
            },
        }
        response = self._http.post(
            path=_START_SESSION_PATH,
            json_body=payload,
            timeout_seconds=self._config.verify_timeout_seconds,
        )
        try:
            body = response.json()
        finally:
            response.close()
        return self._extract_session_id(body)

    def _resolve_compute_id(self) -> int:
        """Resolve the configured t-shirt size to a server compute id.

        ``LakehouseConfig.compute_id`` overrides the t-shirt mapping
        when set explicitly (escape hatch for non-standard ids).
        """
        if self._config.compute_id is not None:
            return self._config.compute_id
        try:
            return _COMPUTE_SIZE_TO_ID[self._config.compute_size]
        except KeyError as ex:
            raise QueryExecutionError(
                f"Unknown compute_size {self._config.compute_size!r}; "
                f"set compute_id explicitly to use a custom value",
            ) from ex

    def stop_session(self, session_id: str) -> None:
        """Tear down the session pod created by :meth:`start_session`."""
        response = self._http.post(
            path=_STOP_SESSION_PATH,
            json_body={"sessionId": session_id},
            timeout_seconds=self._config.verify_timeout_seconds,
        )
        response.close()

    @staticmethod
    def _extract_session_id(body: Any) -> str:
        """Pull the sessionId out of the start-session envelope."""
        if not isinstance(body, dict):
            raise QueryExecutionError(
                "start_session response must be a JSON object"
            )
        response = body.get("response")
        candidate = None
        if isinstance(response, dict):
            candidate = response.get("sessionId") or response.get(
                "sessionIdentifier"
            )
        if not candidate and isinstance(body.get("sessionId"), str):
            candidate = body["sessionId"]
        if not isinstance(candidate, str) or not candidate:
            raise QueryExecutionError(
                "start_session response did not include a sessionId"
            )
        return candidate

    def verify(self) -> None:
        """Verify connectivity and credentials against the provider.

        Uses the existing ``/iceberg/testconnection`` endpoint, which
        is unaffected by the data-plane redesign.
        """
        payload = {
            "connectionConfig": {
                "config": {
                    "userName": self._config.username,
                    "password": self._config.password,
                    "warehouseName": self._config.warehouse,
                    "databaseName": self._config.database,
                },
            },
        }
        response = self._http.post(
            path=_VERIFY_PATH,
            json_body=payload,
            timeout_seconds=self._config.verify_timeout_seconds,
        )
        response.close()

    def close(self) -> None:
        """Release the dedicated S3 session pool and download workers."""
        self._s3_session.close()
        if self._s3_http2_client is not None:
            self._s3_http2_client.close()
            self._s3_http2_client = None
        with self._pool_lock:
            if self._download_pool is not None:
                self._download_pool.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
                self._download_pool = None

    @staticmethod
    def _parse_envelope(envelope: Any) -> _Envelope:
        """Validate the envelope shape and extract the chunk metadata.

        Every field is checked defensively: malformed providers should
        produce typed :class:`QueryExecutionError` exceptions with a
        message that points at the broken field rather than crashing
        with a low-level ``TypeError``.
        """
        if not isinstance(envelope, dict):
            raise QueryExecutionError(
                "Provider envelope must be a JSON object, "
                f"got {type(envelope).__name__}"
            )
        if envelope.get("status") == "error":
            raise QueryExecutionError(
                envelope.get("message", "Unknown query error")
            )
        chunks_raw = envelope.get("chunks")
        if chunks_raw is None:
            chunks_raw = []
        if not isinstance(chunks_raw, list):
            raise QueryExecutionError(
                "Provider envelope 'chunks' must be a list, "
                f"got {type(chunks_raw).__name__}"
            )
        chunks: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks_raw):
            if not isinstance(chunk, dict):
                raise QueryExecutionError(
                    f"Provider chunk[{index}] must be a JSON object, "
                    f"got {type(chunk).__name__}"
                )
            url = chunk.get("url")
            if not isinstance(url, str) or not url:
                raise QueryExecutionError(
                    f"Provider chunk[{index}] missing or empty 'url'"
                )
            if not (url.startswith("http://") or url.startswith("https://")):
                raise QueryExecutionError(
                    f"Provider chunk[{index}] 'url' must be http(s); "
                    f"got {url[:32]!r}"
                )
            chunks.append(chunk)

        schema_raw = envelope.get("schema") or []
        if not isinstance(schema_raw, list):
            raise QueryExecutionError(
                "Provider envelope 'schema' must be a list, "
                f"got {type(schema_raw).__name__}"
            )
        schema: list[dict[str, str]] = []
        for index, descriptor in enumerate(schema_raw):
            if not isinstance(descriptor, dict):
                raise QueryExecutionError(
                    f"Provider schema[{index}] must be a JSON object"
                )
            if "columnName" not in descriptor:
                raise QueryExecutionError(
                    f"Provider schema[{index}] missing 'columnName'"
                )
            schema.append(dict(descriptor))

        return _Envelope(
            chunks=chunks,
            schema=schema,
            total_rows=_safe_int(envelope.get("totalRecords"), default=0),
            truncated=bool(envelope.get("hasMore")),
        )

    def _decode_inline_chunk(self, chunk: dict[str, Any]) -> pa.Table:
        """Parse an inline chunk's base64-encoded Arrow IPC body.

        Phase 3 inline chunks ship the IPC bytes embedded in the
        NDJSON event itself, so the SDK can hand the first rows to
        the caller without an S3 round trip. Pod emits at most one
        inline chunk per query — the first batch flushed by whichever
        encoder wins the race — keeping the wire payload small. All
        subsequent chunks come via presigned URLs as before.
        """
        inline_b64 = chunk.get("inline") or ""
        if not inline_b64:
            raise QueryExecutionError(
                "Inline chunk missing 'inline' payload",
            )
        try:
            body = base64.b64decode(inline_b64)
        except (ValueError, TypeError) as ex:
            raise QueryExecutionError(
                f"Inline chunk has malformed base64 payload: {ex}",
            ) from ex
        try:
            reader = paipc.RecordBatchStreamReader(io.BytesIO(body))
            table = reader.read_all()
        except (pa.ArrowInvalid, pa.ArrowIOError) as ex:
            raise QueryExecutionError(
                f"Inline chunk is not a valid Arrow IPC stream: {ex}",
            ) from ex
        _logger.debug(
            "inline chunk decoded rows=%s bytes=%s",
            table.num_rows,
            len(body),
        )
        return table

    def _download_chunk(self, chunk: dict[str, Any]) -> pa.Table:
        """Fetch a chunk with retries on transient failures.

        Wraps :meth:`_download_chunk_once` in an exponential-backoff
        retry loop. Connection drops, 5xx/408/429 responses, and
        partial-body parse failures all retry. Deterministic 4xx
        (other than 408/429) propagate immediately as
        :class:`QueryExecutionError` since retrying is wasted effort.
        """
        url_preview = _truncate_url(chunk["url"])
        max_attempts = self._config.max_retries + 1
        last_cause: BaseException | None = None
        for attempt in range(max_attempts):
            try:
                return self._download_chunk_once(chunk)
            except _RetryableChunkError as ex:
                last_cause = ex.cause or ex
                if attempt + 1 >= max_attempts:
                    _logger.warning(
                        "chunk %s exhausted retries (%d attempts): %s",
                        url_preview,
                        max_attempts,
                        ex,
                    )
                    break
                delay = self._compute_chunk_backoff(attempt)
                _logger.warning(
                    "chunk %s attempt %d/%d failed, retrying in %.2fs: %s",
                    url_preview,
                    attempt + 1,
                    max_attempts,
                    delay,
                    ex,
                )
                time.sleep(delay)
        raise TransportError(
            f"Cannot download result chunk {url_preview} after "
            f"{max_attempts} attempt(s): {last_cause}",
        ) from last_cause

    def _download_chunk_once(self, chunk: dict[str, Any]) -> pa.Table:
        """Single attempt at fetching and parsing an Arrow IPC chunk.

        Translates transient failures into :class:`_RetryableChunkError`
        for the retry wrapper. Deterministic failures (auth, malformed
        URL) propagate as :class:`QueryExecutionError`. Uses the
        HTTP/2 client when ``enable_http2`` is set; otherwise falls
        back to the ``requests``-backed pool.
        """
        if self._s3_http2_client is not None:
            return self._download_chunk_once_http2(chunk)
        return self._download_chunk_once_requests(chunk)

    def _download_chunk_once_requests(
        self,
        chunk: dict[str, Any],
    ) -> pa.Table:
        """Fetch a chunk over the requests-backed (HTTP/1.1) S3 pool."""
        url = chunk["url"]
        url_preview = _truncate_url(url)
        started = time.monotonic()
        try:
            with self._s3_session.get(
                url,
                stream=True,
                timeout=(
                    self._config.connect_timeout_seconds,
                    self._config.query_timeout_seconds,
                ),
            ) as resp:
                if resp.status_code != 200:
                    body_preview = resp.text[:256] if resp.content else ""
                    if resp.status_code in _S3_RETRYABLE_STATUS:
                        raise _RetryableChunkError(
                            f"HTTP {resp.status_code} from S3 "
                            f"({url_preview}): {body_preview}",
                        )
                    raise QueryExecutionError(
                        f"Failed to fetch result chunk {url_preview} "
                        f"(HTTP {resp.status_code}): {body_preview}",
                    )
                resp.raw.decode_content = True
                try:
                    reader = paipc.RecordBatchStreamReader(resp.raw)
                    table = reader.read_all()
                except (pa.ArrowInvalid, pa.ArrowIOError) as ex:
                    raise _RetryableChunkError(
                        f"chunk {url_preview} parse failed "
                        f"(likely truncated body): {ex}",
                        cause=ex,
                    ) from ex
        except requests.RequestException as ex:
            raise _RetryableChunkError(
                f"transport error fetching chunk {url_preview}: {ex}",
                cause=ex,
            ) from ex

        elapsed_ms = int((time.monotonic() - started) * 1000)
        _logger.debug(
            "chunk done url=%s rows=%s elapsed_ms=%s",
            url_preview,
            table.num_rows,
            elapsed_ms,
        )
        return table

    def _download_chunk_once_http2(
        self,
        chunk: dict[str, Any],
    ) -> pa.Table:
        """Fetch a chunk over the httpx HTTP/2 client, streaming.

        Streams the body from the wire into a pyarrow IPC reader via
        an :class:`_IterBytesReader` adapter so peak memory per chunk
        stays at the IPC reader's internal buffer plus the in-flight
        TCP frame, instead of double-buffering the full chunk in
        ``response.content`` *and* a ``BytesIO``. With 32 workers ×
        64 MB chunks the old path peaked at ~2 GB; this one peaks
        only at the IPC reader's working set.
        """
        assert self._s3_http2_client is not None
        url = chunk["url"]
        url_preview = _truncate_url(url)
        started = time.monotonic()
        try:
            with self._s3_http2_client.stream("GET", url) as response:
                if response.status_code != 200:
                    response.read()
                    body_preview = (
                        response.text[:256] if response.content else ""
                    )
                    if response.status_code in _S3_RETRYABLE_STATUS:
                        raise _RetryableChunkError(
                            f"HTTP {response.status_code} from S3 "
                            f"({url_preview}): {body_preview}",
                        )
                    raise QueryExecutionError(
                        f"Failed to fetch result chunk {url_preview} "
                        f"(HTTP {response.status_code}): {body_preview}",
                    )
                try:
                    reader = paipc.RecordBatchStreamReader(
                        _IterBytesReader(response.iter_bytes()),
                    )
                    table = reader.read_all()
                except (pa.ArrowInvalid, pa.ArrowIOError) as ex:
                    raise _RetryableChunkError(
                        f"chunk {url_preview} parse failed "
                        f"(likely truncated body): {ex}",
                        cause=ex,
                    ) from ex
        except httpx.HTTPError as ex:
            raise _RetryableChunkError(
                f"transport error fetching chunk {url_preview}: {ex}",
                cause=ex,
            ) from ex

        elapsed_ms = int((time.monotonic() - started) * 1000)
        _logger.debug(
            "h2 chunk done url=%s rows=%s elapsed_ms=%s",
            url_preview,
            table.num_rows,
            elapsed_ms,
        )
        return table

    def _compute_chunk_backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped at a sane ceiling.

        Avoids retry storms when many concurrent chunks fail in
        lockstep: each worker draws independent jitter so they spread
        out across the backoff window.
        """
        base = max(self._config.backoff_seconds, 0.05)
        delay = base * (2**attempt)
        jitter = random.uniform(0.0, base)
        return min(delay + jitter, _CHUNK_BACKOFF_CEILING_SECONDS)

    def _build_s3_session(self) -> requests.Session:
        """Build the S3-side session with retries and connection pool.

        urllib3 handles connect/read retries here. Status-code retries
        are deliberately disabled because the app-level retry around
        :meth:`_download_chunk_once` covers them; doubling the retry
        layers would produce ``max_retries^2`` worst-case round trips.
        """
        retry = Retry(
            total=self._config.max_retries,
            connect=self._config.max_retries,
            read=self._config.max_retries,
            status=0,
            backoff_factor=self._config.backoff_seconds,
            status_forcelist=tuple(_S3_RETRYABLE_STATUS),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=self._config.pool_connections,
            pool_maxsize=max(
                self._config.pool_maxsize,
                self._config.parallel_workers,
            ),
        )
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _build_s3_http2_client(self) -> httpx.Client:
        """Build the httpx HTTP/2 client used for chunk downloads.

        Uses an explicit :class:`httpx.HTTPTransport` so the h2 client
        gets the same connect-retry safety net as the requests-backed
        path: a bare TCP RST mid-handshake retries at the transport
        layer rather than failing fast and depending on the
        application-level retry to compensate. Application-level retry
        for status codes (408/429/5xx) and partial-body parse failures
        still lives in :meth:`_download_chunk`.
        """
        timeout = httpx.Timeout(
            connect=float(self._config.connect_timeout_seconds),
            read=float(self._config.query_timeout_seconds),
            write=float(self._config.connect_timeout_seconds),
            pool=float(self._config.connect_timeout_seconds),
        )
        limits = httpx.Limits(
            max_connections=max(
                self._config.pool_maxsize,
                self._config.parallel_workers,
            ),
            max_keepalive_connections=self._config.pool_maxsize,
        )
        transport = httpx.HTTPTransport(
            http2=True,
            retries=self._config.max_retries,
        )
        return httpx.Client(
            timeout=timeout,
            limits=limits,
            transport=transport,
            follow_redirects=False,
        )


def _cancel_pending(pending: Iterator[Future[pa.Table]]) -> None:
    """Cancel any not-yet-running futures left behind by a failed call."""
    for f in pending:
        f.cancel()


def _chunk_byte_estimate(event: dict[str, Any]) -> tuple[int, int]:
    """Return ``(compressed, uncompressed)`` byte counts for an event.

    For S3-presigned chunks the pod fills both ``compressedSize`` and
    ``uncompressedSize`` directly. For inline chunks the pod has been
    seen to omit one or both — the IPC bytes ride embedded in the
    NDJSON event itself, so the wire size is implicit. To keep
    :class:`TransportTimings` accurate (compressed throughput
    dashboards underreported by the inline payload) the SDK estimates
    the compressed size from the base64 payload length when the
    server omits it: a base64 string of length L decodes to roughly
    ``(L * 3) // 4`` bytes minus padding characters.
    """
    declared_compressed = int(event.get("compressedSize") or 0)
    declared_uncompressed = int(event.get("uncompressedSize") or 0)
    if "inline" in event and not declared_compressed:
        b64 = event.get("inline") or ""
        declared_compressed = (len(b64) * 3) // 4 - b64.count("=")
    return (declared_compressed, declared_uncompressed)


def _format_error_event(event: dict[str, Any]) -> str:
    """Render an NDJSON ``error`` event as a single-line string."""
    return (
        f"{event.get('errorType') or 'Unknown'}: "
        f"{event.get('message') or 'no message'}"
    )


def _validate_executor(executor: str) -> None:
    """Reject anything outside ``ExecutorChoice``.

    Caught in the transport before the request goes out so the user
    sees the typo (``"fastpath"`` instead of ``"fast"``) immediately
    rather than as a server-side rejection seconds later.
    """
    if executor not in _VALID_EXECUTORS:
        raise QueryExecutionError(
            f"executor must be one of {_VALID_EXECUTORS!r}; got {executor!r}",
        )


def _safe_int(value: Any, default: int) -> int:
    """Coerce ``value`` to an int, returning ``default`` on any failure.

    Used for envelope fields the provider may send as strings, floats,
    or ``None``. A malformed value never crashes the SDK — it just
    falls back to the provided default.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate_url(url: str, head: int = 80, tail: int = 16) -> str:
    """Return a shortened URL suitable for log lines and error messages.

    Presigned URLs contain a long signature query string that adds
    nothing to a log message but a lot of noise. Showing the host +
    path prefix and the trailing characters is enough to identify the
    chunk while keeping logs scannable.
    """
    if len(url) <= head + tail + 3:
        return url
    return f"{url[:head]}...{url[-tail:]}"


def _resolve_retry_after(
    headers: Any,
    fallback: float,
) -> float:
    """Honor a server-supplied ``Retry-After`` header, clamped to bounds.

    Accepts the seconds-only form (RFC 7231); the HTTP-date form is
    rare in machine-to-machine APIs and would only add complexity for
    no measurable benefit here. Falls back to the caller-supplied
    backoff when the header is absent or unparseable.
    """
    def _clamp(value: float) -> float:
        return min(
            max(value, _DEFER_POLL_MIN_SECONDS),
            _DEFER_POLL_MAX_SECONDS,
        )

    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if raw is None:
        return _clamp(fallback)
    try:
        return _clamp(float(raw))
    except (TypeError, ValueError):
        return _clamp(fallback)
