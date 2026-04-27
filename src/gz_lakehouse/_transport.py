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

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from gz_lakehouse.config import LakehouseConfig

_STATEMENTS_PATH = "/iceberg/v1/statements"
_START_SESSION_PATH = "/iceberg/startsession"
_STOP_SESSION_PATH = "/iceberg/stopsession"
_VERIFY_PATH = "/iceberg/testconnection"

_DOWNLOAD_BUFFER_BYTES = 1 << 20

_S3_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

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

    Useful for benchmarks and ops dashboards: the three numbers add up
    to the total query latency and split cleanly into the two phases
    that dominate it (server-side compute + client-side download).
    """

    def __init__(
        self,
        submit_seconds: float,
        download_seconds: float,
        chunk_count: int,
        compressed_bytes: int,
        uncompressed_bytes: int,
    ) -> None:
        """Hold the timing and size breakdown for a single execution."""
        self.submit_seconds = submit_seconds
        self.download_seconds = download_seconds
        self.chunk_count = chunk_count
        self.compressed_bytes = compressed_bytes
        self.uncompressed_bytes = uncompressed_bytes

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

    def execute(
        self,
        session_id: str,
        sql: str,
    ) -> TransportResult:
        """Submit ``sql`` on ``session_id`` and materialise the result.

        The session must already be alive (created via
        :meth:`start_session`). The provider forwards the statement to
        the session pod's warm Spark cluster, then returns presigned
        chunk URLs which we download in parallel. Chunk count is
        determined server-side from result size — there is no
        per-call chunking knob.
        """
        submit_started = time.monotonic()
        envelope = self._submit(session_id, sql)
        submit_seconds = time.monotonic() - submit_started

        compressed_bytes = sum(
            int(c.get("compressedSize") or 0) for c in envelope.chunks
        )
        uncompressed_bytes = sum(
            int(c.get("uncompressedSize") or 0) for c in envelope.chunks
        )

        if not envelope.chunks:
            return TransportResult(
                table=empty_table_for(envelope.schema),
                truncated=envelope.truncated,
                total_rows=envelope.total_rows,
                schema=envelope.schema,
                timings=TransportTimings(
                    submit_seconds=submit_seconds,
                    download_seconds=0.0,
                    chunk_count=0,
                    compressed_bytes=0,
                    uncompressed_bytes=0,
                ),
            )

        workers = self._workers_for(envelope.chunks)
        _logger.info(
            "Downloading %s chunks with %s workers",
            len(envelope.chunks),
            workers,
        )
        download_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tables = list(pool.map(self._download_chunk, envelope.chunks))
        table = pa.concat_tables(tables, promote_options="default")
        download_seconds = time.monotonic() - download_started

        descriptors = (
            envelope.schema
            if envelope.schema
            else schema_to_descriptors(table.schema)
        )
        total_rows = (
            envelope.total_rows if envelope.total_rows else table.num_rows
        )
        return TransportResult(
            table=table,
            truncated=envelope.truncated,
            total_rows=total_rows,
            schema=descriptors,
            timings=TransportTimings(
                submit_seconds=submit_seconds,
                download_seconds=download_seconds,
                chunk_count=len(envelope.chunks),
                compressed_bytes=compressed_bytes,
                uncompressed_bytes=uncompressed_bytes,
            ),
        )

    def iter_batches(
        self,
        session_id: str,
        sql: str,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Stream the result chunk-by-chunk as :class:`pyarrow.RecordBatch`.

        Chunk downloads run on a thread pool with up to
        ``parallel_workers`` in flight; results are yielded in
        submission order so the caller always sees a stable row order.
        Memory stays bounded to roughly ``parallel_workers`` chunks at
        any moment.
        """
        envelope = self._submit(session_id, sql)
        if not envelope.chunks:
            return

        workers = self._workers_for(envelope.chunks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._download_chunk, chunk)
                for chunk in envelope.chunks
            ]
            for future in futures:
                table = future.result()
                yield from table.to_batches(max_chunksize=batch_size)

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
            "minimumWorkers": 1,
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
        """Release the dedicated S3 session pool."""
        self._s3_session.close()

    def _submit(
        self,
        session_id: str,
        sql: str,
    ) -> _Envelope:
        """POST the SQL on ``session_id`` and parse the chunk envelope."""
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
        }
        response = self._http.post(
            path=_STATEMENTS_PATH,
            json_body=payload,
            timeout_seconds=self._config.query_timeout_seconds,
        )
        try:
            envelope = response.json()
        finally:
            response.close()
        return self._parse_envelope(envelope)

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

    def _workers_for(self, chunks: list[dict[str, Any]]) -> int:
        """Cap the worker count at ``min(parallel_workers, chunk_count)``."""
        return max(1, min(self._config.parallel_workers, len(chunks)))

    def _download_chunk(self, chunk: dict[str, Any]) -> pa.Table:
        """Fetch a single presigned-URL chunk and parse its Arrow IPC stream.

        Every error is rewrapped with the chunk URL (truncated for log
        hygiene) so downstream stack traces always identify the
        offending chunk.
        """
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
                    raise QueryExecutionError(
                        f"Failed to fetch result chunk {url_preview} "
                        f"(HTTP {resp.status_code}): {body_preview}"
                    )
                resp.raw.decode_content = True
                try:
                    reader = paipc.RecordBatchStreamReader(resp.raw)
                    table = reader.read_all()
                except (pa.ArrowInvalid, pa.ArrowIOError) as ex:
                    raise QueryExecutionError(
                        f"Result chunk {url_preview} is not a valid "
                        f"Arrow IPC stream: {ex}"
                    ) from ex
        except requests.RequestException as ex:
            raise TransportError(
                f"Cannot download result chunk {url_preview}: {ex}"
            ) from ex

        elapsed_ms = int((time.monotonic() - started) * 1000)
        _logger.debug(
            "chunk done url=%s rows=%s elapsed_ms=%s",
            url_preview,
            table.num_rows,
            elapsed_ms,
        )
        return table

    def _build_s3_session(self) -> requests.Session:
        """Build the S3-side session with retries and connection pool."""
        retry = Retry(
            total=self._config.max_retries,
            connect=self._config.max_retries,
            read=self._config.max_retries,
            status=self._config.max_retries,
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
