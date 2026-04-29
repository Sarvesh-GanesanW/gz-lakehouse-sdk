"""Tests for the session-aware :class:`gz_lakehouse._transport.Transport`."""

from __future__ import annotations

import io
import json as _json

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
import responses

from gz_lakehouse import LakehouseConfig
from gz_lakehouse._http import HttpClient
from gz_lakehouse._transport import Transport, _IterBytesReader
from gz_lakehouse.exceptions import QueryExecutionError

PROVIDER_URL = "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
S3_URL_TEMPLATE = "https://s3.example.com/chunk-{}.arrow"
SESSION_ID = "session-abc-123"


def _config(**overrides: object) -> LakehouseConfig:
    """Default config used by the transport tests."""
    base = {
        "lakehouse_url": PROVIDER_URL,
        "warehouse": "wh",
        "database": "db",
        "username": "alice",
        "password": "secret",
        "parallel_workers": 4,
    }
    base.update(overrides)
    return LakehouseConfig(**base)


def _http(config: LakehouseConfig) -> HttpClient:
    """Build the HTTP client used by the transport."""
    return HttpClient(
        base_url=config.lakehouse_url,
        site=config.derived_site,
        connect_timeout_seconds=config.connect_timeout_seconds,
        max_retries=0,
        backoff_seconds=0,
        pool_connections=config.pool_connections,
        pool_maxsize=config.pool_maxsize,
        enable_compression=config.enable_compression,
    )


def _arrow_ipc_bytes(table: pa.Table) -> bytes:
    """Serialise an Arrow table as an IPC stream."""
    sink = io.BytesIO()
    with paipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _envelope(chunk_count: int, total_rows: int) -> dict:
    """Build a representative provider envelope."""
    return {
        "schema": [
            {"columnName": "id", "dataType": "BIGINT"},
            {"columnName": "name", "dataType": "VARCHAR"},
        ],
        "totalRecords": total_rows,
        "hasMore": False,
        "chunks": [
            {
                "url": S3_URL_TEMPLATE.format(i),
                "rowCount": total_rows // chunk_count,
                "compressedSize": 1234,
                "uncompressedSize": 5678,
            }
            for i in range(chunk_count)
        ],
    }


@responses.activate
def test_start_session_returns_session_id() -> None:
    """``start_session`` extracts the sessionId from the response."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/startsession",
        json={"status": 200, "response": {"sessionId": SESSION_ID}},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    assert transport.start_session() == SESSION_ID
    transport.close()
    http.close()


@responses.activate
def test_start_session_accepts_session_identifier_alias() -> None:
    """Provider-side payloads using ``sessionIdentifier`` also resolve."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/startsession",
        json={
            "status": 200,
            "response": {"sessionIdentifier": SESSION_ID},
        },
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    assert transport.start_session() == SESSION_ID
    transport.close()
    http.close()


@responses.activate
def test_start_session_missing_id_raises() -> None:
    """Missing sessionId in the response is a typed error."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/startsession",
        json={"status": 200, "response": {}},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    with pytest.raises(QueryExecutionError):
        transport.start_session()
    transport.close()
    http.close()


@responses.activate
def test_stop_session_posts_payload() -> None:
    """``stop_session`` POSTs ``{sessionId}`` to the provider."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/stopsession",
        json={"status": 200, "response": {"message": "ok"}},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    transport.stop_session(SESSION_ID)

    assert len(responses.calls) == 1
    body = _json.loads(responses.calls[0].request.body)
    assert body == {"sessionId": SESSION_ID}
    transport.close()
    http.close()


@responses.activate
def test_execute_downloads_arrow_chunks_in_parallel() -> None:
    """All chunks are fetched and concatenated in submission order."""
    chunks = [
        pa.table({"id": [1, 2], "name": ["a", "b"]}),
        pa.table({"id": [3, 4], "name": ["c", "d"]}),
        pa.table({"id": [5, 6], "name": ["e", "f"]}),
    ]
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json=_envelope(chunk_count=3, total_rows=6),
        status=200,
    )
    for index, chunk in enumerate(chunks):
        responses.add(
            responses.GET,
            S3_URL_TEMPLATE.format(index),
            body=_arrow_ipc_bytes(chunk),
            status=200,
            content_type="application/vnd.apache.arrow.stream",
        )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    outcome = transport.execute(SESSION_ID, "SELECT * FROM customers")

    assert outcome.table.num_rows == 6
    assert outcome.table.column("id").to_pylist() == [1, 2, 3, 4, 5, 6]
    assert outcome.total_rows == 6
    transport.close()
    http.close()


@responses.activate
def test_execute_returns_empty_table_with_schema() -> None:
    """A zero-chunk response still preserves the schema for downstream use."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [
                {"columnName": "id", "dataType": "BIGINT"},
                {"columnName": "name", "dataType": "VARCHAR"},
            ],
            "totalRecords": 0,
            "hasMore": False,
            "chunks": [],
        },
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    outcome = transport.execute(SESSION_ID, "SELECT * FROM empty")

    assert outcome.table.num_rows == 0
    assert outcome.table.column_names == ["id", "name"]
    assert outcome.table.schema.field("id").type == pa.int64()
    transport.close()
    http.close()


@responses.activate
def test_iter_batches_streams_chunks() -> None:
    """``iter_batches`` yields each chunk's batches in submission order."""
    chunks = [
        pa.table({"id": pa.array([1, 2, 3], type=pa.int64())}),
        pa.table({"id": pa.array([4, 5, 6], type=pa.int64())}),
    ]
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [{"columnName": "id", "dataType": "BIGINT"}],
            "totalRecords": 6,
            "hasMore": False,
            "chunks": [{"url": S3_URL_TEMPLATE.format(i)} for i in range(2)],
        },
        status=200,
    )
    for index, chunk in enumerate(chunks):
        responses.add(
            responses.GET,
            S3_URL_TEMPLATE.format(index),
            body=_arrow_ipc_bytes(chunk),
            status=200,
        )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    rows: list[int] = []
    sql = "SELECT id FROM customers"
    for batch in transport.iter_batches(SESSION_ID, sql):
        rows.extend(batch.column("id").to_pylist())

    assert rows == [1, 2, 3, 4, 5, 6]
    transport.close()
    http.close()


@responses.activate
def test_envelope_with_invalid_chunks_raises() -> None:
    """Malformed chunk descriptors surface as :class:`QueryExecutionError`."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"chunks": [{"rowCount": 10}]},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    with pytest.raises(QueryExecutionError):
        transport.execute(SESSION_ID, "SELECT 1")
    transport.close()
    http.close()


@responses.activate
def test_envelope_status_error_raises() -> None:
    """An ``error`` envelope surfaces as :class:`QueryExecutionError`."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"status": "error", "message": "syntax error near 'BAD'"},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    with pytest.raises(QueryExecutionError) as exc_info:
        transport.execute(SESSION_ID, "SELECT BAD FROM")
    assert "syntax error" in str(exc_info.value)
    transport.close()
    http.close()


@responses.activate
def test_submit_payload_includes_session_id() -> None:
    """The execute call sends ``sessionId`` in the body."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"schema": [], "totalRecords": 0, "chunks": []},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    transport.execute(SESSION_ID, "SELECT 1")

    body = _json.loads(responses.calls[0].request.body)
    assert body["sessionId"] == SESSION_ID
    assert body["query"] == "SELECT 1"
    assert "targetChunks" not in body
    transport.close()
    http.close()


@responses.activate
def test_chunk_download_failure_surfaces() -> None:
    """A non-200 from the presigned URL surfaces as a typed exception."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [],
            "chunks": [{"url": S3_URL_TEMPLATE.format(0)}],
            "totalRecords": 1,
        },
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body="access denied",
        status=403,
    )

    config = _config(max_retries=0)
    http = _http(config)
    transport = Transport(http=http, config=config)

    with pytest.raises(QueryExecutionError):
        transport.execute(SESSION_ID, "SELECT 1")
    transport.close()
    http.close()


@responses.activate
def test_chunk_download_retries_on_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 from S3 retries until success without restarting the query."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    table = pa.table({"id": pa.array([1, 2], type=pa.int64())})
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [],
            "chunks": [{"url": S3_URL_TEMPLATE.format(0)}],
            "totalRecords": 2,
        },
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body="slow down",
        status=503,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body=_arrow_ipc_bytes(table),
        status=200,
    )

    config = _config(max_retries=2, backoff_seconds=0.0)
    http = _http(config)
    transport = Transport(http=http, config=config)
    try:
        outcome = transport.execute(SESSION_ID, "SELECT 1")
        assert outcome.table.num_rows == 2
    finally:
        transport.close()
        http.close()


@responses.activate
def test_chunk_download_no_retry_on_403() -> None:
    """A 403 is deterministic and skips the retry loop entirely."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [],
            "chunks": [{"url": S3_URL_TEMPLATE.format(0)}],
            "totalRecords": 1,
        },
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body="forbidden",
        status=403,
    )

    config = _config(max_retries=4)
    http = _http(config)
    transport = Transport(http=http, config=config)
    try:
        with pytest.raises(QueryExecutionError):
            transport.execute(SESSION_ID, "SELECT 1")
    finally:
        transport.close()
        http.close()
    s3_calls = [c for c in responses.calls if c.request.method == "GET"]
    assert len(s3_calls) == 1, (
        f"403 must not retry; saw {len(s3_calls)} attempts"
    )


@responses.activate
def test_query_key_in_payload_is_stable() -> None:
    """Identical queries produce identical queryKey hashes."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"schema": [], "totalRecords": 0, "chunks": []},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"schema": [], "totalRecords": 0, "chunks": []},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)
    try:
        transport.execute(SESSION_ID, "SELECT * FROM t")
        transport.execute(SESSION_ID, "SELECT * FROM t")
    finally:
        transport.close()
        http.close()

    bodies = [_json.loads(c.request.body) for c in responses.calls]
    assert bodies[0]["queryKey"] == bodies[1]["queryKey"]
    assert len(bodies[0]["queryKey"]) == 32


@responses.activate
def test_query_key_changes_with_sql() -> None:
    """Different SQL → different queryKey."""
    for _ in range(2):
        responses.add(
            responses.POST,
            f"{PROVIDER_URL}/iceberg/v1/statements",
            json={"schema": [], "totalRecords": 0, "chunks": []},
            status=200,
        )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)
    try:
        transport.execute(SESSION_ID, "SELECT * FROM a")
        transport.execute(SESSION_ID, "SELECT * FROM b")
    finally:
        transport.close()
        http.close()

    bodies = [_json.loads(c.request.body) for c in responses.calls]
    assert bodies[0]["queryKey"] != bodies[1]["queryKey"]


@responses.activate
def test_download_pool_is_shared_across_executions() -> None:
    """The chunk-download pool is created once and reused across calls."""
    table = pa.table({"id": pa.array([1], type=pa.int64())})
    for _ in range(2):
        responses.add(
            responses.POST,
            f"{PROVIDER_URL}/iceberg/v1/statements",
            json=_envelope(chunk_count=1, total_rows=1),
            status=200,
        )
        responses.add(
            responses.GET,
            S3_URL_TEMPLATE.format(0),
            body=_arrow_ipc_bytes(table),
            status=200,
        )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)
    try:
        assert transport._download_pool is None
        transport.execute(SESSION_ID, "SELECT 1")
        first_pool = transport._download_pool
        assert first_pool is not None
        transport.execute(SESSION_ID, "SELECT 2")
        assert transport._download_pool is first_pool
    finally:
        transport.close()
        http.close()
    assert transport._download_pool is None


@responses.activate
def test_close_shuts_down_download_pool() -> None:
    """Closing the transport tears down the shared download pool."""
    table = pa.table({"id": pa.array([1], type=pa.int64())})
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json=_envelope(chunk_count=1, total_rows=1),
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body=_arrow_ipc_bytes(table),
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)
    transport.execute(SESSION_ID, "SELECT 1")
    pool = transport._download_pool
    assert pool is not None
    transport.close()
    http.close()
    assert transport._download_pool is None
    assert pool._shutdown is True


@responses.activate
def test_heartbeat_does_not_inflate_ttfb() -> None:
    """A leading heartbeat event must not be counted as TTFB.

    The pod can emit periodic ``heartbeat`` events on the NDJSON stream
    so the TCP connection stays alive during long-running queries. The
    SDK must skip them when stamping the TTFB metric, otherwise a slow
    schema-emit followed by a fast first heartbeat would report a TTFB
    of "time to first heartbeat" instead of "time to first useful row
    of metadata."
    """
    table = pa.table({"id": pa.array([1], type=pa.int64())})
    schema_event = (
        b'{"type":"schema","columns":'
        b'[{"columnName":"id","dataType":"BIGINT"}]}'
    )
    chunk_event = (
        b'{"type":"chunk","url":"' + S3_URL_TEMPLATE.format(0).encode() + b'"}'
    )
    done_event = b'{"type":"done","totalRecords":1}'
    body = (
        b'{"type":"heartbeat"}\n'
        + schema_event
        + b"\n"
        + chunk_event
        + b"\n"
        + done_event
        + b"\n"
    )
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        body=body,
        status=200,
        content_type="application/x-ndjson",
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body=_arrow_ipc_bytes(table),
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)
    try:
        outcome = transport.execute(SESSION_ID, "SELECT 1")
    finally:
        transport.close()
        http.close()

    assert outcome.timings.submit_seconds >= 0.0
    assert outcome.table.num_rows == 1


def test_http2_client_uses_transport_with_retries() -> None:
    """When ``enable_http2`` is on, the httpx client gets connect retries."""
    config = _config(enable_http2=True, max_retries=4)
    http = _http(config)
    transport = Transport(http=http, config=config)
    try:
        client = transport._s3_http2_client
        assert client is not None
        underlying = client._transport
        assert underlying is not None
    finally:
        transport.close()
        http.close()


def test_iter_bytes_reader_assembles_small_reads_across_chunks() -> None:
    """Small reads pull exact N bytes even when frames straddle them."""
    reader = _IterBytesReader(iter([b"hello", b" ", b"world"]))

    buf = bytearray(5)
    assert reader.readinto(buf) == 5
    assert bytes(buf) == b"hello"

    buf = bytearray(6)
    assert reader.readinto(buf) == 6
    assert bytes(buf) == b" world"


def test_iter_bytes_reader_returns_zero_on_eof() -> None:
    """Once the iterator is drained, readinto returns 0."""
    reader = _IterBytesReader(iter([b"abc"]))

    buf = bytearray(10)
    assert reader.readinto(buf) == 3
    assert bytes(buf[:3]) == b"abc"
    assert reader.readinto(buf) == 0


def test_iter_bytes_reader_parses_arrow_ipc_stream() -> None:
    """A frame-fragmented IPC stream still parses to the same Arrow table."""
    table = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
    body = _arrow_ipc_bytes(table)
    fragments = [body[i : i + 7] for i in range(0, len(body), 7)]

    reader = _IterBytesReader(iter(fragments))
    parsed = paipc.RecordBatchStreamReader(reader).read_all()

    assert parsed.column("id").to_pylist() == [1, 2, 3]
