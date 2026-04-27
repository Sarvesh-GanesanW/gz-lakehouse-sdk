"""Tests for the Snowflake-style :class:`gz_lakehouse._transport.Transport`."""

from __future__ import annotations

import io
import json as _json

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
import responses

from gz_lakehouse import LakehouseConfig
from gz_lakehouse._http import HttpClient
from gz_lakehouse._transport import Transport
from gz_lakehouse.exceptions import QueryExecutionError

PROVIDER_URL = "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
S3_URL_TEMPLATE = "https://s3.example.com/chunk-{}.arrow"


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
def test_execute_downloads_arrow_chunks_in_parallel() -> None:
    """All presigned chunks are fetched and concatenated in submission order."""
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

    outcome = transport.execute("SELECT * FROM customers")

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

    outcome = transport.execute("SELECT * FROM empty")

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
    for batch in transport.iter_batches("SELECT id FROM customers"):
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
        transport.execute("SELECT 1")
    transport.close()
    http.close()


@responses.activate
def test_envelope_status_error_raises() -> None:
    """An ``error`` envelope status surfaces as :class:`QueryExecutionError`."""
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
        transport.execute("SELECT BAD FROM")
    assert "syntax error" in str(exc_info.value)
    transport.close()
    http.close()


@responses.activate
def test_envelope_with_non_http_chunk_url_raises() -> None:
    """Chunk URLs must be http(s) — anything else is rejected."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [],
            "chunks": [{"url": "ftp://example.com/chunk-0.arrow"}],
        },
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    with pytest.raises(QueryExecutionError) as exc_info:
        transport.execute("SELECT 1")
    assert "http" in str(exc_info.value).lower()
    transport.close()
    http.close()


@responses.activate
def test_envelope_with_non_dict_chunk_raises() -> None:
    """Each chunk descriptor must be a JSON object, not a bare string."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"schema": [], "chunks": ["https://s3/x.arrow"]},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    with pytest.raises(QueryExecutionError):
        transport.execute("SELECT 1")
    transport.close()
    http.close()


@responses.activate
def test_envelope_handles_string_total_records() -> None:
    """``totalRecords`` returned as a string still parses cleanly."""
    table = pa.table({"id": pa.array([1, 2], type=pa.int64())})
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [{"columnName": "id", "dataType": "BIGINT"}],
            "totalRecords": "2",
            "hasMore": False,
            "chunks": [{"url": S3_URL_TEMPLATE.format(0)}],
        },
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

    outcome = transport.execute("SELECT 1")

    assert outcome.total_rows == 2
    transport.close()
    http.close()


@responses.activate
def test_chunk_with_invalid_arrow_bytes_raises() -> None:
    """Garbage bytes from S3 surface as :class:`QueryExecutionError`."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [],
            "chunks": [{"url": S3_URL_TEMPLATE.format(0)}],
        },
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body=b"not-arrow-ipc",
        status=200,
    )

    config = _config(max_retries=0)
    http = _http(config)
    transport = Transport(http=http, config=config)

    with pytest.raises(QueryExecutionError) as exc_info:
        transport.execute("SELECT 1")
    assert "Arrow IPC" in str(exc_info.value)
    transport.close()
    http.close()


@responses.activate
def test_submit_payload_uses_compute_size_default() -> None:
    """Default config sends ``computeSize`` and omits raw ``computeId``."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"schema": [], "totalRecords": 0, "chunks": []},
        status=200,
    )

    config = _config()
    http = _http(config)
    transport = Transport(http=http, config=config)

    transport.execute("SELECT 1")

    submit_call = responses.calls[0]
    body = submit_call.request.body
    if isinstance(body, bytes):
        body = body.decode()
    payload = _json.loads(body)
    assert payload["computeSize"] == "small"
    assert "computeId" not in payload
    transport.close()
    http.close()


@responses.activate
def test_submit_payload_includes_compute_id_when_set() -> None:
    """Explicit ``compute_id`` is sent alongside ``computeSize``."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"schema": [], "totalRecords": 0, "chunks": []},
        status=200,
    )

    config = _config(compute_id=1012)
    http = _http(config)
    transport = Transport(http=http, config=config)

    transport.execute("SELECT 1")

    submit_call = responses.calls[0]
    body = submit_call.request.body
    if isinstance(body, bytes):
        body = body.decode()
    payload = _json.loads(body)
    assert payload["computeId"] == 1012
    assert payload["computeSize"] == "small"
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
        transport.execute("SELECT 1")
    transport.close()
    http.close()
