"""Tests for :class:`LakehouseClient` against a mocked provider."""

import io

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
import responses

from gz_lakehouse import (
    AuthenticationError,
    LakehouseClient,
    LakehouseConfig,
    QueryError,
)

PROVIDER_URL = "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
S3_URL_TEMPLATE = "https://s3.example.com/chunk-{}.arrow"


def _config(**overrides: object) -> LakehouseConfig:
    """Build a minimal config pointing at the mock provider URL."""
    base = {
        "lakehouse_url": PROVIDER_URL,
        "warehouse": "wh",
        "database": "db",
        "username": "alice",
        "password": "secret",
    }
    base.update(overrides)
    return LakehouseConfig(**base)


def _arrow_ipc_bytes(table: pa.Table) -> bytes:
    """Serialise an Arrow table as an IPC stream."""
    sink = io.BytesIO()
    with paipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _add_verify_response() -> None:
    """Register the testconnection mock used by every test."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/testconnection",
        json={"message": "ok"},
        status=200,
    )


@responses.activate
def test_query_returns_pyarrow_table() -> None:
    """A successful execution materialises into a pyarrow Table."""
    _add_verify_response()
    table = pa.table({"id": [1], "name": ["alice"]})
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [
                {"columnName": "id", "dataType": "BIGINT"},
                {"columnName": "name", "dataType": "VARCHAR"},
            ],
            "totalRecords": 1,
            "hasMore": False,
            "chunks": [{"url": S3_URL_TEMPLATE.format(0), "rowCount": 1}],
        },
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body=_arrow_ipc_bytes(table),
        status=200,
    )

    with LakehouseClient(_config()) as client:
        result = client.query("SELECT * FROM customers")

    assert result.total_rows == 1
    assert result.truncated is False
    arrow = result.to_arrow()
    assert arrow.num_rows == 1
    assert arrow.column_names == ["id", "name"]


@responses.activate
def test_query_preserves_schema_for_empty_result() -> None:
    """Zero-chunk results still expose the column metadata."""
    _add_verify_response()
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

    with LakehouseClient(_config()) as client:
        result = client.query("SELECT * FROM empty_table")

    assert result.total_rows == 0
    assert result.to_arrow().column_names == ["id", "name"]


@responses.activate
def test_authentication_failure_raises() -> None:
    """HTTP 401 from the provider surfaces as :class:`AuthenticationError`."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/testconnection",
        status=401,
    )

    with (
        LakehouseClient(_config()) as client,
        pytest.raises(AuthenticationError),
    ):
        client.test_connection()


@responses.activate
def test_query_error_raises() -> None:
    """An ``error`` envelope surfaces as :class:`QueryError`."""
    _add_verify_response()
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"status": "error", "message": "syntax error"},
        status=200,
    )

    with LakehouseClient(_config()) as client, pytest.raises(QueryError):
        client.query("SELECT BAD FROM")


@responses.activate
def test_iter_batches_streams_results() -> None:
    """``iter_batches`` yields record batches across multiple chunks."""
    _add_verify_response()
    chunks = [
        pa.table({"id": pa.array([1, 2], type=pa.int64())}),
        pa.table({"id": pa.array([3, 4], type=pa.int64())}),
    ]
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [{"columnName": "id", "dataType": "BIGINT"}],
            "totalRecords": 4,
            "hasMore": False,
            "chunks": [{"url": S3_URL_TEMPLATE.format(i)} for i in range(2)],
        },
        status=200,
    )
    for index, table in enumerate(chunks):
        responses.add(
            responses.GET,
            S3_URL_TEMPLATE.format(index),
            body=_arrow_ipc_bytes(table),
            status=200,
        )

    rows: list[int] = []
    with LakehouseClient(_config()) as client:
        for batch in client.iter_batches("SELECT id FROM customers"):
            rows.extend(batch.column("id").to_pylist())

    assert rows == [1, 2, 3, 4]


@responses.activate
def test_query_parallel_concatenates_partitions() -> None:
    """``query_parallel`` fans out range-partitioned subqueries."""
    _add_verify_response()
    for value in (1, 2, 3):
        responses.add(
            responses.POST,
            f"{PROVIDER_URL}/iceberg/v1/statements",
            json={
                "schema": [{"columnName": "id", "dataType": "BIGINT"}],
                "totalRecords": 1,
                "hasMore": False,
                "chunks": [{"url": S3_URL_TEMPLATE.format(value)}],
            },
            status=200,
        )
        responses.add(
            responses.GET,
            S3_URL_TEMPLATE.format(value),
            body=_arrow_ipc_bytes(
                pa.table({"id": pa.array([value], type=pa.int64())})
            ),
            status=200,
        )

    with LakehouseClient(_config()) as client:
        result = client.query_parallel(
            sql_template="SELECT id FROM customers",
            partition_column="id",
            bounds=[(1, 1), (2, 2), (3, 3)],
            max_workers=2,
        )

    assert result.total_rows == 3
    assert sorted(row["id"] for row in result.to_list()) == [1, 2, 3]
