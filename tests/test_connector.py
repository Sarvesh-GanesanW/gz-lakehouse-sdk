"""Tests for the Snowflake-style connector facade."""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
import responses

from gz_lakehouse import ConfigurationError, connect

PROVIDER_URL = "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
S3_URL = "https://s3.example.com/chunk-0.arrow"
SESSION_ID = "session-connector-1"


def _arrow_ipc_bytes(table: pa.Table) -> bytes:
    """Serialise an Arrow table as an IPC stream."""
    sink = io.BytesIO()
    with paipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _stub_lifecycle() -> None:
    """Mock the provider endpoints used by one connector query."""
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/testconnection",
        json={"message": "ok"},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/startsession",
        json={"status": 200, "response": {"sessionId": SESSION_ID}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/stopsession",
        json={"status": 200, "response": {}},
        status=200,
    )


def _connection_kwargs() -> dict[str, object]:
    """Return a minimal connector configuration."""
    return {
        "lakehouse_url": PROVIDER_URL,
        "siteName": "admin",
        "warehouse": "wh",
        "database": "db",
        "username": "alice",
        "password": "secret",
    }


@responses.activate
def test_connect_cursor_execute_fetches_rows() -> None:
    """``connect().cursor().execute()`` exposes DB-API-like fetch methods."""
    _stub_lifecycle()
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [
                {"columnName": "id", "dataType": "BIGINT"},
                {"columnName": "name", "dataType": "VARCHAR"},
            ],
            "totalRecords": 2,
            "hasMore": False,
            "chunks": [{"url": S3_URL, "rowCount": 2}],
        },
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL,
        body=_arrow_ipc_bytes(table),
        status=200,
    )

    with connect(**_connection_kwargs()) as connection:
        cursor = connection.cursor().execute("SELECT id, name FROM t")

        assert cursor.fetchone() == {"id": 1, "name": "a"}
        assert cursor.fetchall() == [{"id": 2, "name": "b"}]
        assert cursor.rowcount == 2
        assert cursor.description == [
            ("id", "BIGINT", None, None, None, None, None),
            ("name", "VARCHAR", None, None, None, None, None),
        ]
        assert cursor.fetch_arrow_all().num_rows == 2

    assert responses.calls[0].request.headers["gz-site"] == "admin"


def test_connect_requires_explicit_site_name() -> None:
    """Connector users must pass ``siteName`` explicitly."""
    kwargs = _connection_kwargs()
    kwargs.pop("siteName")

    with pytest.raises(ConfigurationError, match="siteName"):
        connect(**kwargs)
