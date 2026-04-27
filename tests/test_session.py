"""Tests for :class:`gz_lakehouse.Session`."""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
import responses

from gz_lakehouse import (
    LakehouseClient,
    LakehouseConfig,
    QueryValidationError,
)

PROVIDER_URL = "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
S3_URL_TEMPLATE = "https://s3.example.com/chunk-{}.arrow"
SESSION_ID = "session-fixture-1"


def _config(**overrides: Any) -> LakehouseConfig:
    """Default config used by the session tests."""
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


def _stub_session_lifecycle() -> None:
    """Mock testconnection + start/stop session endpoints."""
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
        json={"status": 200, "response": {"message": "ok"}},
        status=200,
    )


@responses.activate
def test_session_context_manager_stops_on_exit() -> None:
    """Leaving the ``with`` block calls ``stop_session`` once."""
    _stub_session_lifecycle()

    with LakehouseClient(_config()) as client:
        with client.start_session() as session:
            assert session.closed is False
        assert session.closed is True

    stop_calls = [
        c
        for c in responses.calls
        if c.request.url.endswith("/iceberg/stopsession")
    ]
    assert len(stop_calls) == 1


@responses.activate
def test_session_stop_is_idempotent() -> None:
    """Calling ``stop`` more than once is safe and does not double-post."""
    _stub_session_lifecycle()

    with LakehouseClient(_config()) as client:
        session = client.start_session()
        session.stop()
        session.stop()

    stop_calls = [
        c
        for c in responses.calls
        if c.request.url.endswith("/iceberg/stopsession")
    ]
    assert len(stop_calls) == 1


@responses.activate
def test_session_query_after_stop_raises() -> None:
    """Calls on a stopped session surface as :class:`QueryValidationError`."""
    _stub_session_lifecycle()

    with LakehouseClient(_config()) as client:
        session = client.start_session()
        session.stop()
        with pytest.raises(QueryValidationError):
            session.query("SELECT 1")


@responses.activate
def test_session_query_runs() -> None:
    """``session.query`` returns a :class:`QueryResult`."""
    _stub_session_lifecycle()
    table = pa.table({"x": pa.array([42], type=pa.int64())})
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [{"columnName": "x", "dataType": "BIGINT"}],
            "totalRecords": 1,
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

    with LakehouseClient(_config()) as client, client.start_session() as s:
        result = s.query("SELECT 42 AS x")

    assert result.total_rows == 1
    assert result.to_list() == [{"x": 42}]


@responses.activate
def test_session_rejects_empty_sql() -> None:
    """``session.query`` validates input before going over the wire."""
    _stub_session_lifecycle()

    with (
        LakehouseClient(_config()) as client,
        client.start_session() as s,
        pytest.raises(QueryValidationError),
    ):
        s.query("")
