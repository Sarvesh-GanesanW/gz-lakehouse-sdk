"""Tests for :class:`gz_lakehouse.Session`."""

from __future__ import annotations

import io
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
import responses

from gz_lakehouse import (
    LakehouseClient,
    LakehouseConfig,
    QueryExecutionError,
    QueryValidationError,
)
from gz_lakehouse.session import (
    _compose_partitioned_sql,
    _render_literal,
    _validate_partition_template,
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


@responses.activate
def test_iter_batches_split_with_explicit_bounds() -> None:
    """Two splits over an integer range produce interleaved batches."""
    _stub_session_lifecycle()
    left = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
    right = pa.table({"id": pa.array([4, 5, 6], type=pa.int64())})
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [{"columnName": "id", "dataType": "BIGINT"}],
            "totalRecords": 3,
            "hasMore": False,
            "chunks": [{"url": S3_URL_TEMPLATE.format(0)}],
        },
        status=200,
    )
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [{"columnName": "id", "dataType": "BIGINT"}],
            "totalRecords": 3,
            "hasMore": False,
            "chunks": [{"url": S3_URL_TEMPLATE.format(1)}],
        },
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(0),
        body=_arrow_ipc_bytes(left),
        status=200,
    )
    responses.add(
        responses.GET,
        S3_URL_TEMPLATE.format(1),
        body=_arrow_ipc_bytes(right),
        status=200,
    )

    with LakehouseClient(_config()) as client, client.start_session() as s:
        rows: list[int] = []
        for batch in s.iter_batches_split(
            "SELECT id FROM customers",
            split_by="id",
            splits=2,
            bounds=(1, 6),
        ):
            rows.extend(batch.column("id").to_pylist())

    assert sorted(rows) == [1, 2, 3, 4, 5, 6]


@responses.activate
def test_iter_batches_split_consumer_break_does_not_hang() -> None:
    """Breaking out of the iterator early releases workers via shutdown drain.

    Without the drain in the main generator's finally, a worker
    blocked on a full ``out_queue.put`` would not see ``shutdown``
    and session teardown would hang. With the drain it exits.
    """
    _stub_session_lifecycle()
    big = pa.table({"id": pa.array(list(range(1000)), type=pa.int64())})
    for i in range(2):
        responses.add(
            responses.POST,
            f"{PROVIDER_URL}/iceberg/v1/statements",
            json={
                "schema": [{"columnName": "id", "dataType": "BIGINT"}],
                "totalRecords": 1000,
                "hasMore": False,
                "chunks": [{"url": S3_URL_TEMPLATE.format(i)}],
            },
            status=200,
        )
        responses.add(
            responses.GET,
            S3_URL_TEMPLATE.format(i),
            body=_arrow_ipc_bytes(big),
            status=200,
        )

    rows_seen: list[int] = []
    with LakehouseClient(_config()) as client, client.start_session() as s:
        for batch in s.iter_batches_split(
            "SELECT id FROM customers",
            split_by="id",
            splits=2,
            bounds=(0, 999),
            batch_size=10,
        ):
            rows_seen.extend(batch.column("id").to_pylist())
            if len(rows_seen) >= 5:
                break

    assert len(rows_seen) >= 5


def test_render_literal_renders_native_types() -> None:
    """Numbers, bool, None, and strings render with the right SQL syntax."""
    assert _render_literal(42) == "42"
    assert _render_literal(3.14) == "3.14"
    assert _render_literal(True) == "TRUE"
    assert _render_literal(False) == "FALSE"
    assert _render_literal(None) == "NULL"
    assert _render_literal("plain") == "'plain'"


def test_render_literal_escapes_single_quotes_ansi_correctly() -> None:
    """Single quotes double up so the literal cannot break out."""
    assert _render_literal("O'Brien") == "'O''Brien'"
    assert _render_literal("'; DROP TABLE x; --") == "'''; DROP TABLE x; --'"


def test_render_literal_renders_decimal_unquoted() -> None:
    """Decimal renders as a numeric literal, not a string."""
    assert _render_literal(Decimal("123.45")) == "123.45"


def test_render_literal_renders_dates_and_timestamps() -> None:
    """date and datetime render with typed SQL literal syntax."""
    assert _render_literal(date(2025, 1, 15)) == "DATE '2025-01-15'"
    assert (
        _render_literal(datetime(2025, 1, 15, 12, 30, 45))
        == "TIMESTAMP '2025-01-15 12:30:45'"
    )


def test_render_literal_rejects_unsupported_types() -> None:
    """Bytes / lists / dicts have no SQL representation; reject loudly."""
    with pytest.raises(QueryValidationError):
        _render_literal(b"raw")
    with pytest.raises(QueryValidationError):
        _render_literal([1, 2, 3])
    with pytest.raises(QueryValidationError):
        _render_literal({"key": "value"})


@responses.activate
def test_iter_batches_split_validates_inputs() -> None:
    """Negative splits, empty split_by, and stopped sessions are rejected."""
    _stub_session_lifecycle()

    with LakehouseClient(_config()) as client, client.start_session() as s:
        with pytest.raises(QueryValidationError):
            list(
                s.iter_batches_split(
                    "SELECT 1",
                    split_by="",
                    splits=2,
                    bounds=(0, 10),
                ),
            )
        with pytest.raises(QueryValidationError):
            list(
                s.iter_batches_split(
                    "SELECT 1",
                    split_by="id",
                    splits=0,
                    bounds=(0, 10),
                ),
            )


def test_compose_partitioned_sql_appends_where_when_absent() -> None:
    """Bare SELECT gets a ``WHERE`` appended."""
    composed = _compose_partitioned_sql(
        "SELECT id FROM orders",
        "id",
        1,
        100,
    )
    assert composed == "SELECT id FROM orders WHERE id BETWEEN 1 AND 100"


def test_compose_partitioned_sql_appends_and_with_outer_where() -> None:
    """An existing outer ``WHERE`` causes ``AND`` instead."""
    composed = _compose_partitioned_sql(
        "SELECT id FROM orders WHERE region = 'US'",
        "id",
        1,
        100,
    )
    assert composed == (
        "SELECT id FROM orders WHERE region = 'US' AND id BETWEEN 1 AND 100"
    )


def test_compose_partitioned_sql_ignores_where_in_string_literal() -> None:
    """A ``WHERE`` substring inside a literal must not trigger AND-mode."""
    composed = _compose_partitioned_sql(
        "SELECT 'WHERE NOT' AS msg, id FROM t",
        "id",
        1,
        2,
    )
    assert composed == (
        "SELECT 'WHERE NOT' AS msg, id FROM t WHERE id BETWEEN 1 AND 2"
    )


def test_compose_partitioned_sql_ignores_where_inside_subquery() -> None:
    """An inner subquery WHERE doesn't trigger AND on the outer query."""
    composed = _compose_partitioned_sql(
        "SELECT id FROM (SELECT id FROM raw WHERE valid) sub",
        "id",
        1,
        2,
    )
    assert composed == (
        "SELECT id FROM (SELECT id FROM raw WHERE valid) sub "
        "WHERE id BETWEEN 1 AND 2"
    )


def test_validate_partition_template_rejects_outer_order_by() -> None:
    """Outer ORDER BY is rejected so composition produces no broken SQL."""
    with pytest.raises(QueryValidationError, match="ORDER BY"):
        _validate_partition_template(
            "SELECT id FROM t ORDER BY id",
        )


def test_validate_partition_template_rejects_outer_limit() -> None:
    """Outer LIMIT is rejected."""
    with pytest.raises(QueryValidationError, match="LIMIT"):
        _validate_partition_template("SELECT id FROM t LIMIT 10")


def test_validate_partition_template_rejects_set_operator() -> None:
    """UNION at the outer level is rejected (composition would break)."""
    with pytest.raises(QueryValidationError, match="UNION"):
        _validate_partition_template(
            "SELECT id FROM a UNION SELECT id FROM b",
        )


def test_validate_partition_template_allows_clause_inside_subquery() -> None:
    """ORDER BY inside a CTE / subquery is fine — only outer-level rejected."""
    _validate_partition_template(
        "SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT 10) sub",
    )


def test_validate_partition_template_allows_keyword_in_literal() -> None:
    """``LIMIT`` inside a literal must not trigger validation."""
    _validate_partition_template(
        "SELECT id, 'LIMIT 10 mention' AS note FROM t",
    )


@responses.activate
def test_query_parallel_rejects_outer_order_by() -> None:
    """``query_parallel`` validates the template before going over wire."""
    _stub_session_lifecycle()

    with (
        LakehouseClient(_config()) as client,
        client.start_session() as s,
        pytest.raises(QueryValidationError, match="ORDER BY"),
    ):
        s.query_parallel(
            sql_template="SELECT id FROM t ORDER BY id",
            partition_column="id",
            bounds=[(0, 10)],
        )


@responses.activate
def test_query_parallel_merges_timings_across_partitions() -> None:
    """``QueryResult.timings`` is populated and aggregates across legs."""
    _stub_session_lifecycle()
    for value in (1, 2):
        responses.add(
            responses.POST,
            f"{PROVIDER_URL}/iceberg/v1/statements",
            json={
                "schema": [{"columnName": "id", "dataType": "BIGINT"}],
                "totalRecords": 1,
                "hasMore": False,
                "chunks": [
                    {
                        "url": S3_URL_TEMPLATE.format(value),
                        "compressedSize": 100,
                        "uncompressedSize": 200,
                    },
                ],
            },
            status=200,
        )
        responses.add(
            responses.GET,
            S3_URL_TEMPLATE.format(value),
            body=_arrow_ipc_bytes(
                pa.table({"id": pa.array([value], type=pa.int64())}),
            ),
            status=200,
        )

    with LakehouseClient(_config()) as client, client.start_session() as s:
        result = s.query_parallel(
            sql_template="SELECT id FROM t",
            partition_column="id",
            bounds=[(1, 1), (2, 2)],
            max_workers=2,
        )

    assert result.timings is not None
    assert result.timings.chunk_count == 2
    assert result.timings.compressed_bytes == 200
    assert result.timings.uncompressed_bytes == 400


@responses.activate
def test_query_parallel_fail_fast_propagates_first_error() -> None:
    """A failing partition surfaces; the others are cancelled / discarded."""
    _stub_session_lifecycle()
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={"status": "error", "message": "boom"},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{PROVIDER_URL}/iceberg/v1/statements",
        json={
            "schema": [],
            "totalRecords": 0,
            "hasMore": False,
            "chunks": [],
        },
        status=200,
    )

    with (
        LakehouseClient(_config()) as client,
        client.start_session() as s,
        pytest.raises(QueryExecutionError, match="boom"),
    ):
        s.query_parallel(
            sql_template="SELECT id FROM t",
            partition_column="id",
            bounds=[(1, 1), (2, 2)],
            max_workers=2,
        )
