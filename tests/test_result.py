"""Tests for :class:`gz_lakehouse.QueryResult` conversions."""

import pyarrow as pa

from gz_lakehouse import QueryResult


def _result_from_arrow(table: pa.Table) -> QueryResult:
    """Build a result wrapper from an Arrow table for testing conversions."""
    schema = [
        {"columnName": field.name, "dataType": str(field.type)}
        for field in table.schema
    ]
    return QueryResult(
        table=table,
        schema=schema,
        truncated=False,
        total_rows=table.num_rows,
    )


def test_to_arrow_returns_underlying_table() -> None:
    """``to_arrow`` exposes the wrapped table directly."""
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    result = _result_from_arrow(table)

    assert result.to_arrow() is table


def test_to_pandas_round_trip() -> None:
    """``to_pandas`` produces a DataFrame with the right columns."""
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    result = _result_from_arrow(table)

    df = result.to_pandas()

    assert list(df.columns) == ["id", "name"]
    assert df.shape == (2, 2)


def test_to_list_returns_row_dicts() -> None:
    """``to_list`` converts the table into row dicts."""
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    result = _result_from_arrow(table)

    assert result.to_list() == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]


def test_iter_batches_respects_batch_size() -> None:
    """``iter_batches`` yields slices no larger than ``batch_size`` rows."""
    table = pa.table({"id": list(range(10))})
    result = _result_from_arrow(table)

    batches = list(result.iter_batches(batch_size=3))

    assert sum(batch.num_rows for batch in batches) == 10
    assert all(batch.num_rows <= 3 for batch in batches)


def test_arrow_schema_property_exposes_authoritative_schema() -> None:
    """``arrow_schema`` exposes the table's pa.Schema directly."""
    table = pa.table({"id": pa.array([1, 2], type=pa.int64())})
    result = _result_from_arrow(table)

    assert result.arrow_schema.field("id").type == pa.int64()


def test_repr_includes_row_count_and_schema() -> None:
    """``repr`` summarises the result without dumping every row."""
    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["a", "b"], type=pa.string()),
        }
    )
    result = _result_from_arrow(table)

    text = repr(result)
    assert "rows=2" in text
    assert "id:int64" in text
    assert "name:string" in text
