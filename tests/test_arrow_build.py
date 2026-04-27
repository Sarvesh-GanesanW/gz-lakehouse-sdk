"""Tests for the schema helpers in :mod:`gz_lakehouse._arrow_build`."""

import pyarrow as pa

from gz_lakehouse._arrow_build import (
    arrow_type_for,
    empty_table_for,
    schema_to_descriptors,
)


def test_arrow_type_for_known_types() -> None:
    """Provider type strings map to the expected Arrow types."""
    assert arrow_type_for("BIGINT") == pa.int64()
    assert arrow_type_for("varchar(255)") == pa.string()
    assert arrow_type_for("TIMESTAMP") == pa.timestamp("us")
    assert arrow_type_for("boolean") == pa.bool_()


def test_arrow_type_for_unknown_falls_back_to_string() -> None:
    """Unrecognised types degrade to string rather than fail."""
    assert arrow_type_for("CUSTOM_TYPE") == pa.string()
    assert arrow_type_for(None) == pa.string()


def test_arrow_type_for_native_arrow_strings() -> None:
    """Arrow's own ``str(type)`` form (used by the pod manifest) maps cleanly."""
    assert arrow_type_for("int64") == pa.int64()
    assert arrow_type_for("int32") == pa.int32()
    assert arrow_type_for("float64") == pa.float64()
    assert arrow_type_for("float32") == pa.float32()
    assert arrow_type_for("bool") == pa.bool_()
    assert arrow_type_for("date32") == pa.date32()


def test_arrow_type_for_strips_bracket_suffixes() -> None:
    """``timestamp[us]`` and similar Arrow-native forms map to timestamp."""
    assert arrow_type_for("timestamp[us]") == pa.timestamp("us")
    assert arrow_type_for("timestamp[ms]") == pa.timestamp("us")
    assert arrow_type_for("time64[us]") == pa.time64("us")


def test_schema_to_descriptors_round_trip() -> None:
    """Arrow schemas convert back to provider-style descriptors."""
    schema = pa.schema(
        [pa.field("id", pa.int64()), pa.field("name", pa.string())]
    )

    descriptors = schema_to_descriptors(schema)

    assert descriptors == [
        {"columnName": "id", "dataType": "int64"},
        {"columnName": "name", "dataType": "string"},
    ]


def test_empty_table_for_preserves_columns() -> None:
    """Empty results keep their column names + types."""
    descriptors = [
        {"columnName": "id", "dataType": "BIGINT"},
        {"columnName": "name", "dataType": "VARCHAR"},
    ]

    table = empty_table_for(descriptors)

    assert table.num_rows == 0
    assert table.column_names == ["id", "name"]
    assert table.schema.field("id").type == pa.int64()


def test_empty_table_for_handles_no_descriptors() -> None:
    """No descriptors yields a truly empty table."""
    table = empty_table_for(None)

    assert table.num_rows == 0
    assert table.column_names == []
