"""Schema helpers shared between the transport and result layers.

The Snowflake-style data plane carries data as Arrow IPC chunks on S3,
so there is no row-major JSON to convert. The two helpers here exist
only to (a) translate provider type strings into :class:`pyarrow.DataType`
instances when constructing an *empty* table for a zero-row result,
and (b) translate an Arrow schema back into the
``[{columnName, dataType}, ...]`` descriptor list that the
:class:`QueryResult` API exposes for round-trip use.
"""

from __future__ import annotations

import re

import pyarrow as pa

_TYPE_MAP: dict[str, pa.DataType] = {
    "boolean": pa.bool_(),
    "bool": pa.bool_(),
    "tinyint": pa.int8(),
    "int8": pa.int8(),
    "smallint": pa.int16(),
    "int16": pa.int16(),
    "int": pa.int32(),
    "int32": pa.int32(),
    "integer": pa.int32(),
    "bigint": pa.int64(),
    "int64": pa.int64(),
    "long": pa.int64(),
    "float": pa.float32(),
    "float32": pa.float32(),
    "real": pa.float32(),
    "double": pa.float64(),
    "float64": pa.float64(),
    "string": pa.string(),
    "varchar": pa.string(),
    "char": pa.string(),
    "binary": pa.binary(),
    "varbinary": pa.binary(),
    "date": pa.date32(),
    "date32": pa.date32(),
    "date64": pa.date64(),
    "timestamp": pa.timestamp("us"),
    "timestamptz": pa.timestamp("us", tz="UTC"),
    "time": pa.time64("us"),
    "time32": pa.time32("ms"),
    "time64": pa.time64("us"),
}


_DECIMAL_PATTERN = re.compile(
    r"decimal\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)",
    re.IGNORECASE,
)


def arrow_type_for(data_type: str | None) -> pa.DataType:
    """Map a provider ``dataType`` string to a :class:`pyarrow.DataType`.

    Accepts both Iceberg/Trino-style names (``BIGINT``, ``VARCHAR``,
    ``TIMESTAMP``) and Arrow-native string forms (``int64``, ``string``,
    ``timestamp[us]``). The pod writes manifests using
    ``str(field.type)`` which produces the latter; dataset metadata
    elsewhere in the platform uses the former. The mapper handles both
    so empty-result schema preservation works regardless of which side
    produced the descriptor.

    Decimal types preserve their precision and scale: ``DECIMAL(10,2)``
    maps to ``pa.decimal128(10, 2)`` instead of degrading to ``string``,
    so empty-result downstream consumers (pandas, Spark) keep numeric
    semantics. Precisions above 38 widen to ``pa.decimal256``;
    precisions above 76 (un-representable in Arrow) fall back to
    ``string`` since silently truncating precision is worse than
    surfacing a type mismatch.

    Args:
        data_type: Provider-supplied type string. ``None`` and unknown
            types fall back to ``string`` so the caller never has to
            handle a missing mapping.
    """
    if not data_type:
        return pa.string()
    normalised = data_type.strip().lower()
    if normalised.startswith("decimal"):
        return _decimal_type_for(data_type)
    normalised = normalised.split("(", 1)[0]
    normalised = normalised.split("[", 1)[0]
    return _TYPE_MAP.get(normalised, pa.string())


def _decimal_type_for(data_type: str) -> pa.DataType:
    """Parse ``decimal(p,s)`` into a precise Arrow decimal type.

    A bare ``decimal`` without precision falls back to
    ``decimal128(38, 18)`` — the widest decimal128 with a reasonable
    scale, matching what Iceberg/Spark emit for un-annotated DECIMAL.
    Precisions above what Arrow can represent (>76) degrade to
    string so the SDK never silently rounds the value space.
    """
    match = _DECIMAL_PATTERN.search(data_type)
    if not match:
        return pa.decimal128(38, 18)
    precision = int(match.group(1))
    scale = int(match.group(2) or 0)
    if precision < 1:
        return pa.string()
    if precision <= 38:
        return pa.decimal128(precision, scale)
    if precision <= 76:
        return pa.decimal256(precision, scale)
    return pa.string()


def schema_to_descriptors(schema: pa.Schema) -> list[dict[str, str]]:
    """Convert a :class:`pyarrow.Schema` into provider-style descriptors.

    Used so :class:`QueryResult` exposes a uniform ``schema`` list
    regardless of whether the descriptors came directly from the
    provider envelope or were derived from an Arrow chunk's own schema.
    """
    return [
        {"columnName": field.name, "dataType": str(field.type)}
        for field in schema
    ]


def empty_table_for(
    descriptors: list[dict[str, str]] | None,
) -> pa.Table:
    """Return an empty :class:`pyarrow.Table` with the column metadata.

    When the provider returns zero result chunks the SDK still needs to
    deliver a table whose schema matches what the query *would* have
    returned, so downstream consumers (pandas, Spark) keep their column
    names and types. ``descriptors`` is the provider's column list.
    """
    if not descriptors:
        return pa.table({})
    fields = [
        pa.field(d["columnName"], arrow_type_for(d.get("dataType")))
        for d in descriptors
    ]
    arrow_schema = pa.schema(fields)
    return pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in fields],
        schema=arrow_schema,
    )
