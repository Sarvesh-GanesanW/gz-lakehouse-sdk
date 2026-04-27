"""Tests for the Spark conversion helpers in :mod:`gz_lakehouse._spark`.

These tests use a fake :class:`SparkSession` rather than spinning up a
real JVM. The fake captures which conversion path the SDK chose
(Arrow Table direct, arrow-enabled pandas, or plain pandas) so we can
assert the version-aware dispatch picks the right one without paying
the cost of a real Spark instance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gz_lakehouse._spark import (
    _ARROW_CONFIG,
    arrow_to_spark,
    parse_pyspark_version,
    write_arrow_to_parquet_for_spark,
)


def _read_parquet(path: str) -> pa.Table:
    """Read a Parquet file back into a pa.Table for round-trip checks."""
    return pq.read_table(path)


class _FakeSparkConf:
    """Tracks Spark conf get/set/unset calls for assertions."""

    def __init__(self) -> None:
        """Initialise an empty conf store."""
        self._values: dict[str, str] = {}
        self.history: list[tuple[str, str, str | None]] = []

    def get(self, key: str) -> str:
        """Mimic Spark's get behaviour, raising when the key is unset."""
        if key not in self._values:
            raise Exception("Configuration key not set")
        return self._values[key]

    def set(self, key: str, value: str) -> None:
        """Record the set and update the value."""
        self.history.append(("set", key, value))
        self._values[key] = value

    def unset(self, key: str) -> None:
        """Record the unset and drop the key."""
        self.history.append(("unset", key, None))
        self._values.pop(key, None)


class _FakeSpark:
    """Fake :class:`SparkSession` that records what was passed in."""

    def __init__(self, accept_arrow_table: bool) -> None:
        """Configure whether ``createDataFrame`` accepts a pa.Table."""
        self.conf = _FakeSparkConf()
        self.accept_arrow_table = accept_arrow_table
        self.calls: list[tuple[str, Any]] = []

    def createDataFrame(  # noqa: N802
        self, data: Any, schema: Any | None = None
    ) -> str:
        """Pretend to build a DataFrame, recording the input type."""
        if isinstance(data, pa.Table):
            if not self.accept_arrow_table:
                raise TypeError(
                    "createDataFrame does not support pyarrow.Table"
                )
            self.calls.append(("arrow_table", data))
            return "spark-df-from-arrow"
        self.calls.append(("pandas", data))
        return "spark-df-from-pandas"


def test_parse_pyspark_version_handles_dev_suffix() -> None:
    """Version strings with ``dev``/``rc`` suffixes parse cleanly."""
    assert parse_pyspark_version("3.4.1") == (3, 4, 1)
    assert parse_pyspark_version("3.5.0.dev0") == (3, 5, 0)
    assert parse_pyspark_version("4.0.0rc1") == (4, 0, 0)
    assert parse_pyspark_version("3.4") == (3, 4, 0)


def test_arrow_to_spark_uses_arrow_table_on_modern_pyspark() -> None:
    """PySpark 3.4+ takes the zero-copy Arrow Table path."""
    table = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
    spark = _FakeSpark(accept_arrow_table=True)

    with patch("gz_lakehouse._spark._supports_arrow_table_input") as flag:
        flag.return_value = True
        df = arrow_to_spark(table, spark)

    assert df == "spark-df-from-arrow"
    assert spark.calls[0][0] == "arrow_table"


def test_arrow_to_spark_falls_back_to_arrow_pandas() -> None:
    """On older PySpark the helper enables the Arrow pandas path."""
    table = pa.table({"id": pa.array([1, 2], type=pa.int64())})
    spark = _FakeSpark(accept_arrow_table=False)

    with patch("gz_lakehouse._spark._supports_arrow_table_input") as flag:
        flag.return_value = False
        df = arrow_to_spark(table, spark)

    assert df == "spark-df-from-pandas"
    set_events = [item for item in spark.conf.history if item[0] == "set"]
    assert any(
        key == _ARROW_CONFIG and value == "true"
        for _, key, value in set_events
    )


def test_arrow_to_spark_handles_empty_table() -> None:
    """Empty results still produce an empty DataFrame with the schema."""
    table = pa.table(
        {"id": pa.array([], type=pa.int64())},
        schema=pa.schema([pa.field("id", pa.int64())]),
    )
    spark = MagicMock()
    spark.createDataFrame.return_value = "empty-df"

    df = arrow_to_spark(table, spark)

    assert df == "empty-df"
    args, kwargs = spark.createDataFrame.call_args
    assert args[0] == []
    assert kwargs.get("schema") == "id bigint"


def test_write_arrow_to_parquet_for_spark_returns_path(
    tmp_path: Any,
) -> None:
    """The helper writes Parquet and returns the destination path."""
    table = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
    destination = tmp_path / "out.parquet"

    result = write_arrow_to_parquet_for_spark(table, str(destination))

    assert result == str(destination)
    assert destination.exists()
    round_tripped = _read_parquet(str(destination))
    assert round_tripped.column("id").to_pylist() == [1, 2, 3]


def test_arrow_to_spark_arrow_path_falls_back_on_failure() -> None:
    """A failing Arrow Table path drops back to arrow-enabled pandas."""
    table = pa.table({"id": pa.array([1], type=pa.int64())})
    spark = _FakeSpark(accept_arrow_table=False)

    with patch("gz_lakehouse._spark._supports_arrow_table_input") as flag:
        flag.return_value = True
        df = arrow_to_spark(table, spark)

    assert df == "spark-df-from-pandas"


@pytest.fixture(autouse=True)
def _silence_pyspark_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the optional ``pyspark`` import for version detection."""
    fake_pyspark = type("pyspark", (), {"__version__": "3.4.0"})()
    monkeypatch.setitem(__import__("sys").modules, "pyspark", fake_pyspark)
