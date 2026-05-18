"""Snowflake-style connector facade for GroundZero Lakehouse."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import TracebackType
from typing import Any

import pyarrow as pa

from gz_lakehouse._transport import ExecutorChoice
from gz_lakehouse.client import LakehouseClient
from gz_lakehouse.config import LakehouseConfig
from gz_lakehouse.exceptions import ConfigurationError, QueryError
from gz_lakehouse.pipeline_config import PipelineConfig
from gz_lakehouse.result import QueryResult
from gz_lakehouse.session import Session


def connect(**kwargs: Any) -> LakehouseConnection:
    """Return a connector-style connection.

    The public call shape intentionally accepts ``siteName`` to match the
    GroundZero tenant vocabulary used by provider APIs and notebooks.
    """
    config = LakehouseConfig(**_normalise_connector_kwargs(kwargs))
    return LakehouseConnection(LakehouseClient(config))


class LakehouseConnection:
    """Stateful connection that owns one warm compute session."""

    def __init__(self, client: LakehouseClient) -> None:
        """Bind a high-level client to the connector facade."""
        self._client = client
        self._session: Session | None = None
        self._closed = False

    def cursor(self) -> LakehouseCursor:
        """Return a new cursor bound to this connection."""
        self._ensure_open()
        return LakehouseCursor(self)

    def execute(
        self,
        sql: str,
        executor: ExecutorChoice = "auto",
        pipeline: PipelineConfig | None = None,
    ) -> LakehouseCursor:
        """Create a cursor, execute ``sql``, and return the cursor."""
        return self.cursor().execute(
            sql,
            executor=executor,
            pipeline=pipeline,
        )

    def close(self) -> None:
        """Stop the warm session and close HTTP resources."""
        if self._closed:
            return
        try:
            if self._session is not None:
                self._session.stop()
                self._session = None
        finally:
            self._client.close()
            self._closed = True

    def _get_session(self) -> Session:
        """Create or reuse the connection's warm compute session."""
        self._ensure_open()
        if self._session is None:
            self._session = self._client.start_session()
        return self._session

    def _ensure_open(self) -> None:
        """Raise when the connection has already been closed."""
        if self._closed:
            raise QueryError("Lakehouse connection is closed")

    def __enter__(self) -> LakehouseConnection:
        """Enter a context-manager block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the connection on context-manager exit."""
        self.close()


class LakehouseCursor:
    """Cursor-like object for executing SQL and fetching result sets."""

    arraysize = 1

    def __init__(self, connection: LakehouseConnection) -> None:
        """Create an empty cursor bound to ``connection``."""
        self._connection = connection
        self._result: QueryResult | None = None
        self._rows: list[Mapping[str, Any]] = []
        self._row_index = 0
        self._closed = False

    def execute(
        self,
        sql: str,
        executor: ExecutorChoice = "auto",
        pipeline: PipelineConfig | None = None,
    ) -> LakehouseCursor:
        """Execute ``sql`` and make its result available to fetch methods."""
        self._ensure_open()
        self._result = None
        self._rows = []
        self._row_index = 0
        result = self._connection._get_session().query(
            sql,
            executor=executor,
            pipeline=pipeline,
        )
        self._result = result
        self._rows = result.to_list()
        self._row_index = 0
        return self

    def fetchone(self) -> Mapping[str, Any] | None:
        """Return one remaining row, or ``None`` when exhausted."""
        self._ensure_result()
        if self._row_index >= len(self._rows):
            return None
        row = self._rows[self._row_index]
        self._row_index += 1
        return row

    def fetchmany(self, size: int | None = None) -> list[Mapping[str, Any]]:
        """Return up to ``size`` remaining rows."""
        self._ensure_result()
        fetch_size = self.arraysize if size is None else size
        if fetch_size < 0:
            raise QueryError("fetchmany size must be non-negative")
        start = self._row_index
        end = min(start + fetch_size, len(self._rows))
        self._row_index = end
        return self._rows[start:end]

    def fetchall(self) -> list[Mapping[str, Any]]:
        """Return all remaining rows."""
        self._ensure_result()
        rows = self._rows[self._row_index :]
        self._row_index = len(self._rows)
        return rows

    def fetch_arrow_all(self) -> pa.Table:
        """Return the complete result as a PyArrow table."""
        return self._current_result().to_arrow()

    def fetch_pandas_all(self) -> Any:
        """Return the complete result as a pandas DataFrame."""
        return self._current_result().to_pandas()

    @property
    def description(self) -> list[tuple[Any, ...]] | None:
        """Return DB-API-style column descriptors for the current result."""
        if self._result is None:
            return None
        return [
            (
                column.get("columnName"),
                column.get("dataType"),
                None,
                None,
                None,
                None,
                None,
            )
            for column in self._result.schema
        ]

    @property
    def rowcount(self) -> int:
        """Return the row count for the current result, or ``-1``."""
        if self._result is None:
            return -1
        return self._result.total_rows

    @property
    def result(self) -> QueryResult:
        """Return the underlying :class:`QueryResult`."""
        return self._current_result()

    def close(self) -> None:
        """Close the cursor without closing the parent connection."""
        self._closed = True
        self._result = None
        self._rows = []
        self._row_index = 0

    def _current_result(self) -> QueryResult:
        """Return the current result or raise a connector error."""
        self._ensure_result()
        if self._result is None:
            raise QueryError("No query has been executed")
        return self._result

    def _ensure_result(self) -> None:
        """Raise when fetch methods are called before ``execute``."""
        self._ensure_open()
        if self._result is None:
            raise QueryError("No query has been executed")

    def _ensure_open(self) -> None:
        """Raise when the cursor has already been closed."""
        if self._closed:
            raise QueryError("Lakehouse cursor is closed")

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Iterate over remaining rows."""
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row

    def __enter__(self) -> LakehouseCursor:
        """Enter a context-manager block."""
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the cursor on context-manager exit."""
        self.close()


def _normalise_connector_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Translate connector-style keyword names to ``LakehouseConfig``."""
    options = dict(kwargs)
    _move_alias(options, source="siteName", target="site_name")
    _move_alias(options, source="site", target="site_name")
    _move_alias(options, source="user", target="username")
    if not options.get("site_name"):
        raise ConfigurationError("connect requires explicit siteName")
    return options


def _move_alias(
    options: dict[str, Any],
    source: str,
    target: str,
) -> None:
    """Move ``source`` to ``target`` while rejecting ambiguous inputs."""
    if source not in options:
        return
    if target in options:
        raise ConfigurationError(
            f"Pass only one of {source!r} or {target!r}",
        )
    options[target] = options.pop(source)
