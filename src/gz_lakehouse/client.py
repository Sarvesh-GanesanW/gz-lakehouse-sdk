"""High-level client for the GroundZero Lakehouse provider.

The client owns the HTTP transport and produces :class:`Session`
objects via :meth:`LakehouseClient.start_session`. Sessions are the
unit of compute: each session is a warm pod with workers registered,
and statements run on the session amortise the pod-boot cost.

For one-off scripts the convenience methods :meth:`query`,
:meth:`iter_batches`, and :meth:`query_parallel` auto-create and stop
a session per call. For repeated work, prefer the explicit Session
form so the cost is paid once.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Any

import pyarrow as pa

from gz_lakehouse._http import HttpClient
from gz_lakehouse._logging import get_logger
from gz_lakehouse._transport import Transport
from gz_lakehouse.config import LakehouseConfig
from gz_lakehouse.exceptions import ConfigurationError
from gz_lakehouse.result import QueryResult
from gz_lakehouse.session import Session

_logger = get_logger("client")


class LakehouseClient:
    """Connection to a GroundZero Lakehouse provider."""

    def __init__(self, config: LakehouseConfig) -> None:
        """Initialise the client with the supplied configuration."""
        if not isinstance(config, LakehouseConfig):
            raise ConfigurationError(
                "LakehouseClient requires a LakehouseConfig instance",
            )
        self._config = config
        self._http = HttpClient(
            base_url=config.lakehouse_url,
            site=config.derived_site,
            connect_timeout_seconds=config.connect_timeout_seconds,
            max_retries=config.max_retries,
            backoff_seconds=config.backoff_seconds,
            pool_connections=config.pool_connections,
            pool_maxsize=config.pool_maxsize,
            enable_compression=config.enable_compression,
        )
        self._transport = Transport(http=self._http, config=config)
        self._verified = False

    @classmethod
    def from_kwargs(
        cls,
        lakehouse_url: str,
        warehouse: str,
        database: str,
        username: str,
        password: str,
        site: str | None = None,
        compute_size: str = "small",
        compute_id: int | None = None,
        **extra: Any,
    ) -> LakehouseClient:
        """Build a client without explicitly constructing a config."""
        config = LakehouseConfig(
            lakehouse_url=lakehouse_url,
            warehouse=warehouse,
            database=database,
            username=username,
            password=password,
            site=site,
            compute_size=compute_size,
            compute_id=compute_id,
            **extra,
        )
        return cls(config)

    @classmethod
    def from_env(
        cls,
        prefix: str = "GZ_LAKEHOUSE_",
        **overrides: Any,
    ) -> LakehouseClient:
        """Build a client from environment variables."""
        return cls(LakehouseConfig.from_env(prefix=prefix, **overrides))

    def test_connection(self) -> bool:
        """Verify the provider is reachable and credentials are valid."""
        self._transport.verify()
        self._verified = True
        return True

    def start_session(self) -> Session:
        """Create a warm compute session and return it.

        Always pair with :meth:`Session.stop` (or use the session as a
        context manager) to release the pod. The first call also
        verifies provider connectivity if it has not been verified yet.
        """
        self._ensure_verified()
        session_id = self._transport.start_session()
        _logger.info("session started: %s", session_id)
        return Session(
            session_id=session_id,
            transport=self._transport,
            config=self._config,
        )

    def query(
        self,
        sql: str,
    ) -> QueryResult:
        """Convenience: create a session, run ``sql``, stop the session.

        Pays the full session-start cost on every call. For repeated
        work, prefer :meth:`start_session` so the cost is amortised.
        """
        with self.start_session() as session:
            return session.query(sql)

    def iter_batches(
        self,
        sql: str,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Convenience-streaming wrapper.

        Memory-bound but pays the full session-start cost. The
        underlying session is held open until the iterator is fully
        consumed, then stopped.
        """
        session = self.start_session()
        try:
            yield from session.iter_batches(
                sql,
                batch_size=batch_size,
            )
        finally:
            session.stop()

    def query_parallel(
        self,
        sql_template: str,
        partition_column: str,
        bounds: Sequence[tuple[Any, Any]],
        max_workers: int | None = None,
    ) -> QueryResult:
        """Convenience wrapper for fan-out partitioned queries."""
        with self.start_session() as session:
            return session.query_parallel(
                sql_template=sql_template,
                partition_column=partition_column,
                bounds=bounds,
                max_workers=max_workers,
            )

    def close(self) -> None:
        """Release HTTP connection pools held by the client."""
        self._transport.close()
        self._http.close()

    def __enter__(self) -> LakehouseClient:
        """Enter a context-manager block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the HTTP session when leaving a ``with`` block."""
        self.close()

    def _ensure_verified(self) -> None:
        """Run the verification handshake the first time it's needed."""
        if not self._verified:
            self.test_connection()
