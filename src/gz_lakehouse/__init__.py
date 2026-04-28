"""Public API for the gz-lakehouse SDK.

Importing :mod:`gz_lakehouse` exposes the small surface a caller needs:
configuration, the client itself, the result wrapper, and the exception
hierarchy. Internal modules (``_http``, ``_transport``, ``_arrow_build``,
``_logging``, ``_version``) are not re-exported.
"""

from gz_lakehouse._transport import ExecutorChoice
from gz_lakehouse._version import __version__
from gz_lakehouse.client import LakehouseClient
from gz_lakehouse.config import LakehouseConfig
from gz_lakehouse.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    GzLakehouseError,
    QueryError,
    QueryExecutionError,
    QueryValidationError,
    TransportError,
)
from gz_lakehouse.result import QueryResult
from gz_lakehouse.session import Session

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "ExecutorChoice",
    "GzLakehouseError",
    "LakehouseClient",
    "LakehouseConfig",
    "QueryError",
    "QueryExecutionError",
    "QueryResult",
    "QueryValidationError",
    "Session",
    "TransportError",
    "__version__",
]
