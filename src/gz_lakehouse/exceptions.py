"""Exception hierarchy for the gz-lakehouse SDK.

All errors derive from :class:`GzLakehouseError` so callers can catch the
whole family with a single ``except`` clause while still distinguishing
specific failure modes.

The hierarchy splits :class:`QueryError` into a *validation* failure
(caller-side mistake, never retried) and an *execution* failure
(provider-side, may be retried). Existing code that catches the parent
:class:`QueryError` continues to work.
"""


class GzLakehouseError(Exception):
    """Base class for every error raised by the SDK."""


class ConfigurationError(GzLakehouseError):
    """Raised when the client is configured with invalid or missing fields."""


class AuthenticationError(GzLakehouseError):
    """Raised when the lakehouse provider rejects the supplied credentials."""


class AuthorizationError(GzLakehouseError):
    """Raised when the user lacks permission for the requested operation."""


class TransportError(GzLakehouseError):
    """Raised when the provider URL cannot be reached or times out."""


class QueryError(GzLakehouseError):
    """Raised when a query cannot be completed.

    Generic parent retained for backwards compatibility. Prefer the more
    specific subclasses :class:`QueryValidationError` and
    :class:`QueryExecutionError` going forward.
    """


class QueryValidationError(QueryError):
    """Raised for caller-side query problems that should never be retried."""


class QueryExecutionError(QueryError):
    """Raised when the provider executed the query and returned a failure."""
