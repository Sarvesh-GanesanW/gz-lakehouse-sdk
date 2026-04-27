"""Internal HTTP client for the gz-lakehouse SDK.

Wraps :mod:`requests` with:

* a :class:`urllib3.util.retry.Retry`-backed adapter that handles
  transient failures (429/408/500/502/503/504) and respects the
  ``Retry-After`` header when present,
* configurable connection pool sizing for high-concurrency callers,
* separate connect and read timeouts (a hung TCP/TLS handshake no
  longer blocks indefinitely),
* opt-in transport compression (gzip / deflate / zstd) so the JSON
  envelope shrinks dramatically on the wire,
* a fast :mod:`orjson` decoder for JSON responses, with a graceful
  fallback to the standard library if ``orjson`` is unavailable,
* a ``post_stream`` entry point that returns the raw streaming response
  for binary transports such as Arrow IPC and chunked Parquet.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gz_lakehouse._logging import get_logger, new_request_id
from gz_lakehouse.exceptions import (
    AuthenticationError,
    AuthorizationError,
    QueryError,
    QueryExecutionError,
    TransportError,
)

try:
    import orjson as _orjson
except ImportError:
    _orjson = None

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_TRANSPORT_STATUS = frozenset({408, 429, 502, 503, 504})
_RETRYABLE_METHODS = frozenset({"POST", "GET"})

_logger = get_logger("http")


class HttpResponse:
    """Lightweight wrapper around :class:`requests.Response`.

    Exposes the methods needed by the transport layer without leaking
    the raw :mod:`requests` API. Allows decoding to JSON via
    :meth:`json` or streaming the binary body via :attr:`raw` /
    :meth:`iter_content`.
    """

    def __init__(self, response: requests.Response) -> None:
        """Wrap a :class:`requests.Response` for downstream consumers."""
        self._response = response

    @property
    def status_code(self) -> int:
        """HTTP status code returned by the provider."""
        return self._response.status_code

    @property
    def content_type(self) -> str:
        """Lower-cased ``Content-Type`` header value (without parameters)."""
        header = self._response.headers.get("Content-Type", "")
        return header.split(";", 1)[0].strip().lower()

    @property
    def headers(self) -> Mapping[str, str]:
        """All response headers."""
        return self._response.headers

    @property
    def raw(self) -> Any:
        """Raw urllib3 response stream for zero-copy binary readers."""
        self._response.raw.decode_content = True
        return self._response.raw

    def iter_content(self, chunk_size: int) -> Any:
        """Stream the body in ``chunk_size`` byte chunks."""
        return self._response.iter_content(chunk_size=chunk_size)

    def json(self) -> Mapping[str, Any]:
        """Decode the body as JSON using ``orjson`` when available."""
        body = self._response.content
        if not body:
            return {}
        if _orjson is not None:
            try:
                return _orjson.loads(body)
            except _orjson.JSONDecodeError as ex:
                raise QueryExecutionError(
                    f"Provider returned invalid JSON: {ex}"
                ) from ex
        try:
            return self._response.json()
        except ValueError as ex:
            raise QueryExecutionError(
                f"Provider returned invalid JSON: {ex}"
            ) from ex

    def text(self) -> str:
        """Decoded body for use in error messages."""
        return self._response.text

    def close(self) -> None:
        """Release the underlying connection back to the pool."""
        self._response.close()


class HttpClient:
    """HTTP client scoped to a single lakehouse provider URL."""

    def __init__(
        self,
        base_url: str,
        site: str,
        connect_timeout_seconds: int = 10,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        pool_connections: int = 4,
        pool_maxsize: int = 16,
        enable_compression: bool = True,
    ) -> None:
        """Initialise the client.

        Args:
            base_url: Provider URL, e.g. ``http://...groundzerodev.cloud``.
            site: Value sent in the ``gz-site`` header.
            connect_timeout_seconds: TCP/TLS handshake timeout.
            max_retries: Maximum retry attempts for retryable failures.
            backoff_seconds: Base delay for exponential backoff.
            pool_connections: Number of connection pools to keep.
            pool_maxsize: Maximum connections per pool.
            enable_compression: Whether to advertise gzip/deflate/zstd.
        """
        self._base_url = base_url.rstrip("/")
        self._site = site
        self._connect_timeout = connect_timeout_seconds
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._enable_compression = enable_compression
        self._session = self._build_session(
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
        )

    @staticmethod
    def _build_session(
        max_retries: int,
        backoff_seconds: float,
        pool_connections: int,
        pool_maxsize: int,
    ) -> requests.Session:
        """Construct a :class:`requests.Session` with retry + pool config."""
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_seconds,
            status_forcelist=tuple(_RETRYABLE_STATUS),
            allowed_methods=tuple(_RETRYABLE_METHODS),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
        )
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def post(
        self,
        path: str,
        json_body: Mapping[str, Any],
        timeout_seconds: int,
        accept: str = "application/json",
        stream: bool = False,
    ) -> HttpResponse:
        """POST a JSON body and return an :class:`HttpResponse`.

        Retries are handled inside the underlying :class:`Retry` adapter,
        so this method translates the *final* response (after retries)
        into either a successful :class:`HttpResponse` or a typed SDK
        exception.

        Args:
            path: Path relative to the base URL, e.g. ``/iceberg/v1/statements``.
            json_body: Payload serialised as JSON.
            timeout_seconds: Per-request read timeout. The connect
                timeout is taken from the client configuration.
            accept: Value of the ``Accept`` header. Defaults to
                ``application/json``.
            stream: When ``True`` the response body is *not* eagerly
                consumed. Callers must read :attr:`HttpResponse.raw` /
                :meth:`HttpResponse.iter_content` and call
                :meth:`HttpResponse.close` themselves.

        Returns:
            The wrapped :class:`HttpResponse`.

        Raises:
            AuthenticationError: HTTP 401 from the provider.
            AuthorizationError: HTTP 403 from the provider.
            QueryError: 4xx (other) or 5xx after retries.
            TransportError: Connection failure or timeout after retries.
        """
        endpoint = f"{self._base_url}{path}"
        request_id = new_request_id()
        headers = self._build_headers(accept=accept, request_id=request_id)

        _logger.debug(
            "POST %s accept=%s stream=%s req_id=%s",
            endpoint,
            accept,
            stream,
            request_id,
        )
        started = time.monotonic()
        try:
            response = self._session.post(
                endpoint,
                json=json_body,
                headers=headers,
                timeout=(self._connect_timeout, timeout_seconds),
                stream=stream,
            )
        except requests.RequestException as ex:
            _logger.warning(
                "POST %s failed transport req_id=%s err=%s",
                endpoint,
                request_id,
                ex,
            )
            raise TransportError(
                f"Cannot reach lakehouse provider {endpoint}: {ex}"
            ) from ex

        elapsed_ms = int((time.monotonic() - started) * 1000)
        _logger.debug(
            "POST %s status=%s elapsed_ms=%s req_id=%s",
            endpoint,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return self._handle_status(response, endpoint, request_id)

    def _build_headers(self, accept: str, request_id: str) -> dict[str, str]:
        """Compose the per-request header set."""
        headers = {
            "gz-site": self._site,
            "Content-Type": "application/json",
            "Accept": accept,
            "x-gz-request-id": request_id,
        }
        if self._enable_compression:
            headers["Accept-Encoding"] = "gzip, deflate, zstd"
        return headers

    @staticmethod
    def _handle_status(
        response: requests.Response,
        endpoint: str,
        request_id: str,
    ) -> HttpResponse:
        """Translate non-success status codes into typed SDK exceptions."""
        status = response.status_code
        if 200 <= status < 300:
            return HttpResponse(response)

        body_preview = response.text[:512] if response.content else ""
        try:
            if status == 401:
                raise AuthenticationError(
                    "Authentication failed against lakehouse provider"
                )
            if status == 403:
                raise AuthorizationError("Access denied by lakehouse provider")
            if status == 404:
                raise QueryError(f"Provider endpoint not found: {endpoint}")
            if status in _TRANSPORT_STATUS:
                raise TransportError(
                    f"Provider returned HTTP {status} after retries "
                    f"(req_id={request_id}): {body_preview}"
                )
            raise QueryExecutionError(
                f"Provider returned HTTP {status} "
                f"(req_id={request_id}): {body_preview}"
            )
        finally:
            response.close()

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._session.close()
