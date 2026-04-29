"""Configuration dataclass for the gz-lakehouse SDK.

Holds everything needed to address a GroundZero Lakehouse provider and
authenticate against it. Validates the URL and derives the ``site``
component used by the provider's middleware to route requests.

Performance knobs (``parallel_workers``, ``pool_maxsize``,
``prefer_arrow``) tune the new high-throughput data plane introduced in
``0.2.0``. They can be left at their defaults for plug-and-play use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal
from urllib.parse import urlparse

from gz_lakehouse.exceptions import ConfigurationError

_PASSWORD_REDACTED = "***"

ComputeSize = Literal["small", "medium", "large", "xlarge", "2xlarge"]
_VALID_COMPUTE_SIZES: frozenset[str] = frozenset(
    {"small", "medium", "large", "xlarge", "2xlarge"},
)


@dataclass(frozen=True)
class LakehouseConfig:
    """Immutable configuration for a :class:`LakehouseClient`.

    Attributes:
        lakehouse_url: Provider endpoint url, e.g.
            ``http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud``.
        warehouse: Iceberg warehouse name (catalog).
        database: Iceberg database (namespace).
        username: Cognito username for the lakehouse tenant.
        password: Cognito password for the lakehouse tenant.
        site: Optional explicit ``gz-site`` header value. If omitted the
            site is derived from ``lakehouse_url`` by taking the second
            hyphen-separated part of the host's first label
            (``dev-admin-icebergprovider`` → ``admin``). The server then
            combines this with its own ``ENV`` to form the full
            ``<env>-<site>`` identifier used to route requests.
        compute_size: T-shirt-size compute selector. One of
            ``small`` / ``medium`` / ``large`` / ``xlarge`` / ``2xlarge``.
            The provider maps the name to a concrete compute id
            (currently 1003 / 1006 / 1009 / 1012 / 1015), each backed
            by a fixed memory + vCPU spec maintained server-side
            (small ≈ 8 GB, 2xlarge ≈ 64 GB at the time of writing).
            Defaults to ``small``.
        compute_id: Raw compute id escape hatch. When set, overrides
            ``compute_size``. Useful only when the provider exposes a
            compute id not yet covered by the t-shirt vocabulary.
        verify_timeout_seconds: Timeout for the provider verification call.
        query_timeout_seconds: Timeout for synchronous query execution.
        connect_timeout_seconds: TCP/TLS handshake timeout enforced
            independently from ``query_timeout_seconds``.
        prefer_arrow: When ``True`` (default) the client advertises
            Arrow IPC and chunked-Parquet content types via ``Accept`` and
            uses whichever the provider responds with. Falls back to the
            JSON envelope path automatically when the provider has not
            yet enabled the high-throughput data plane.
        parallel_workers: Worker count for the chunked-Parquet transport
            (parallel S3 fetches) and for :meth:`query_parallel`.
        pool_connections: Number of connection pools the underlying
            session keeps. Forwarded to ``HTTPAdapter``.
        pool_maxsize: Maximum number of connections kept inside each
            pool. Forwarded to ``HTTPAdapter``.
        max_retries: Retry attempts for transient HTTP failures.
        backoff_seconds: Base for exponential backoff between retries.
        enable_compression: When ``True`` (default) the client sends
            ``Accept-Encoding: gzip, deflate, zstd`` so a compression-aware
            load balancer can shrink JSON payloads on the wire.
        enable_http2: When ``True``, the chunk-download path uses an
            ``httpx`` client with HTTP/2 enabled instead of the
            ``requests``-backed session. HTTP/2 multiplexes parallel
            chunk fetches over a single TCP connection, eliminating the
            TLS-handshake-per-chunk and slow-start overhead of HTTP/1.1
            (typical 5-10% wall reduction on 100M+ row pulls when the
            S3 endpoint supports h2). Defaults to ``False`` while the
            new path bakes; flip to ``True`` for production after
            verifying in your environment.
    """

    lakehouse_url: str
    warehouse: str
    database: str
    username: str
    password: str
    site: str | None = None
    compute_size: ComputeSize = "small"
    compute_id: int | None = None
    minimum_workers: int = 1
    verify_timeout_seconds: int = 30
    query_timeout_seconds: int = 900
    connect_timeout_seconds: int = 10
    prefer_arrow: bool = True
    parallel_workers: int = 32
    pool_connections: int = 4
    pool_maxsize: int = 64
    max_retries: int = 3
    backoff_seconds: float = 0.5
    enable_compression: bool = True
    enable_http2: bool = False

    derived_site: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate required fields and derive the ``gz-site`` value."""
        for name in (
            "lakehouse_url",
            "warehouse",
            "database",
            "username",
            "password",
        ):
            value = getattr(self, name)
            if not value or not isinstance(value, str):
                raise ConfigurationError(
                    f"LakehouseConfig.{name} is required and must be a string"
                )

        for name in (
            "verify_timeout_seconds",
            "query_timeout_seconds",
            "connect_timeout_seconds",
            "parallel_workers",
            "pool_connections",
            "pool_maxsize",
            "max_retries",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ConfigurationError(
                    f"LakehouseConfig.{name} must be a non-negative int"
                )

        if self.compute_size not in _VALID_COMPUTE_SIZES:
            allowed = sorted(_VALID_COMPUTE_SIZES)
            raise ConfigurationError(
                f"LakehouseConfig.compute_size must be one of {allowed}",
            )

        if self.compute_id is not None and (
            not isinstance(self.compute_id, int) or self.compute_id < 0
        ):
            raise ConfigurationError(
                "LakehouseConfig.compute_id must be a non-negative int "
                "or None",
            )

        if self.parallel_workers == 0:
            raise ConfigurationError(
                "LakehouseConfig.parallel_workers must be at least 1"
            )

        if (
            not isinstance(self.minimum_workers, int)
            or self.minimum_workers < 1
        ):
            raise ConfigurationError(
                "LakehouseConfig.minimum_workers must be an int >= 1"
            )

        site_value = self.site or self._derive_site_from_url(
            self.lakehouse_url
        )
        if not site_value:
            raise ConfigurationError(
                "Could not derive 'site' from lakehouse_url; "
                "pass `site` explicitly"
            )
        object.__setattr__(self, "derived_site", site_value)

    def __repr__(self) -> str:
        """Return a representation that redacts the password."""
        safe = replace(self, password=_PASSWORD_REDACTED)
        return (
            f"LakehouseConfig(lakehouse_url={safe.lakehouse_url!r}, "
            f"warehouse={safe.warehouse!r}, database={safe.database!r}, "
            f"username={safe.username!r}, password={_PASSWORD_REDACTED!r}, "
            f"site={safe.site!r}, compute_id={safe.compute_id}, "
            f"prefer_arrow={safe.prefer_arrow}, "
            f"parallel_workers={safe.parallel_workers})"
        )

    @classmethod
    def from_env(
        cls,
        prefix: str = "GZ_LAKEHOUSE_",
        **overrides: object,
    ) -> LakehouseConfig:
        """Build a config from environment variables.

        Environment variables follow the pattern ``<prefix><FIELD>`` in
        upper case (e.g. ``GZ_LAKEHOUSE_URL``,
        ``GZ_LAKEHOUSE_WAREHOUSE``, ``GZ_LAKEHOUSE_USERNAME``,
        ``GZ_LAKEHOUSE_PASSWORD``). The lakehouse url uses the special
        suffix ``URL`` instead of ``LAKEHOUSE_URL`` to avoid the
        repetitive ``GZ_LAKEHOUSE_LAKEHOUSE_URL``.

        Args:
            prefix: Environment variable prefix.
            **overrides: Keyword overrides applied after the env lookup,
                useful for tests or callers that want to mix sources.

        Returns:
            A validated :class:`LakehouseConfig`.

        Raises:
            ConfigurationError: When a required variable is missing.
        """
        env_map = {
            "lakehouse_url": f"{prefix}URL",
            "warehouse": f"{prefix}WAREHOUSE",
            "database": f"{prefix}DATABASE",
            "username": f"{prefix}USERNAME",
            "password": f"{prefix}PASSWORD",
            "site": f"{prefix}SITE",
        }
        kwargs: dict[str, object] = {}
        for field_name, env_name in env_map.items():
            value = os.environ.get(env_name)
            if value is not None:
                kwargs[field_name] = value
        kwargs.update(overrides)
        try:
            return cls(**kwargs)
        except TypeError as ex:
            raise ConfigurationError(
                f"Missing required environment variable for "
                f"LakehouseConfig: {ex}"
            ) from ex

    @staticmethod
    def _derive_site_from_url(url: str) -> str:
        """Extract the site identifier from a provider URL.

        The provider URL host has the shape
        ``<env>-<site>-<service>.<env>.api.<root>.<tld>``. The
        ``gz-site`` header expects just the ``<site>`` value, which is
        the second hyphen-separated part of the host's first label;
        the server's middleware combines it with the pod's ``ENV`` to
        form the full ``<env>-<site>`` identifier used for matching.

        Pass ``site`` explicitly to :class:`LakehouseConfig` to override
        this derivation when the URL does not follow the convention.
        """
        host = urlparse(url).hostname or ""
        first_label = host.split(".")[0] if host else ""
        if not first_label:
            return ""
        parts = first_label.split("-")
        if len(parts) < 2:
            return ""
        return parts[1]
