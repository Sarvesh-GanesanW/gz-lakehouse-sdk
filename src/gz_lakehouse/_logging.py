"""Logging helpers for the gz-lakehouse SDK.

The SDK uses a single namespaced logger so callers can route or silence
its output via standard :mod:`logging` configuration. No handlers are
attached by default; the SDK adds a :class:`logging.NullHandler` to the
package logger to avoid the "no handler" warning on import.
"""

from __future__ import annotations

import logging
import uuid

_PACKAGE_LOGGER = logging.getLogger("gz_lakehouse")
if not _PACKAGE_LOGGER.handlers:
    _PACKAGE_LOGGER.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the SDK-namespaced logger for ``name``.

    Args:
        name: A short, human-readable component name (``http``,
            ``transport``, ``client``). It is appended to the
            ``gz_lakehouse`` namespace.

    Returns:
        A :class:`logging.Logger` instance under ``gz_lakehouse.<name>``.
    """
    return logging.getLogger(f"gz_lakehouse.{name}")


def new_request_id() -> str:
    """Return a short request id used to correlate client and server logs."""
    return uuid.uuid4().hex[:12]
