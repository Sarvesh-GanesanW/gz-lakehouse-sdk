"""Per-query tuning knobs for the server-side fast-path pipeline.

The fast path on the pod runs a three-stage producer/consumer
pipeline: pyarrow.dataset reader → encoder thread pool →
S3-upload thread pool. Default sizings work for the common workload
(medium pod, nested complex schema, 10M-100M rows). Callers who know
their workload — flat schema, very large pods, cross-region S3, very
small or very large results — can override individual knobs here so
the pipeline matches the data shape.

Every field is optional. ``None`` means "use the server default."
The SDK validates ranges client-side before the request leaves the
laptop, so a typo (``num_encoders=-1``) fails fast instead of after
a server round trip.

Wire format on ``POST /v1/statements``::

    {
        ...,
        "executor": "fast",
        "pipelineConfig": {
            "numEncoders": 8,
            "uploadWorkers": 32,
            "batchRows": 32768,
            "zstdLevel": 0
        }
    }

Server merges these on top of its own env-var defaults, which sit
on top of the code defaults. Anything the caller didn't set takes
the next-most-specific value.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from gz_lakehouse.exceptions import QueryValidationError


@dataclass(frozen=True)
class PipelineConfig:
    """Per-query overrides for the fast-path pipeline tuning.

    Attributes:
        num_encoders: Worker count in the encoder thread pool.
            Each encoder owns its own IPC sink + zstd writer and
            consumes Arrow batches off the shared queue. Bottleneck
            is CPU (zstd releases the GIL during native compress)
            so scale roughly with the pod's vCPU count. Range 1–32.
        upload_workers: Worker count in the S3 upload pool. Bound
            by S3 ingest per prefix; 16–32 is comfortable. Range 1–64.
        batch_queue_size: Bounded queue between reader and encoders.
            Larger smooths out reader/encoder rate mismatches at
            the cost of memory. Range 1–128.
        batch_rows: ``pyarrow.dataset.Scanner.batch_size``. Smaller
            batches = lower per-batch memory but more loop overhead.
            Range 1024–1048576.
        fragment_readahead: How many Parquet files the scanner reads
            concurrently. Tables with many small files benefit from
            higher values. Range 1–32.
        batch_readahead: Per-fragment batch buffer depth in pyarrow.
            Range 1–32.
        zstd_level: 0 disables compression entirely (fastest, biggest
            wire), 1 is the fastest compressing level, 3 is the
            pyarrow default, up to 22 for densest output. Range 0–22.
        chunk_bytes: Per-chunk uncompressed Arrow buffer threshold.
            Bigger chunks = fewer S3 PUTs but larger SDK download
            grain. Range 1 MB–1 GB.
    """

    num_encoders: int | None = None
    upload_workers: int | None = None
    batch_queue_size: int | None = None
    batch_rows: int | None = None
    fragment_readahead: int | None = None
    batch_readahead: int | None = None
    zstd_level: int | None = None
    chunk_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate ranges client-side so typos fail before the request."""
        for name, low, high in _BOUNDS:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise QueryValidationError(
                    f"PipelineConfig.{name} must be an int, "
                    f"got {type(value).__name__}",
                )
            if not low <= value <= high:
                raise QueryValidationError(
                    f"PipelineConfig.{name}={value} is outside "
                    f"the allowed range [{low}, {high}]",
                )

    def to_wire(self) -> dict[str, Any]:
        """Render to the JSON shape the server expects.

        Camel-case keys, only set fields included so the server can
        distinguish "caller didn't override" from "caller set to 0."
        """
        out: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            out[_FIELD_TO_WIRE[field.name]] = value
        return out


_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("num_encoders", 1, 32),
    ("upload_workers", 1, 64),
    ("batch_queue_size", 1, 128),
    ("batch_rows", 1024, 1_048_576),
    ("fragment_readahead", 1, 32),
    ("batch_readahead", 1, 32),
    ("zstd_level", 0, 22),
    ("chunk_bytes", 1024 * 1024, 1024 * 1024 * 1024),
)


_FIELD_TO_WIRE: dict[str, str] = {
    "num_encoders": "numEncoders",
    "upload_workers": "uploadWorkers",
    "batch_queue_size": "batchQueueSize",
    "batch_rows": "batchRows",
    "fragment_readahead": "fragmentReadahead",
    "batch_readahead": "batchReadahead",
    "zstd_level": "zstdLevel",
    "chunk_bytes": "chunkBytes",
}
