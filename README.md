# gz-lakehouse

Plug-and-play Python SDK for the GroundZero Lakehouse — query Iceberg tables over HTTPS from any Python environment, no Spark or JDBC required.

`0.3.0` introduces an explicit **session API** — start a warm compute pod once, run many statements against it, stop when done. The data plane stays Snowflake-style: the provider hands back presigned S3 URLs to Arrow IPC chunks, and the client downloads them in parallel.

## Install

```bash
pip install gz-lakehouse              # core
pip install "gz-lakehouse[pandas]"    # adds pandas DataFrame conversion
pip install "gz-lakehouse[spark]"     # adds Spark DataFrame conversion
```

## Quickstart

```python
from gz_lakehouse import LakehouseClient

with LakehouseClient.from_kwargs(
    lakehouse_url="http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud",
    warehouse="my_warehouse",
    database="sales",
    username="user@example.com",
    password="****",
) as client:

    # Recommended: explicit session, reuses the warm pod across queries.
    with client.start_session() as session:
        df = session.query("SELECT * FROM sales.orders LIMIT 1000").to_pandas()
        arrow = session.query("SELECT * FROM customers LIMIT 50").to_arrow()
        for batch in session.iter_batches(
            "SELECT * FROM sales.orders WHERE year = 2025",
        ):
            ...

    # One-off convenience: auto-creates and stops a session per call.
    df = client.query("SELECT count(*) FROM sales.orders").to_pandas()
```

Or pull credentials from the environment:

```python
import os
os.environ["GZ_LAKEHOUSE_URL"] = "..."
os.environ["GZ_LAKEHOUSE_WAREHOUSE"] = "..."
os.environ["GZ_LAKEHOUSE_DATABASE"] = "..."
os.environ["GZ_LAKEHOUSE_USERNAME"] = "..."
os.environ["GZ_LAKEHOUSE_PASSWORD"] = "..."

with LakehouseClient.from_env() as client, client.start_session() as session:
    result = session.query("SELECT count(*) FROM sales.orders")
```

The client picks `gz-site` automatically from the URL host (the second hyphen-segment of the first label, i.e. `<env>-<site>-<service>` → `<site>`) and verifies the provider on the first call. The password is redacted from `repr(config)` so it never leaks into stack traces or logs.

### Session vs. one-off

| Mode | Cost | When to use |
|---|---|---|
| `with client.start_session() as session: ...` | Pod boot once (~17 s), every query after that is fast | Notebooks, ETL with many statements, anything interactive |
| `client.query(sql)` | Pod boot **per call** | One-off scripts, smoke tests, stateless workers |

Sessions are explicit so the cost is visible; the convenience wrappers exist so plug-and-play users don't have to think about it.

## Public API

| Symbol                 | Purpose                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| `LakehouseClient`      | Connect, verify, start sessions, plus one-off convenience wrappers        |
| `Session`              | Warm compute pod; run queries, stream batches, fan-out partitioned queries |
| `LakehouseConfig`      | Frozen dataclass for connection params + perf knobs                       |
| `QueryResult`          | Wraps an Arrow table + schema; converts to pandas / Spark / list          |
| `GzLakehouseError`     | Base exception                                                            |
| `AuthenticationError`  | HTTP 401 from provider                                                    |
| `AuthorizationError`   | HTTP 403 from provider                                                    |
| `QueryError`           | Parent for query failures (kept for back-compat)                          |
| `QueryValidationError` | Caller-side query problem (never retried)                                 |
| `QueryExecutionError`  | Provider-side execution failure                                           |
| `TransportError`       | Network failure or timeout                                                |
| `ConfigurationError`   | Invalid configuration                                                     |

## Data plane

```
   Your Python code
         │
         │ POST /iceberg/v1/statements   { query, ... }
         ▼
   GroundZero provider
         │
         │ executes SQL on ephemeral compute, writes the result to S3 as
         │ Arrow IPC chunks (~20 MB each, zstd-compressed at the IPC body
         │ level), returns a small JSON envelope with presigned URLs:
         │
         │ { schema, totalRecords, hasMore, chunks: [{url, rowCount, ...}] }
         ▼
   Client
         │ ThreadPoolExecutor (parallel_workers)
         │   ├── GET <presigned-S3-url-0>  ──► pa.ipc.open_stream → pa.Table
         │   ├── GET <presigned-S3-url-1>  ──► pa.ipc.open_stream → pa.Table
         │   └── ...
         │
         ▼ pa.concat_tables
     pa.Table → pandas / Spark / list
```

Each chunk is parsed straight into a `pyarrow.Table` with no Python-object intermediate. Concatenation is constant time (Arrow tables share buffers), so the only real work the client does is the network read.

### Throughput targets

| Workload                               | Target throughput          | Knob                                       |
| -------------------------------------- | -------------------------- | ------------------------------------------ |
| 8 workers, 20 MB chunks, 1 Gb/s link   | **~110 MB/s** (link-bound) | `parallel_workers=8`                       |
| 16 workers, 20 MB chunks, 10 Gb/s link | **~700 MB/s – 1 GB/s**     | `parallel_workers=16` + `pool_maxsize>=16` |
| Single-chunk small result              | **~50–100 ms** end to end  | first-chunk latency dominates              |

The numbers scale with `parallel_workers` up to the smaller of (a) the client's bandwidth, (b) the provider's S3 fan-out cap, and (c) the chunk count. Snowflake-class throughput is built in.

### Spark conversion

`result.to_spark(spark)` picks the fastest available conversion path automatically:

| PySpark version | Path                                                                            | Throughput       |
| --------------- | ------------------------------------------------------------------------------- | ---------------- |
| **3.4+**        | `spark.createDataFrame(arrow_table)` direct                                     | **100–200 MB/s** |
| **3.0–3.3**     | Arrow-enabled pandas (auto-toggles `spark.sql.execution.arrow.pyspark.enabled`) | 50–100 MB/s      |
| **< 3.0**       | Plain pandas (warned about)                                                     | 5–10 MB/s        |

```python
df = result.to_spark(spark)
```

Empty results keep their column metadata so downstream Spark code does not lose its schema.

For results larger than a few hundred MB, route through Parquet so Spark executors load in parallel instead of pushing every row through the Python driver:

```python
df = result.to_spark_via_parquet(spark, "s3://staging-bucket/qid-42/")
```

The Arrow table is written as zstd-compressed Parquet and the returned DataFrame is `spark.read.parquet(path)` — Spark fans the load out across executors.

### Streaming large results

```python
import pyarrow.parquet as pq

with LakehouseClient.from_env() as client:
    writer = None
    for batch in client.iter_batches(
        "SELECT * FROM sales.orders WHERE year = 2025",
        batch_size=131_072,
    ):
        if writer is None:
            writer = pq.ParquetWriter("orders.parquet", batch.schema)
        writer.write_batch(batch)
    if writer is not None:
        writer.close()
```

`iter_batches` yields `pyarrow.RecordBatch` objects in submission order. At most `parallel_workers` chunks are in flight at once, so memory stays bounded regardless of total result size.

### Parallel partitioned queries

`query_parallel` fans out range-partitioned subqueries across the same thread pool. Useful when a single query is heavier than the provider's per-statement budget:

```python
result = client.query_parallel(
    sql_template="SELECT * FROM sales.orders",
    partition_column="order_id",
    bounds=[(i * 100_000, (i + 1) * 100_000 - 1) for i in range(10)],
    max_workers=8,
)
df = result.to_pandas()
```

Each subquery hits `/iceberg/v1/statements` independently, downloads its own presigned chunks, and the resulting Arrow tables are concatenated in submission order.

### Performance knobs

All exposed on `LakehouseConfig`:

| Field                     | Default | Purpose                                                        |
| ------------------------- | ------- | -------------------------------------------------------------- |
| `parallel_workers`        | `8`     | Concurrent chunk downloads (and `query_parallel` fan-out)      |
| `pool_connections`        | `4`     | HTTP connection pools the session keeps                        |
| `pool_maxsize`            | `16`    | Max connections per pool (auto-bumped to ≥ `parallel_workers`) |
| `connect_timeout_seconds` | `10`    | TCP/TLS handshake timeout                                      |
| `query_timeout_seconds`   | `900`   | Per-request read timeout                                       |
| `max_retries`             | `3`     | Retries on 408/429/500/502/503/504 (respects `Retry-After`)    |
| `enable_compression`      | `True`  | Advertise `gzip, deflate, zstd` for the metadata envelope      |

### Logging

The SDK logs to the `gz_lakehouse` namespace with no handlers attached by default. Wire it up the standard way:

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("gz_lakehouse").setLevel(logging.DEBUG)
```

Each request is tagged with a short `x-gz-request-id` so client and provider logs can be correlated.

## Provider expectations

The data plane requires the provider to:

1. Expose `POST /iceberg/v1/statements` accepting `{ query, computeId, connectionConfig: { config: { userName, password, warehouseName } } }`.
2. Execute the query on ephemeral compute and write the result to S3 as **Arrow IPC streams** with zstd compression at the IPC body level, ~20 MB per chunk (compressed).
3. Return a small JSON envelope:
   ```json
   {
       "schema": [{"columnName": "id", "dataType": "BIGINT"}, ...],
       "totalRecords": 10500000,
       "hasMore": false,
       "chunks": [
           {
               "url": "https://s3.<region>.amazonaws.com/...&X-Amz-...",
               "rowCount": 200000,
               "compressedSize": 19834234,
               "uncompressedSize": 67234234
           },
           ...
       ]
   }
   ```
4. On query failure, return either an HTTP error with a JSON body, or `{ "status": "error", "message": "..." }` with HTTP 200.
5. Keep the existing `POST /iceberg/testconnection` for credential verification.

## Roadmap

- Async-native client (`AsyncLakehouseClient`) sharing the same transport layer.
- `client.write_table(df, ...)` write path uploading Arrow IPC chunks to S3.
- OpenTelemetry tracing as an optional extra.

The public API is stable across these upgrades — only the wire format and helpers evolve underneath.

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest
```
