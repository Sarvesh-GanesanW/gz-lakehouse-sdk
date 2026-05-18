# gz-lakehouse

Python connector for GroundZero Lakehouse. It lets Python applications query
Iceberg tables over HTTPS without running Spark locally, and it returns results
as Arrow, pandas, Spark DataFrames, or plain Python rows.

The connector uses a Snowflake-style data plane: a GroundZero provider runs the
statement on warm lakehouse compute, writes Arrow IPC chunks to S3, and returns
small metadata events with presigned URLs. The SDK downloads those chunks in
parallel and exposes a familiar `connect().cursor().execute()` workflow.

## Install

```bash
pip install gz-lakehouse
pip install "gz-lakehouse[pandas]"
pip install "gz-lakehouse[spark]"
```

## Quickstart

```python
import gz_lakehouse

with gz_lakehouse.connect(
    lakehouse_url="https://dev-admin-icebergprovider.dev.api.groundzerodev.cloud",
    siteName="admin",
    warehouse="TestGZWarehouse",
    database="lakehouse_db",
    username="user@example.com",
    password="****",
) as conn:
    cur = conn.cursor()
    cur.execute("SELECT * FROM lakehouse_db.orders LIMIT 1000")

    df = cur.fetch_pandas_all()
```

`siteName` is required. The SDK does not extract the tenant from
`lakehouse_url`. The value is sent as the provider's `gz-site` header, so the
caller must pass the tenant/site intentionally.

## Connector API

The top-level connector API is the recommended shape for application code.

```python
import gz_lakehouse

conn = gz_lakehouse.connect(
    lakehouse_url="https://dev-admin-icebergprovider.dev.api.groundzerodev.cloud",
    siteName="admin",
    warehouse="TestGZWarehouse",
    database="lakehouse_db",
    username="user@example.com",
    password="****",
)

try:
    cur = conn.cursor()
    cur.execute("SELECT region, count(*) AS rows FROM lakehouse_db.orders GROUP BY region")

    rows = cur.fetchall()
    arrow_table = cur.fetch_arrow_all()
    pandas_df = cur.fetch_pandas_all()
finally:
    conn.close()
```

The connection owns one warm compute session. Multiple cursors on the same
connection reuse that session until `conn.close()` is called.

| API | Purpose |
| --- | --- |
| `gz_lakehouse.connect(...)` | Create a connector-style connection. |
| `conn.cursor()` | Create a cursor bound to the connection's warm session. |
| `cursor.execute(sql)` | Execute SQL and keep the result on the cursor. |
| `cursor.fetchone()` | Fetch one remaining row as a mapping. |
| `cursor.fetchmany(size)` | Fetch up to `size` remaining rows. |
| `cursor.fetchall()` | Fetch all remaining rows. |
| `cursor.fetch_arrow_all()` | Return the complete result as a `pyarrow.Table`. |
| `cursor.fetch_pandas_all()` | Return the complete result as a pandas DataFrame. |
| `cursor.description` | DB-API-style column metadata. |
| `cursor.rowcount` | Provider-reported row count for the current result. |

## Client API

The lower-level client API is still available when you want explicit control
over session objects.

```python
from gz_lakehouse import LakehouseClient

with LakehouseClient.from_kwargs(
    lakehouse_url="https://dev-admin-icebergprovider.dev.api.groundzerodev.cloud",
    siteName="admin",
    warehouse="TestGZWarehouse",
    database="lakehouse_db",
    username="user@example.com",
    password="****",
) as client:
    with client.start_session() as session:
        result = session.query("SELECT count(*) FROM lakehouse_db.orders")
        print(result.to_list())
```

Use the lower-level client when you need `query_parallel`, `iter_batches`, or
fine-grained executor/pipeline tuning.

## Environment Variables

`LakehouseClient.from_env()` reads these variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `GZ_LAKEHOUSE_URL` | Yes | Provider base URL. |
| `GZ_LAKEHOUSE_SITE_NAME` | Yes | Tenant/site sent as the `gz-site` header. |
| `GZ_LAKEHOUSE_WAREHOUSE` | Yes | Iceberg warehouse/catalog name. |
| `GZ_LAKEHOUSE_DATABASE` | Yes | Default database/namespace. |
| `GZ_LAKEHOUSE_USERNAME` | Yes | Lakehouse user. |
| `GZ_LAKEHOUSE_PASSWORD` | Yes | Lakehouse password. |

`GZ_LAKEHOUSE_SITE` is accepted as a compatibility alias, but new code should
use `GZ_LAKEHOUSE_SITE_NAME`.

## Data Plane

```text
Python application
  |
  | POST /iceberg/startsession
  |   { computeId, minimumWorkers, connectionConfig }
  v
GroundZero provider starts a warm compute session
  |
  | POST /iceberg/v1/statements
  |   { sessionId, query, connectionConfig, executor?, pipelineConfig? }
  v
Session pod executes SQL and writes Arrow IPC chunks to S3
  |
  | schema/chunk/done NDJSON events
  | chunk events contain inline Arrow bytes or presigned S3 URLs
  v
SDK downloads chunks in parallel and builds a pyarrow.Table
```

For long-running statements the provider may emit:

```json
{"type": "deferred", "queryId": "TestGZWarehouse/<queryExecutionId>"}
```

The SDK then polls:

```text
GET /iceberg/v1/statements/TestGZWarehouse/<queryExecutionId>
```

until the manifest is available and the provider replays the same
schema/chunk/done event stream.

## Result Conversion

```python
result = conn.execute("SELECT * FROM lakehouse_db.orders LIMIT 1000").result

arrow_table = result.to_arrow()
pandas_df = result.to_pandas()
rows = result.to_list()
```

For Spark:

```python
spark_df = result.to_spark(spark)
```

For larger results, write to Parquet and let Spark read in parallel:

```python
spark_df = result.to_spark_via_parquet(
    spark,
    "s3://my-staging-bucket/gz-results/query-123/",
)
```

## Streaming Large Results

```python
from gz_lakehouse import LakehouseClient

with LakehouseClient.from_env() as client:
    with client.start_session() as session:
        for batch in session.iter_batches(
            "SELECT * FROM lakehouse_db.orders",
            batch_size=131_072,
        ):
            process(batch)
```

`iter_batches` yields `pyarrow.RecordBatch` objects in result order. At most
`parallel_workers` chunks are downloaded at once.

## Parallel Partitioned Queries

```python
with LakehouseClient.from_env() as client:
    result = client.query_parallel(
        sql_template="SELECT * FROM lakehouse_db.orders",
        partition_column="order_id",
        bounds=[
            (0, 99_999),
            (100_000, 199_999),
            (200_000, 299_999),
        ],
        max_workers=3,
    )
```

Each partition is submitted as its own statement and the Arrow tables are
concatenated in submission order.

## Performance Knobs

All knobs are available on `LakehouseConfig` and through
`LakehouseClient.from_kwargs`.

| Field | Default | Purpose |
| --- | --- | --- |
| `parallel_workers` | `32` | Concurrent chunk downloads and `query_parallel` fan-out. |
| `pool_connections` | `4` | HTTP connection pools kept by the requests session. |
| `pool_maxsize` | `64` | Maximum connections per pool. |
| `connect_timeout_seconds` | `10` | TCP/TLS handshake timeout. |
| `query_timeout_seconds` | `900` | Per-request read timeout. |
| `max_retries` | `3` | Retries for `408`, `429`, and common `5xx` responses. |
| `enable_http2` | `False` | Use HTTP/2 for S3 chunk downloads when enabled. |

Throughput scales with the number of chunks, available client bandwidth, and
S3 read throughput. Small result latency is dominated by session startup unless
you reuse a connection/session.

## Provider Contract

The current SDK expects the provider to expose these endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/iceberg/testconnection` | Verify credentials and warehouse access. |
| `POST` | `/iceberg/startsession` | Start a warm compute session and return `sessionId`. |
| `POST` | `/iceberg/v1/statements` | Execute SQL on an existing session. |
| `GET` | `/iceberg/v1/statements/<warehouse>/<queryExecutionId>` | Poll deferred statement results. |
| `POST` | `/iceberg/stopsession` | Stop the warm compute session. |

`POST /iceberg/v1/statements` sends:

```json
{
  "sessionId": "<session-id>",
  "connectionConfig": {
    "config": {
      "userName": "<username>",
      "password": "<password>",
      "warehouseName": "<warehouse>"
    }
  },
  "query": "SELECT * FROM lakehouse_db.orders",
  "executor": "fast",
  "pipelineConfig": {}
}
```

The provider may return either a JSON envelope or an NDJSON event stream. Chunk
events must contain one of:

```json
{"type": "chunk", "url": "https://presigned-s3-url", "rowCount": 1000}
```

or:

```json
{"type": "chunk", "inline": "<base64-arrow-ipc>", "rowCount": 1000}
```

## REST Catalog Scope

This SDK is not Spark or DuckDB's Iceberg REST catalog client. Spark and DuckDB
use `/iceberg/v1/config`, namespace, table, view, commit, and credential
endpoints directly through their Iceberg catalog implementations.

`gz-lakehouse` is the application data connector. It uses GroundZero's
statement API to execute SQL on GroundZero compute and return result chunks to
Python clients.

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
pytest
```

## Roadmap

- Async connector API.
- Server-side cancellation for running statements.
- Upload/write helpers for DataFrame-to-Iceberg workflows.
- Optional OpenTelemetry tracing.
- Catalog metadata helper methods for namespaces, tables, views, and tags.
