"""Smallest possible end-to-end example: read a table into pandas.

Run with credentials from your GroundZero tenant:

    pip install gz-lakehouse[pandas]
    python examples/simple_read.py
"""

import gz_lakehouse


def main() -> None:
    """Connect to a lakehouse provider and print the first rows."""
    with gz_lakehouse.connect(
        lakehouse_url=(
            "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
        ),
        siteName="admin",
        warehouse="my_warehouse",
        database="sales",
        username="user@example.com",
        password="REDACTED",
    ) as conn:
        result = conn.execute("SELECT * FROM sales.orders LIMIT 100").result
        print(f"rows={result.total_rows} truncated={result.truncated}")
        print(result.to_pandas().head())


if __name__ == "__main__":
    main()
