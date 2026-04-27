"""ETL example: pull rows from the lakehouse, transform, write to a CSV.

Demonstrates how the SDK plugs into a downstream pandas pipeline.
"""

from pathlib import Path

from gz_lakehouse import LakehouseClient


def main() -> None:
    """Read lakehouse data and persist a transformed CSV."""
    with LakehouseClient.from_kwargs(
        lakehouse_url=(
            "http://dev-admin-icebergprovider.dev.api.groundzerodev.cloud"
        ),
        warehouse="my_warehouse",
        database="sales",
        username="user@example.com",
        password="REDACTED",
    ) as client:
        df = client.query(
            "SELECT region, sum(amount) AS total "
            "FROM sales.orders GROUP BY region"
        ).to_pandas()

    df["total_thousands"] = df["total"] / 1_000
    out_path = Path("region_totals.csv")
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
