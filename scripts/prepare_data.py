"""Clean the raw Superstore orders and write the processed data used downstream.

Outputs (into ``data/processed/``, which is git-ignored and regenerated):

    orders_clean.csv    the typed order lines
    daily_sales.csv     total sales per calendar day, gap-free

Run from the project root:

    python scripts/prepare_data.py
"""

from __future__ import annotations

from forecasting import config, data


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = data.load_clean()
    orders_path = config.DATA_PROCESSED / "orders_clean.csv"
    df.to_csv(orders_path, index=False)

    daily = data.daily_sales(df)
    daily_path = config.DATA_PROCESSED / "daily_sales.csv"
    daily.to_csv(daily_path, index=False)

    start = df[config.DATE_COL].min()
    end = df[config.DATE_COL].max()
    print(f"Cleaned {len(df):,} order lines ({start:%Y-%m-%d} to {end:%Y-%m-%d}).")
    print(f"  wrote {orders_path.relative_to(config.ROOT)}")
    print(f"  wrote {daily_path.relative_to(config.ROOT)} ({len(daily):,} days)")


if __name__ == "__main__":
    main()
