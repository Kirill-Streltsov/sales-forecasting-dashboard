"""KPI and year-over-year aggregations.

These are the Python counterparts of the DAX measures in the Power BI report
(see ``powerbi/dax_measures.dax``). Keeping the same definitions in both places
means the Streamlit dashboard and the Power BI report show identical numbers.
"""

from __future__ import annotations

import pandas as pd

from forecasting import config


def total_sales(df: pd.DataFrame) -> float:
    return float(df[config.SALES_COL].sum())


def total_profit(df: pd.DataFrame) -> float:
    return float(df[config.PROFIT_COL].sum())


def profit_margin(df: pd.DataFrame) -> float:
    """Profit as a share of sales; 0 when there are no sales."""
    sales = total_sales(df)
    return total_profit(df) / sales if sales else 0.0


def order_count(df: pd.DataFrame) -> int:
    """Distinct orders. One order usually spans several product rows."""
    return int(df[config.ORDER_ID_COL].nunique())


def avg_order_value(df: pd.DataFrame) -> float:
    orders = order_count(df)
    return total_sales(df) / orders if orders else 0.0


def sales_by(df: pd.DataFrame, dimension: str) -> pd.Series:
    """Total sales grouped by a dimension, largest first."""
    return (
        df.groupby(dimension)[config.SALES_COL]
        .sum()
        .sort_values(ascending=False)
    )


def sales_by_year(df: pd.DataFrame) -> pd.Series:
    """Total sales per calendar year, indexed by year."""
    by_year = df.groupby(df[config.DATE_COL].dt.year)[config.SALES_COL].sum()
    by_year.index = by_year.index.astype(int)
    by_year.index.name = "year"
    return by_year


def yoy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sales per year with the absolute and percentage change on the year before.

    This is the ``Sales YoY %`` measure expressed as a small table: the first
    year has no prior year to compare against, so its change columns are NaN.
    """
    by_year = sales_by_year(df)
    table = by_year.to_frame("sales")
    table["prev_year_sales"] = table["sales"].shift(1)
    table["yoy_change"] = table["sales"] - table["prev_year_sales"]
    table["yoy_pct"] = table["yoy_change"] / table["prev_year_sales"] * 100
    return table


def latest_yoy_pct(df: pd.DataFrame) -> float | None:
    """Year-over-year growth for the most recent year, as a percentage.

    Returns ``None`` when there is only a single year of data.
    """
    table = yoy_table(df)
    if len(table) < 2:
        return None
    value = table["yoy_pct"].iloc[-1]
    return None if pd.isna(value) else float(value)


def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Sales resampled to month starts, for the trend chart."""
    monthly = (
        df.set_index(config.DATE_COL)[config.SALES_COL]
        .resample("MS")
        .sum()
        .rename("sales")
    )
    return monthly.reset_index().rename(columns={config.DATE_COL: "month"})


def kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    """The headline numbers behind the KPI cards, in one call."""
    return {
        "total_sales": total_sales(df),
        "total_profit": total_profit(df),
        "profit_margin": profit_margin(df),
        "orders": order_count(df),
        "avg_order_value": avg_order_value(df),
        "yoy_pct": latest_yoy_pct(df),
    }
