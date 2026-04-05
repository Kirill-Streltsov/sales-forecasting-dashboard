"""Load and clean the raw Superstore orders, and build daily sales series.

The raw file is the classic US "Sample - Superstore" extract: 9,994 order lines
from 2014 to 2017 with one row per product in an order. A few things are worth
handling explicitly:

* ``Order Date`` and ``Ship Date`` arrive as ``M/D/YYYY`` strings -> parsed to
  real datetimes so we can resample and compare years.
* ``Sales``, ``Profit``, ``Quantity`` and ``Discount`` are numeric but read as
  objects when a stray thousands separator sneaks in -> coerced to numbers.
* For forecasting we need a continuous daily series, so days with no orders are
  filled with zero sales rather than left as gaps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting import config

# Numeric columns that must be real numbers before any aggregation.
_NUMERIC_COLS = [
    config.SALES_COL,
    config.PROFIT_COL,
    config.QUANTITY_COL,
    "Discount",
]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSV exactly as shipped, no transformations."""
    # The dataset is latin-1 encoded (a few product names carry accented
    # characters); pandas defaults to utf-8 and would choke on them.
    return pd.read_csv(path or config.DATA_RAW, encoding="latin-1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy of the raw frame ready for aggregation."""
    df = df.copy()

    for col in (config.DATE_COL, config.SHIP_DATE_COL):
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")

    for col in _NUMERIC_COLS:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # An order line without a date or a sales figure cannot be aggregated.
    df = df.dropna(subset=[config.DATE_COL, config.SALES_COL]).reset_index(drop=True)

    return df


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """Convenience wrapper: load the raw file and clean it in one call."""
    return clean(load_raw(path))


def filter_frame(
    df: pd.DataFrame,
    region: str | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    """Filter orders by region and/or category.

    ``None`` or the ``config.ALL`` sentinel means "keep every value" for that
    dimension, which is how the dashboard's "All regions" option behaves.
    """
    mask = pd.Series(True, index=df.index)
    if region and region != config.ALL:
        mask &= df[config.REGION_COL] == region
    if category and category != config.ALL:
        mask &= df[config.CATEGORY_COL] == category
    return df[mask]


def daily_sales(
    df: pd.DataFrame,
    region: str | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    """Total sales per calendar day as a continuous, gap-free series.

    Returns a frame with ``date`` and ``sales`` columns. Every day between the
    first and last order is present; days without orders get zero sales so the
    Prophet model sees a regular daily frequency.
    """
    subset = filter_frame(df, region, category)
    daily = (
        subset.groupby(subset[config.DATE_COL].dt.normalize())[config.SALES_COL]
        .sum()
        .rename("sales")
    )

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    daily.index.name = "date"

    return daily.reset_index()


def to_prophet(daily: pd.DataFrame) -> pd.DataFrame:
    """Rename a ``daily_sales`` frame to Prophet's ``ds`` / ``y`` convention."""
    return daily.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
