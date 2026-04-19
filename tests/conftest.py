"""Shared fixtures.

The real dataset loads once per session; the small synthetic frames keep the
feature and forecast tests fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import config, data


@pytest.fixture(scope="session")
def clean_df():
    return data.load_clean()


@pytest.fixture
def tiny_orders():
    """A hand-built order book with known totals across two years."""
    rows = [
        # date, order id, region, category, sub-category, segment, sales, profit
        ("1/5/2016", "A-1", "West", "Furniture", "Chairs", "Consumer", 100.0, 10.0),
        ("1/5/2016", "A-1", "West", "Technology", "Phones", "Consumer", 200.0, 40.0),
        ("6/1/2016", "A-2", "East", "Furniture", "Tables", "Corporate", 300.0, -30.0),
        ("2/1/2017", "A-3", "West", "Technology", "Phones", "Consumer", 500.0, 100.0),
        ("3/1/2017", "A-4", "East", "Furniture", "Chairs", "Home Office", 100.0, 20.0),
    ]
    cols = [
        config.DATE_COL, config.ORDER_ID_COL, config.REGION_COL,
        config.CATEGORY_COL, config.SUBCATEGORY_COL, config.SEGMENT_COL,
        config.SALES_COL, config.PROFIT_COL,
    ]
    df = pd.DataFrame(rows, columns=cols)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], format="%m/%d/%Y")
    return df


@pytest.fixture
def synthetic_daily():
    """A year and a half of daily sales with a clear weekly pattern.

    Long enough to fit Prophet and to hold out a short backtest horizon, small
    enough to stay fast.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=540, freq="D")
    weekday_lift = np.where(dates.dayofweek < 5, 1.0, 0.4)
    trend = np.linspace(100, 200, len(dates))
    noise = rng.normal(0, 8, len(dates))
    sales = np.clip(trend * weekday_lift + noise, 0, None)
    return pd.DataFrame({"ds": dates, "y": sales})
