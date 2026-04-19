"""Tests for the KPI and year-over-year measures."""

from __future__ import annotations

import pandas as pd

from forecasting import config, features


def test_headline_kpis(tiny_orders):
    # Sales: 100 + 200 + 300 + 500 + 100 = 1200; profit: 10 + 40 - 30 + 100 + 20.
    assert features.total_sales(tiny_orders) == 1200.0
    assert features.total_profit(tiny_orders) == 140.0
    assert features.profit_margin(tiny_orders) == 140.0 / 1200.0
    # Four distinct order ids across five product rows.
    assert features.order_count(tiny_orders) == 4
    assert features.avg_order_value(tiny_orders) == 300.0


def test_sales_by_is_sorted_descending(tiny_orders):
    by_cat = features.sales_by(tiny_orders, config.CATEGORY_COL)
    assert by_cat.index.tolist() == ["Technology", "Furniture"]
    assert by_cat.tolist() == [700.0, 500.0]


def test_yoy_table(tiny_orders):
    table = features.yoy_table(tiny_orders)
    assert table.index.tolist() == [2016, 2017]
    assert table["sales"].tolist() == [600.0, 600.0]
    # The first year has no prior year to compare against.
    assert pd.isna(table["yoy_pct"].iloc[0])
    # Both years total 600, so the most recent change is flat.
    assert features.latest_yoy_pct(tiny_orders) == 0.0


def test_latest_yoy_none_with_single_year():
    one_year = pd.DataFrame(
        {
            config.DATE_COL: pd.to_datetime(["2017-01-01", "2017-02-01"]),
            config.SALES_COL: [10.0, 20.0],
        }
    )
    assert features.latest_yoy_pct(one_year) is None


def test_kpis_keys(tiny_orders):
    k = features.kpis(tiny_orders)
    assert set(k) == {
        "total_sales", "total_profit", "profit_margin",
        "orders", "avg_order_value", "yoy_pct",
    }


def test_kpis_on_real_data(clean_df):
    # The published Superstore total is $2,297,200.86.
    assert round(features.total_sales(clean_df), 2) == 2297200.86
    assert features.order_count(clean_df) == 5009
