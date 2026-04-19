"""Tests for loading, cleaning and shaping the orders."""

from __future__ import annotations

import pandas as pd

from forecasting import config, data


def test_clean_frame_types_and_completeness(clean_df):
    assert len(clean_df) == 9994
    assert pd.api.types.is_datetime64_any_dtype(clean_df[config.DATE_COL])
    assert pd.api.types.is_numeric_dtype(clean_df[config.SALES_COL])
    # Cleaning drops rows that cannot be aggregated.
    assert clean_df[config.DATE_COL].notna().all()
    assert clean_df[config.SALES_COL].notna().all()


def test_expected_regions_and_categories(clean_df):
    assert sorted(clean_df[config.REGION_COL].unique()) == config.REGIONS
    assert sorted(clean_df[config.CATEGORY_COL].unique()) == config.CATEGORIES


def test_daily_sales_is_gap_free(clean_df):
    daily = data.daily_sales(clean_df)
    # A continuous daily index means every step is exactly one day.
    gaps = daily["date"].diff().dropna().dt.days
    assert (gaps == 1).all()
    assert (daily["sales"] >= 0).all()


def test_filter_frame_by_region_and_category(tiny_orders):
    west = data.filter_frame(tiny_orders, region="West")
    assert set(west[config.REGION_COL]) == {"West"}

    tech = data.filter_frame(tiny_orders, category="Technology")
    assert set(tech[config.CATEGORY_COL]) == {"Technology"}

    # The ALL sentinel and None both mean "no filter".
    assert len(data.filter_frame(tiny_orders, region=config.ALL)) == len(tiny_orders)
    assert len(data.filter_frame(tiny_orders, region=None)) == len(tiny_orders)


def test_daily_sales_filtered_sums_match(tiny_orders):
    west_tech = data.daily_sales(tiny_orders, region="West", category="Technology")
    # West + Technology in the fixture is 200 (2016) + 500 (2017) = 700.
    assert west_tech["sales"].sum() == 700.0


def test_to_prophet_columns(tiny_orders):
    pf = data.to_prophet(data.daily_sales(tiny_orders))
    assert list(pf.columns) == ["ds", "y"]
