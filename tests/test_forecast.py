"""Tests for the Prophet forecast wrapper.

These fit Prophet on the small synthetic series from conftest with a short
horizon, so they exercise the real model without being slow.
"""

from __future__ import annotations

import math

import pandas as pd

from forecasting import forecast

HORIZON = 30


def test_forecast_frame_shape_and_flag(synthetic_daily):
    fc = forecast.forecast_frame(synthetic_daily, horizon_days=HORIZON)

    assert list(fc.columns) == [
        "ds", "actual", "yhat", "yhat_lower", "yhat_upper", "is_forecast",
    ]
    assert len(fc) == len(synthetic_daily) + HORIZON
    assert fc["is_forecast"].sum() == HORIZON
    # History carries the actuals; the future does not.
    assert fc.loc[fc["is_forecast"], "actual"].isna().all()
    assert fc.loc[~fc["is_forecast"], "actual"].notna().all()


def test_forecast_bounds_are_ordered_and_non_negative(synthetic_daily):
    fc = forecast.forecast_frame(synthetic_daily, horizon_days=HORIZON)
    assert (fc["yhat_lower"] <= fc["yhat"] + 1e-6).all()
    assert (fc["yhat"] <= fc["yhat_upper"] + 1e-6).all()
    assert (fc[["yhat", "yhat_lower", "yhat_upper"]] >= 0).all().all()


def test_forecast_extends_beyond_history(synthetic_daily):
    fc = forecast.forecast_frame(synthetic_daily, horizon_days=HORIZON)
    last_observed = synthetic_daily["ds"].max()
    future_dates = fc.loc[fc["is_forecast"], "ds"]
    assert future_dates.min() > last_observed
    assert (future_dates.max() - last_observed).days == HORIZON


def test_backtest_returns_finite_scores(synthetic_daily):
    scores = forecast.backtest(synthetic_daily, horizon_days=HORIZON)
    for key in ("mae", "rmse", "wape_pct", "total_error_pct"):
        assert math.isfinite(scores[key])
    assert scores["mae"] >= 0
    assert scores["horizon_days"] == HORIZON


def test_backtest_rejects_short_series():
    short = pd.DataFrame(
        {"ds": pd.date_range("2020-01-01", periods=40, freq="D"), "y": range(40)}
    )
    try:
        forecast.backtest(short, horizon_days=HORIZON)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a too-short series")
