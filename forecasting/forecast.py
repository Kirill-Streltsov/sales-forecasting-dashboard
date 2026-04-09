"""The Prophet forecasting model.

The model works on a daily total-sales series and is set up the way the resume
describes it: weekly and yearly seasonality are modelled separately, and US
public holidays are added as their own component so spikes around, say,
Thanksgiving are not smeared into the ordinary weekly pattern.

Prophet is a training-time dependency only. The dashboard reads the exported
forecast CSV, so importing this module (and therefore Prophet) is never needed
to serve the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet

from forecasting import config, data


def build_model(
    interval_width: float = config.INTERVAL_WIDTH,
    country: str = config.HOLIDAY_COUNTRY,
) -> Prophet:
    """A Prophet model with weekly + yearly seasonality and holiday effects."""
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=interval_width,
    )
    # Holidays get their own regressors instead of leaking into the seasonalities.
    model.add_country_holidays(country_name=country)
    return model


def fit_forecast(
    prophet_df: pd.DataFrame,
    horizon_days: int = config.FORECAST_HORIZON_DAYS,
    interval_width: float = config.INTERVAL_WIDTH,
    country: str = config.HOLIDAY_COUNTRY,
) -> tuple[Prophet, pd.DataFrame]:
    """Fit the model and predict ``horizon_days`` beyond the last observation.

    ``prophet_df`` must have Prophet's ``ds`` / ``y`` columns (see
    ``data.to_prophet``). Returns the fitted model and the raw prediction frame.
    """
    model = build_model(interval_width=interval_width, country=country)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future)
    return model, forecast


def forecast_frame(
    prophet_df: pd.DataFrame,
    horizon_days: int = config.FORECAST_HORIZON_DAYS,
    interval_width: float = config.INTERVAL_WIDTH,
    country: str = config.HOLIDAY_COUNTRY,
) -> pd.DataFrame:
    """A tidy forecast table ready to export or plot.

    Columns: ``ds``, ``actual`` (NaN in the future), ``yhat``, ``yhat_lower``,
    ``yhat_upper`` and a boolean ``is_forecast``. Predictions and their lower
    bound are floored at zero because negative daily sales are not meaningful.
    """
    model, forecast = fit_forecast(
        prophet_df, horizon_days, interval_width, country
    )

    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        out[col] = out[col].clip(lower=0.0)

    out = out.merge(prophet_df.rename(columns={"y": "actual"}), on="ds", how="left")
    last_observed = prophet_df["ds"].max()
    out["is_forecast"] = out["ds"] > last_observed

    return out[["ds", "actual", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]]


def backtest(
    prophet_df: pd.DataFrame,
    horizon_days: int = config.FORECAST_HORIZON_DAYS,
    country: str = config.HOLIDAY_COUNTRY,
) -> dict[str, float]:
    """Hold out the last ``horizon_days`` and score the forecast against them.

    Reports MAE and RMSE in dollars and WAPE (weighted absolute percentage
    error, ``sum|e| / sum|y|``). WAPE is used instead of MAPE because many
    individual days have zero sales, which would make a plain MAPE blow up.
    """
    if len(prophet_df) <= horizon_days + 30:
        raise ValueError("Series too short to hold out a full horizon for backtesting.")

    train = prophet_df.iloc[:-horizon_days]
    test = prophet_df.iloc[-horizon_days:]

    model = build_model(country=country)
    model.fit(train)
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future).set_index("ds")

    pred = forecast.loc[test["ds"], "yhat"].clip(lower=0.0).to_numpy()
    actual = test["y"].to_numpy()
    error = actual - pred

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(error)) / denom * 100) if denom else float("nan")

    # Daily sales are spiky, so the day-level error is naturally large. What a
    # sales forecast is actually used for is the period total, which is far more
    # accurate; report it separately so the honest daily error is not the only
    # number on show.
    total_actual = float(np.sum(actual))
    total_pred = float(np.sum(pred))
    if total_actual:
        total_error_pct = (total_pred - total_actual) / total_actual * 100
    else:
        total_error_pct = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "wape_pct": wape,
        "total_actual": total_actual,
        "total_pred": total_pred,
        "total_error_pct": total_error_pct,
        "horizon_days": int(horizon_days),
        "train_days": int(len(train)),
    }


def build_daily_and_forecast(
    df: pd.DataFrame,
    region: str | None = None,
    category: str | None = None,
    horizon_days: int = config.FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    """End-to-end helper: clean orders in, forecast frame out, for one slice."""
    daily = data.daily_sales(df, region=region, category=category)
    return forecast_frame(data.to_prophet(daily), horizon_days=horizon_days)
