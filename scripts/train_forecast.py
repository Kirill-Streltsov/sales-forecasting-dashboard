"""Train the Prophet forecast and export it for the dashboard and Power BI.

For every region / category slice the dashboard can select (including the "All"
roll-ups) this fits a Prophet model, produces a 3-month daily forecast with a
confidence interval, and stacks the results into a single tidy CSV. It also runs
a hold-out backtest on the overall series so the reported error is honest.

Outputs (checked into the repo so the app needs no retraining):

    outputs/forecast.csv           one row per (region, category, day)
    outputs/forecast_metrics.json  backtest error + run metadata

Run from the project root (needs the dev dependencies for Prophet):

    python scripts/train_forecast.py
"""

from __future__ import annotations

import json

import pandas as pd

from forecasting import config, data, features, forecast


def _slices() -> list[tuple[str, str]]:
    """Every (region, category) combination the dashboard exposes."""
    regions = [config.ALL, *config.REGIONS]
    categories = [config.ALL, *config.CATEGORIES]
    return [(r, c) for r in regions for c in categories]


def main() -> None:
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    df = data.load_clean()

    frames: list[pd.DataFrame] = []
    for region, category in _slices():
        daily = data.daily_sales(df, region=region, category=category)
        fc = forecast.forecast_frame(data.to_prophet(daily))
        fc.insert(0, "category", category)
        fc.insert(0, "region", region)
        frames.append(fc)
        label = f"{region} / {category}"
        print(f"  forecast done: {label:<28} ({len(daily):,} days of history)")

    forecast_all = pd.concat(frames, ignore_index=True)
    forecast_all.to_csv(config.FORECAST_CSV, index=False)
    print(f"wrote {config.FORECAST_CSV.relative_to(config.ROOT)} "
          f"({len(forecast_all):,} rows)")

    # Honest error estimate: hold out the last 3 months of the overall series.
    overall = data.to_prophet(data.daily_sales(df))
    scores = forecast.backtest(overall)

    metrics = {
        "backtest": scores,
        "interval_width": config.INTERVAL_WIDTH,
        "horizon_days": config.FORECAST_HORIZON_DAYS,
        "holiday_country": config.HOLIDAY_COUNTRY,
        "history_start": f"{df[config.DATE_COL].min():%Y-%m-%d}",
        "history_end": f"{df[config.DATE_COL].max():%Y-%m-%d}",
        "n_order_lines": int(len(df)),
        "n_orders": features.order_count(df),
        "total_sales": features.total_sales(df),
    }
    config.FORECAST_METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {config.FORECAST_METRICS.relative_to(config.ROOT)}")
    print(
        "backtest (last 90 days): "
        f"MAE ${scores['mae']:,.0f}, RMSE ${scores['rmse']:,.0f}, "
        f"WAPE {scores['wape_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
