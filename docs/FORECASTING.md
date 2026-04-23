# Forecasting notes

How the Prophet model is set up and how well it does. The code is in
`forecasting/forecast.py`; this is the reasoning behind it.

## The series

Superstore records one row per product in an order, so the first step is to sum
`Sales` to one number per calendar day (`forecasting/data.py`). Days with no
orders are filled with zero rather than left as gaps, so Prophet sees a regular
daily frequency. That gives 1,458 days from 2015-01-03 to 2018-12-30, of which
about 220 are genuine no-order days.

## The model

Prophet with three components modelled separately, which is what the resume
describes:

- **Weekly seasonality** - the day-of-week pattern. Superstore sells much less
  at weekends, and this soaks that up instead of leaving it in the noise.
- **Yearly seasonality** - the within-year shape, including the Q4 ramp.
- **US public holidays** - added with `add_country_holidays`, so spikes and
  dips around holidays get their own component rather than distorting the
  weekly and yearly curves.

The confidence interval is set to 90% (`interval_width=0.9`). Predictions and
their lower bound are floored at zero, because a negative daily sales figure is
not meaningful.

The dashboard needs a forecast for every region and category the user can pick,
so `scripts/train_forecast.py` fits the model once per slice (5 regions
including "All" × 4 categories including "All" = 20 models) and stacks the
results into `outputs/forecast.csv`.

## How well it does

`forecasting/forecast.backtest` holds out the last 90 days of the overall
series, fits on the rest, and scores the forecast against the held-out days:

- **90-day total within about 13% of actual.** This is what a sales forecast is
  actually used for, and it is the honest headline. The model slightly
  under-forecasts because the last quarter of 2018 kept growing.
- **Day-level WAPE around 70%.** Daily sales are spiky - a single large order
  moves a day a lot - so the day-to-day error is naturally high. WAPE
  (`sum|error| / sum|actual|`) is used instead of MAPE because many days have
  zero sales, which would make a plain MAPE blow up.

Both numbers are written to `outputs/forecast_metrics.json` and the aggregate
one is shown under the forecast chart in the app, so the day-level noise is not
the only thing on display.

## What I would do next

- Model at weekly granularity for a smoother headline series, and keep the
  daily model only for the day-of-week story.
- Add promotion and discount regressors - `Discount` is in the data and clearly
  moves sales.
- Cross-validate the horizon with `prophet.diagnostics` rather than a single
  hold-out.
