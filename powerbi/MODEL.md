# Power BI report

The resume lists this project as a Power BI report on the Superstore data. This
note describes how that report is built, so it can be rebuilt in Power BI
Desktop from the files in this folder and the data in the repo. The DAX for
every measure is in [dax_measures.dax](dax_measures.dax).

A Power BI report needs a paid license or the Windows-only desktop app to open,
so it cannot be hosted for free in a browser. The Streamlit app in this repo is
the openly clickable version and shows the same numbers, because the measures
below are the exact counterparts of the functions in `forecasting/features.py`.

## Data sources

- **Orders** - `data/raw/superstore.csv`, loaded with Get Data > Text/CSV. The
  9,994 order lines from 2015 to 2018, one row per product in an order.
- **Forecast** - `outputs/forecast.csv`, the Prophet output. Loaded either as a
  refreshable Text/CSV source or pushed into a streaming dataset by
  `scripts/push_to_powerbi.py` (see "Keeping the forecast fresh" below).

## Model

A small star schema:

```
        Date (date dimension)
          |
          | 1..*        1..*
        Orders  ----  Forecast
      (fact: Sales,   (yhat, yhat_lower,
       Profit, ...)    yhat_upper, is_forecast)
```

- **Date** is a calculated `CALENDAR` table (see `dax_measures.dax`), marked as
  a date table on `[Date]`. Both Orders and Forecast join to it, so the trend
  visual can put actuals and the forecast on one axis.
- **Orders[Order Date]** → **Date[Date]** (single direction, one-to-many).
- **Forecast[ds]** → **Date[Date]** (single direction, one-to-many).
- Implicit measures are discouraged; every number comes from an explicit
  measure so the report and the app cannot drift apart.

## Measures

Grouped in `dax_measures.dax`:

- **Core** - Total Sales, Total Profit, Total Quantity, Profit Margin.
- **Orders** - Orders (distinct count), Avg Order Value.
- **Year over year** - Sales PY, Sales YoY, Sales YoY % using
  `SAMEPERIODLASTYEAR`.
- **Forecast** - Forecast Sales / Lower / Upper (filtered to the future
  horizon), and Sales or Forecast for a single continuous trend line.

## Report pages

1. **Overview** - KPI cards for Total Sales, Sales YoY %, Profit Margin, Orders
   and Avg Order Value; a sales trend line; and bar charts for sales by
   category, region, sub-category and segment.
2. **Forecast** - the Sales or Forecast line with the 90% interval drawn as a
   shaded area from Forecast Lower to Forecast Upper, plus cards for the next
   90-day forecast total and the implied change.
3. **Year over year** - Sales and Sales YoY % by year, with the region and
   category slicers driving the comparison.

## Drilldown

The category visuals use a drilldown hierarchy so a reader can go from the big
picture down to a single product line:

```
Region  ->  Category  ->  Sub-Category
```

## Row-level security

A **Region manager** role filters the Orders table to a single region, so a
signed-in manager only ever sees their own numbers. The table filter DAX is at
the bottom of `dax_measures.dax`; it maps the signed-in user to a region
through a small `UserRegion` mapping table. Roles are tested in Desktop with
"View as role" and enforced after publishing to the Power BI Service.

## Keeping the forecast fresh

Rather than editing the report by hand, the forecast is regenerated and loaded
automatically:

1. `python scripts/train_forecast.py` retrains Prophet and writes
   `outputs/forecast.csv`.
2. `python scripts/push_to_powerbi.py` authenticates as a service principal and
   pushes the rows into the Power BI dataset over the REST API, replacing the
   previous forecast (see `forecasting/powerbi.py`).

For a plain Text/CSV import the same file is picked up by a scheduled refresh in
the Power BI Service instead.
