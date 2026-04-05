"""Project-wide configuration: paths, column names and forecast settings.

Keeping these in one place means the notebook, the tests, the training scripts
and the Streamlit dashboard all agree on where the data lives, what the columns
are called and how the forecast is set up.
"""

from __future__ import annotations

from pathlib import Path

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "superstore.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

# The training script writes these; the dashboard reads them. They are checked
# into the repo so the app shows a real forecast without retraining on startup.
FORECAST_CSV = OUTPUTS / "forecast.csv"
FORECAST_METRICS = OUTPUTS / "forecast_metrics.json"

# --- raw schema ----------------------------------------------------------
DATE_COL = "Order Date"
SHIP_DATE_COL = "Ship Date"
SALES_COL = "Sales"
PROFIT_COL = "Profit"
QUANTITY_COL = "Quantity"
ORDER_ID_COL = "Order ID"
REGION_COL = "Region"
CATEGORY_COL = "Category"
SUBCATEGORY_COL = "Sub-Category"
SEGMENT_COL = "Segment"

# The four Superstore regions and three product categories. Fixed lists so the
# dashboard filters and the Power BI row-level-security roles line up.
REGIONS = ["Central", "East", "South", "West"]
CATEGORIES = ["Furniture", "Office Supplies", "Technology"]

# Sentinel used by the filters to mean "do not filter on this dimension".
ALL = "All"

# --- forecast settings ---------------------------------------------------
# Three months of daily forecast, matching the dashboard's forecast card.
FORECAST_HORIZON_DAYS = 90

# Prophet models weekly and yearly seasonality plus US public-holiday effects.
HOLIDAY_COUNTRY = "US"

# Width of the forecast confidence interval (0.9 -> 90% interval).
INTERVAL_WIDTH = 0.9
