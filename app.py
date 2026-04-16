"""Superstore sales forecasting dashboard (Streamlit).

Run locally:

    streamlit run app.py

Four tabs: an overview with the KPI cards and region / category drilldown, a
Prophet forecast with its confidence interval, a year-over-year comparison, and
a page describing the Power BI build that mirrors this dashboard.

The forecast is pre-computed by ``scripts/train_forecast.py`` and read from
``outputs/forecast.csv``, so the app stays light and never trains Prophet on
startup. Region and category filters drive every tab at once, the same way the
slicers and the row-level-security role work in the Power BI report.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import config, data, features

st.set_page_config(
    page_title="Superstore Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

PRIMARY = "#2f6df6"
BAND = "rgba(47, 109, 246, 0.18)"
HISTORY_DAYS = 180


def compact_money(value: float) -> str:
    """Short dollar label so wide totals fit inside a KPI card."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 10_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:,.0f}"


# --------------------------------------------------------------------------- #
# Cached data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_orders() -> pd.DataFrame:
    return data.load_clean()


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame | None:
    if not config.FORECAST_CSV.exists():
        return None
    fc = pd.read_csv(config.FORECAST_CSV, parse_dates=["ds"])
    fc["is_forecast"] = fc["is_forecast"].astype(bool)
    return fc


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    path = config.FORECAST_METRICS
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_dax() -> str:
    path = config.ROOT / "powerbi" / "dax_measures.dax"
    return path.read_text() if path.exists() else ""


orders = load_orders()
forecast_all = load_forecast()
metrics = load_metrics()


# --------------------------------------------------------------------------- #
# Sidebar filters (drilldown + the Power BI RLS analog)
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
region = st.sidebar.selectbox("Region", [config.ALL, *config.REGIONS])
category = st.sidebar.selectbox("Category", [config.ALL, *config.CATEGORIES])
st.sidebar.caption(
    "The region filter mirrors the row-level-security role in the Power BI "
    "report, where a regional manager only ever sees their own region."
)

frame = data.filter_frame(orders, region=region, category=category)


# --------------------------------------------------------------------------- #
# Header + KPI cards
# --------------------------------------------------------------------------- #
st.title("📈 Superstore Sales Forecasting")
st.markdown(
    "Sales KPIs, region and category drilldown, and a three-month Prophet "
    "forecast on the Sample Superstore dataset (9,994 orders, 2015 to 2018). "
    "The same model and measures back a Power BI report."
)

k = features.kpis(frame)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total sales", compact_money(k["total_sales"]))
c2.metric(
    "Sales YoY",
    "n/a" if k["yoy_pct"] is None else f"{k['yoy_pct']:+.1f}%",
    help="Change in the most recent year against the year before.",
)
c3.metric("Profit margin", f"{k['profit_margin'] * 100:.1f}%")
c4.metric("Orders", f"{k['orders']:,}")
c5.metric("Avg order value", f"${k['avg_order_value']:,.0f}")

scope = " · ".join(
    part for part in (
        region if region != config.ALL else "All regions",
        category if category != config.ALL else "All categories",
    )
)
st.caption(f"Showing: {scope}")

tab_overview, tab_forecast, tab_yoy, tab_powerbi = st.tabs(
    ["📊 Overview", "🔮 Forecast", "📅 Year over year", "🟨 Power BI"]
)


# --------------------------------------------------------------------------- #
# Overview tab
# --------------------------------------------------------------------------- #
with tab_overview:
    st.subheader("Sales trend")
    monthly = features.monthly_sales(frame).set_index("month")["sales"]
    st.line_chart(monthly, color=PRIMARY, height=280)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Sales by category**")
        st.bar_chart(
            features.sales_by(frame, config.CATEGORY_COL),
            color=PRIMARY,
            horizontal=True,
            height=240,
        )
        st.markdown("**Sales by region**")
        st.bar_chart(
            features.sales_by(frame, config.REGION_COL),
            color=PRIMARY,
            horizontal=True,
            height=240,
        )
    with g2:
        st.markdown("**Top sub-categories by sales**")
        st.bar_chart(
            features.sales_by(frame, config.SUBCATEGORY_COL).head(10),
            color=PRIMARY,
            horizontal=True,
            height=240,
        )
        st.markdown("**Sales by segment**")
        st.bar_chart(
            features.sales_by(frame, config.SEGMENT_COL),
            color=PRIMARY,
            horizontal=True,
            height=240,
        )

    with st.expander("Peek at the orders"):
        st.dataframe(frame.head(50), width="stretch")


# --------------------------------------------------------------------------- #
# Forecast tab
# --------------------------------------------------------------------------- #
def forecast_slice(fc: pd.DataFrame, region: str, category: str) -> pd.DataFrame:
    return fc[(fc["region"] == region) & (fc["category"] == category)].copy()


def forecast_chart(fc_slice: pd.DataFrame) -> go.Figure:
    """History plus the forecast line and its confidence band."""
    history = fc_slice[~fc_slice["is_forecast"]].copy()
    future = fc_slice[fc_slice["is_forecast"]].copy()

    # Trim the noisy daily history to the most recent stretch, and bridge the
    # gap so the forecast line starts where the history ends.
    history = history.tail(HISTORY_DAYS)
    if not history.empty:
        future = pd.concat([history.tail(1), future], ignore_index=True)

    roll = history.assign(actual_7d=history["actual"].rolling(7, min_periods=1).mean())

    fig = go.Figure()
    # Confidence band first, so everything else draws on top of it.
    fig.add_trace(
        go.Scatter(
            x=list(future["ds"]) + list(future["ds"][::-1]),
            y=list(future["yhat_upper"]) + list(future["yhat_lower"][::-1]),
            fill="toself",
            fillcolor=BAND,
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name="90% interval",
        )
    )
    # Daily sales are spiky, so the history is shown as a 7-day average; that
    # keeps it on the same smooth footing as the Prophet line.
    fig.add_trace(
        go.Scatter(
            x=roll["ds"], y=roll["actual_7d"], name="Actual (7-day avg)",
            mode="lines", line={"color": "#1b2a4a", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future["ds"], y=future["yhat"], name="Forecast",
            mode="lines", line={"color": PRIMARY, "width": 2.5, "dash": "dash"},
        )
    )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
        yaxis_title="Daily sales ($)",
    )
    return fig


with tab_forecast:
    if forecast_all is None:
        st.warning(
            "No forecast found. Run `python scripts/train_forecast.py` to "
            "generate `outputs/forecast.csv`."
        )
    else:
        fc_slice = forecast_slice(forecast_all, region, category)
        future = fc_slice[fc_slice["is_forecast"]]
        history = fc_slice[~fc_slice["is_forecast"]]

        next_90 = float(future["yhat"].sum())
        # Compare against the same three months a year earlier, not the prior 90
        # days: Q1 against Q4 would look like a crash when it is just the normal
        # seasonal dip, whereas year-over-year isolates real growth.
        py_start = future["ds"].min() - pd.Timedelta(days=365)
        py_end = future["ds"].max() - pd.Timedelta(days=365)
        same_ly = float(
            history.loc[
                history["ds"].between(py_start, py_end), "actual"
            ].sum()
        )
        change = (next_90 / same_ly - 1) * 100 if same_ly else None

        st.subheader("Three-month sales forecast")
        f1, f2, f3 = st.columns(3)
        f1.metric("Forecast, next 90 days", f"${next_90:,.0f}")
        f2.metric("Same 90 days last year", f"${same_ly:,.0f}")
        f3.metric(
            "Implied YoY",
            "n/a" if change is None else f"{change:+.1f}%",
        )

        st.plotly_chart(forecast_chart(fc_slice), use_container_width=True)

        note = (
            "Prophet with weekly and yearly seasonality plus US public-holiday "
            "effects, each modelled separately. The shaded area is the "
            f"{int(config.INTERVAL_WIDTH * 100)}% confidence interval."
        )
        if metrics and "backtest" in metrics:
            bt = metrics["backtest"]
            note += (
                " On a 90-day hold-out of the overall series the forecast of the "
                f"period total lands within {abs(bt['total_error_pct']):.1f}% of "
                f"actual; day-level error is naturally higher (WAPE "
                f"{bt['wape_pct']:.0f}%) because daily sales are spiky."
            )
        st.caption(note)


# --------------------------------------------------------------------------- #
# Year-over-year tab
# --------------------------------------------------------------------------- #
with tab_yoy:
    st.subheader("Year-over-year sales")
    table = features.yoy_table(frame)

    st.bar_chart(table["sales"], color=PRIMARY, height=280)

    show = table.copy()
    show.index = show.index.astype(str)
    show = show.rename(
        columns={
            "sales": "Sales ($)",
            "prev_year_sales": "Prev year ($)",
            "yoy_change": "Change ($)",
            "yoy_pct": "YoY (%)",
        }
    )
    st.dataframe(
        show.style.format(
            {
                "Sales ($)": "{:,.0f}",
                "Prev year ($)": "{:,.0f}",
                "Change ($)": "{:+,.0f}",
                "YoY (%)": "{:+.1f}",
            },
            na_rep="-",
        ),
        width="stretch",
    )
    st.caption(
        "Sales YoY % is the same measure as in the Power BI report: this year's "
        "sales against the same period last year, driven by the slicers above."
    )


# --------------------------------------------------------------------------- #
# Power BI tab
# --------------------------------------------------------------------------- #
with tab_powerbi:
    st.subheader("The Power BI report")
    st.markdown(
        """
        The resume version of this project is a **Power BI** report on the same
        Superstore data. It reproduces this dashboard with:

        - **KPI cards** for total sales, profit margin, orders and YoY growth.
        - **Drilldown** from region to category to sub-category.
        - **DAX measures** for the KPIs and the year-over-year comparison.
        - **Row-level security** so a regional manager only sees their region.
        - The **Prophet forecast** loaded in as its own table, so the report
          shows the same forecast line and confidence band you see here.

        A Power BI report needs a paid license or the Windows desktop app to
        open, so it cannot be hosted for free in a browser. This Streamlit app
        is the openly clickable version; the Power BI project files, the DAX
        measures and the push script all live in the repo under `powerbi/`.
        """
    )

    dax = load_dax()
    if dax:
        with st.expander("DAX measures used in the report"):
            st.code(dax, language="dax")

    if forecast_all is not None:
        st.download_button(
            "Download forecast.csv",
            data=config.FORECAST_CSV.read_bytes(),
            file_name="forecast.csv",
            mime="text/csv",
            help="The exported forecast that scripts/push_to_powerbi.py loads "
            "into the Power BI dataset.",
        )

st.divider()
st.caption(
    "Built with pandas, Prophet, Plotly and Streamlit, with a matching Power BI "
    "report. Educational demo on a public dataset, so check the numbers before "
    "acting on any single forecast."
)
