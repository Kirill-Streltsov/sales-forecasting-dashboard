"""Sales forecasting on the Sample Superstore dataset.

The package is intentionally small and modular so the same code powers the
notebook, the test-suite, the training scripts and the Streamlit dashboard:

    config    -> paths, column names and forecast settings
    data      -> load / clean the raw orders and build daily sales series
    features  -> KPI and year-over-year aggregations (the DAX measures in Python)
    forecast  -> the Prophet model, the forecast frame and a backtest
    powerbi   -> push the exported forecast into a Power BI dataset

Import the submodules you need directly, e.g. ``from forecasting import data``.
The dashboard reads a pre-computed forecast, so it never imports ``forecast``
and therefore never needs Prophet at runtime.
"""

__version__ = "0.1.0"
