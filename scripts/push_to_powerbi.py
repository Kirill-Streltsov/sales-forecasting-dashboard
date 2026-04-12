"""Load the exported forecast into a Power BI push dataset.

This automates the manual "refresh the report" step: it reads
``outputs/forecast.csv`` and pushes the rows into a Power BI streaming / push
dataset via the REST API, replacing whatever is there.

Credentials come from the environment (a ``.env`` file works too) so nothing
secret is committed:

    PBI_TENANT_ID       Azure AD tenant id
    PBI_CLIENT_ID       service principal application id
    PBI_CLIENT_SECRET   service principal secret
    PBI_DATASET_ID      target push dataset id
    PBI_GROUP_ID        workspace id (optional; omit for "My workspace")
    PBI_TABLE           table name in the dataset (default: Forecast)

Run from the project root once the credentials are set:

    python scripts/push_to_powerbi.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from forecasting import config, powerbi


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    if not config.FORECAST_CSV.exists():
        sys.exit("outputs/forecast.csv not found. Run scripts/train_forecast.py first.")

    tenant_id = _require("PBI_TENANT_ID")
    client_id = _require("PBI_CLIENT_ID")
    client_secret = _require("PBI_CLIENT_SECRET")
    dataset_id = _require("PBI_DATASET_ID")
    group_id = os.environ.get("PBI_GROUP_ID") or None
    table = os.environ.get("PBI_TABLE", "Forecast")

    forecast = pd.read_csv(config.FORECAST_CSV)

    print("Authenticating with Power BI ...")
    token = powerbi.get_access_token(tenant_id, client_id, client_secret)

    print(f"Pushing {len(forecast):,} rows into table '{table}' ...")
    sent = powerbi.load_forecast(
        forecast, dataset_id, table, token, group_id=group_id, replace=True
    )
    print(f"Done. {sent:,} rows loaded into the Power BI dataset.")


if __name__ == "__main__":
    main()
