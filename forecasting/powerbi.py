"""Push the exported forecast into a Power BI dataset over the REST API.

This is the piece the resume describes as loading the forecast CSV into the
Power BI dataset automatically, so the report never has to be refreshed by
hand. It authenticates as a service principal, optionally clears the target
table, and pushes the rows in batches (the push API accepts at most 10,000 rows
per request).

Nothing here runs in CI: it needs real Azure AD / Power BI credentials, which
live in environment variables (see ``scripts/push_to_powerbi.py``). The module
imports cleanly without them and is unit-tested with a fake HTTP session, so the
request building is covered without hitting the network.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd

# Power BI's push API caps a single request at 10,000 rows.
MAX_ROWS_PER_REQUEST = 10_000

_AUTHORITY = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
_SCOPE = "https://analysis.windows.net/powerbi/api/.default"
_API_ROOT = "https://api.powerbi.com/v1.0/myorg"
_TIMEOUT = 60


def _new_session():
    """Import requests lazily so the module loads even without it installed."""
    import requests

    return requests.Session()


def get_access_token(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    session: Any | None = None,
) -> str:
    """Acquire an access token via the client-credentials (service principal) flow."""
    session = session or _new_session()
    response = session.post(
        _AUTHORITY.format(tenant_id=tenant_id),
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": _SCOPE,
        },
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _rows_url(dataset_id: str, table_name: str, group_id: str | None) -> str:
    """Build the rows endpoint, scoped to a workspace when one is given."""
    prefix = f"{_API_ROOT}/groups/{group_id}" if group_id else _API_ROOT
    return f"{prefix}/datasets/{dataset_id}/tables/{table_name}/rows"


def _chunks(rows: Sequence[dict], size: int) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def clear_rows(
    dataset_id: str,
    table_name: str,
    token: str,
    group_id: str | None = None,
    session: Any | None = None,
) -> None:
    """Delete every row currently in the target table."""
    session = session or _new_session()
    response = session.delete(
        _rows_url(dataset_id, table_name, group_id),
        headers={"Authorization": f"Bearer {token}"},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()


def push_rows(
    dataset_id: str,
    table_name: str,
    rows: Sequence[dict],
    token: str,
    group_id: str | None = None,
    chunk_size: int = MAX_ROWS_PER_REQUEST,
    session: Any | None = None,
) -> int:
    """Push rows to the dataset table in batches. Returns the number sent."""
    session = session or _new_session()
    url = _rows_url(dataset_id, table_name, group_id)
    headers = {"Authorization": f"Bearer {token}"}

    sent = 0
    for chunk in _chunks(rows, chunk_size):
        response = session.post(
            url, headers=headers, json={"rows": chunk}, timeout=_TIMEOUT
        )
        response.raise_for_status()
        sent += len(chunk)
    return sent


def forecast_to_rows(forecast: pd.DataFrame) -> list[dict]:
    """Turn a forecast frame into JSON-serialisable rows for the push API.

    Timestamps become ISO date strings and pandas NaN becomes ``None`` (JSON
    null), which is what the Power BI table schema expects.
    """
    frame = forecast.copy()
    if "ds" in frame.columns:
        frame["ds"] = pd.to_datetime(frame["ds"]).dt.strftime("%Y-%m-%d")
    frame = frame.astype(object).where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


def load_forecast(
    forecast: pd.DataFrame,
    dataset_id: str,
    table_name: str,
    token: str,
    group_id: str | None = None,
    replace: bool = True,
    session: Any | None = None,
) -> int:
    """Load a forecast frame into a Power BI table, replacing it by default."""
    session = session or _new_session()
    if replace:
        clear_rows(dataset_id, table_name, token, group_id, session=session)
    rows = forecast_to_rows(forecast)
    return push_rows(
        dataset_id, table_name, rows, token, group_id, session=session
    )
