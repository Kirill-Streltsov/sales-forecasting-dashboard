"""Tests for the Power BI push client.

A fake HTTP session records the calls, so request building, chunking and the
row conversion are all covered without touching the network or needing
credentials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting import powerbi


class FakeResponse:
    def __init__(self, payload=None):
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    """Captures every request so the test can assert on it afterwards."""

    def __init__(self, token="tok"):
        self.token = token
        self.posts = []
        self.deletes = []

    def post(self, url, **kwargs):
        self.posts.append({"url": url, **kwargs})
        if "oauth2" in url:
            return FakeResponse({"access_token": self.token})
        return FakeResponse()

    def delete(self, url, **kwargs):
        self.deletes.append({"url": url, **kwargs})
        return FakeResponse()


def test_get_access_token():
    session = FakeSession(token="abc123")
    token = powerbi.get_access_token("tenant", "client", "secret", session=session)
    assert token == "abc123"

    call = session.posts[0]
    assert "tenant/oauth2/v2.0/token" in call["url"]
    assert call["data"]["grant_type"] == "client_credentials"
    assert call["data"]["scope"].endswith("/.default")


def test_rows_url_with_and_without_group():
    with_group = powerbi._rows_url("ds1", "Forecast", "grp1")
    assert with_group.endswith("/groups/grp1/datasets/ds1/tables/Forecast/rows")

    no_group = powerbi._rows_url("ds1", "Forecast", None)
    assert "/groups/" not in no_group
    assert no_group.endswith("/myorg/datasets/ds1/tables/Forecast/rows")


def test_push_rows_chunks_requests():
    session = FakeSession()
    rows = [{"i": i} for i in range(25)]
    sent = powerbi.push_rows(
        "ds1", "Forecast", rows, "tok", chunk_size=10, session=session
    )
    assert sent == 25
    # 25 rows in batches of 10 -> 3 requests of 10, 10, 5.
    assert [len(p["json"]["rows"]) for p in session.posts] == [10, 10, 5]
    assert all(p["headers"]["Authorization"] == "Bearer tok" for p in session.posts)


def test_forecast_to_rows_serialises_dates_and_nulls():
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2019-01-01", "2019-01-02"]),
            "actual": [10.0, np.nan],
            "yhat": [11.0, 12.0],
        }
    )
    rows = powerbi.forecast_to_rows(frame)
    assert rows[0]["ds"] == "2019-01-01"
    assert rows[0]["actual"] == 10.0
    # NaN becomes JSON null.
    assert rows[1]["actual"] is None


def test_load_forecast_clears_then_pushes():
    session = FakeSession()
    frame = pd.DataFrame(
        {"ds": pd.to_datetime(["2019-01-01"]), "yhat": [11.0]}
    )
    sent = powerbi.load_forecast(
        frame, "ds1", "Forecast", "tok", group_id="grp1",
        replace=True, session=session,
    )
    assert sent == 1
    assert len(session.deletes) == 1
    assert len(session.posts) == 1
