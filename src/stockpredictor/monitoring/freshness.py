"""Data freshness checks (§23: "per-source freshness & completeness").

A stale price feed is a silent failure mode distinct from an outright fetch
error: yfinance can return 200 with yesterday's (or last week's) data during
an outage without raising anything ingestion/prices.py would catch. This
checks the *latest date actually present* in the lake against wall-clock
time, independent of whether any individual fetch call "succeeded".
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.storage.lake import Lake


def check_domain_freshness(
    lake: Lake,
    layer: DataLayer,
    domain: str,
    max_staleness_days: int,
    as_of: dt.date | None = None,
) -> dict:
    """Returns a dict describing freshness for one (layer, domain). `ok` is
    False if the domain has no data at all, or its latest date is more than
    `max_staleness_days` calendar days behind `as_of` (default: today).
    `max_staleness_days` should be generous enough to cover weekends/holidays
    -- this is a freshness check, not a trading-calendar check.
    """
    as_of = as_of or dt.date.today()
    df = lake.read_all(layer, domain)
    if df.empty or "date" not in df.columns:
        return {
            "domain": domain,
            "ok": False,
            "latest_date": None,
            "staleness_days": None,
            "detail": "no data present",
        }

    latest_date = pd.to_datetime(df["date"]).max().date()
    staleness_days = (as_of - latest_date).days
    return {
        "domain": domain,
        "ok": staleness_days <= max_staleness_days,
        "latest_date": latest_date.isoformat(),
        "staleness_days": staleness_days,
        "detail": None,
    }
