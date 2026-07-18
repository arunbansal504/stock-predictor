from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.monitoring.freshness import check_domain_freshness


def test_freshness_no_data_is_not_ok(tmp_lake):
    result = check_domain_freshness(tmp_lake, DataLayer.SILVER, "prices", max_staleness_days=3)
    assert result["ok"] is False
    assert result["latest_date"] is None


def test_freshness_ok_when_data_is_current(tmp_lake):
    today = dt.date(2024, 6, 15)
    df = pd.DataFrame({"symbol": ["AAA"], "date": pd.to_datetime([today]), "close_adj": [100.0]})
    tmp_lake.write(df, DataLayer.SILVER, "prices", "AAA", key_cols=["symbol", "date"])

    result = check_domain_freshness(tmp_lake, DataLayer.SILVER, "prices", max_staleness_days=3, as_of=today)
    assert result["ok"] is True
    assert result["staleness_days"] == 0


def test_freshness_not_ok_when_stale(tmp_lake):
    old_date = dt.date(2024, 1, 1)
    as_of = dt.date(2024, 6, 15)
    df = pd.DataFrame({"symbol": ["AAA"], "date": pd.to_datetime([old_date]), "close_adj": [100.0]})
    tmp_lake.write(df, DataLayer.SILVER, "prices", "AAA", key_cols=["symbol", "date"])

    result = check_domain_freshness(tmp_lake, DataLayer.SILVER, "prices", max_staleness_days=3, as_of=as_of)
    assert result["ok"] is False
    assert result["staleness_days"] > 3


def test_freshness_exactly_at_threshold_is_ok(tmp_lake):
    as_of = dt.date(2024, 6, 15)
    old_date = as_of - dt.timedelta(days=3)
    df = pd.DataFrame({"symbol": ["AAA"], "date": pd.to_datetime([old_date]), "close_adj": [100.0]})
    tmp_lake.write(df, DataLayer.SILVER, "prices", "AAA", key_cols=["symbol", "date"])

    result = check_domain_freshness(tmp_lake, DataLayer.SILVER, "prices", max_staleness_days=3, as_of=as_of)
    assert result["ok"] is True
    assert result["staleness_days"] == 3
