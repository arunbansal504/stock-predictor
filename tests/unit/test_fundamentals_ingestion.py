from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.ingestion import fundamentals as fundamentals_ingestion


def _fake_fundamentals_df(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "period_end": pd.Timestamp("2025-03-31").date(),
                "knowable_date": pd.Timestamp("2025-04-25").date(),
                "revenue": 1.0e12,
                "net_income": 1.0e11,
                "eps": 50.0,
                "total_equity": 5.0e11,
                "total_debt": 2.0e11,
                "total_assets": 1.0e12,
                "shares_outstanding": 1.0e9,
            }
        ]
    )


def test_ingest_symbol_fundamentals_writes_to_silver(tmp_lake, monkeypatch):
    monkeypatch.setattr(
        fundamentals_ingestion, "fetch_fundamentals", lambda symbol, exchange="NSE": _fake_fundamentals_df(symbol)
    )
    rows = fundamentals_ingestion.ingest_symbol_fundamentals(tmp_lake, "RELIANCE")
    assert rows == 1

    out = tmp_lake.read(DataLayer.SILVER, "fundamentals", "RELIANCE")
    assert len(out) == 1
    assert out.iloc[0]["eps"] == 50.0


def test_ingest_symbol_fundamentals_empty_fetch_returns_zero(tmp_lake, monkeypatch):
    monkeypatch.setattr(
        fundamentals_ingestion,
        "fetch_fundamentals",
        lambda symbol, exchange="NSE": pd.DataFrame(columns=["symbol", "period_end", "knowable_date"]),
    )
    rows = fundamentals_ingestion.ingest_symbol_fundamentals(tmp_lake, "NODATA")
    assert rows == 0


def test_ingest_symbol_fundamentals_upserts_on_period_end(tmp_lake, monkeypatch):
    monkeypatch.setattr(
        fundamentals_ingestion, "fetch_fundamentals", lambda symbol, exchange="NSE": _fake_fundamentals_df(symbol)
    )
    fundamentals_ingestion.ingest_symbol_fundamentals(tmp_lake, "RELIANCE")

    revised = _fake_fundamentals_df("RELIANCE")
    revised.loc[0, "eps"] = 55.0  # a restated figure for the same fiscal year
    monkeypatch.setattr(fundamentals_ingestion, "fetch_fundamentals", lambda symbol, exchange="NSE": revised)
    fundamentals_ingestion.ingest_symbol_fundamentals(tmp_lake, "RELIANCE")

    out = tmp_lake.read(DataLayer.SILVER, "fundamentals", "RELIANCE")
    assert len(out) == 1  # updated in place, not duplicated
    assert out.iloc[0]["eps"] == 55.0
