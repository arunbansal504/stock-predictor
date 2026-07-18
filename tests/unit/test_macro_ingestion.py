from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.ingestion import macro as macro_ingestion


def _fake_macro_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series": ["NIFTY50", "NIFTY50", "INDIA_VIX", "INDIA_VIX"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "close": [21500.0, 21600.0, 14.0, 13.5],
            "source": ["yfinance"] * 4,
        }
    )


def test_ingest_macro_series_writes_one_file_per_series(tmp_lake, monkeypatch):
    monkeypatch.setattr(
        macro_ingestion.macro_yfinance, "fetch_macro_series", lambda *a, **k: _fake_macro_df()
    )
    rows = macro_ingestion.ingest_macro_series(
        tmp_lake, ["NIFTY50", "INDIA_VIX"], dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert rows == 4

    nifty = macro_ingestion.read_macro_series(tmp_lake, "NIFTY50")
    vix = macro_ingestion.read_macro_series(tmp_lake, "INDIA_VIX")
    assert len(nifty) == 2
    assert len(vix) == 2
    assert list(nifty["close"]) == [21500.0, 21600.0]


def test_ingest_macro_series_empty_fetch_returns_zero(tmp_lake, monkeypatch):
    monkeypatch.setattr(
        macro_ingestion.macro_yfinance,
        "fetch_macro_series",
        lambda *a, **k: pd.DataFrame(columns=macro_ingestion.macro_yfinance.MACRO_COLUMNS),
    )
    rows = macro_ingestion.ingest_macro_series(
        tmp_lake, ["NIFTY50"], dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert rows == 0
