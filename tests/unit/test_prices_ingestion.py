from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.connectors.base import PRICE_BRONZE_COLUMNS
from stockpredictor.ingestion import prices as prices_ingestion


def _bronze_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["RELIANCE", "RELIANCE"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            # adj_close differs from close to simulate a split adjustment --
            # silver's close_adj must reflect this, not raw close.
            "adj_close": [50.5, 51.0],
            "volume": pd.array([1000, 1100], dtype="int64"),
            "source": ["yfinance", "yfinance"],
        }
    )


def test_bronze_to_silver_uses_adjusted_close_and_pit_stamps():
    silver = prices_ingestion.bronze_to_silver(_bronze_frame())
    assert list(silver["close_adj"]) == [50.5, 51.0]
    # Prices are same-day knowable: knowable_date == date (see common/pit.py).
    assert list(silver["knowable_date"]) == list(silver["date"])
    assert set(prices_ingestion.SILVER_COLUMNS) == set(silver.columns)


def test_bronze_to_silver_empty_input_returns_empty_output():
    empty = pd.DataFrame(columns=list(PRICE_BRONZE_COLUMNS.keys()))
    out = prices_ingestion.bronze_to_silver(empty)
    assert out.empty
    assert list(out.columns) == prices_ingestion.SILVER_COLUMNS


def test_ingest_symbol_prices_writes_bronze_and_silver(tmp_lake, monkeypatch):
    monkeypatch.setattr(
        prices_ingestion.prices_yfinance, "fetch_prices", lambda *a, **k: _bronze_frame()
    )

    rows = prices_ingestion.ingest_symbol_prices(
        tmp_lake, "RELIANCE", dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert rows == 2

    bronze = tmp_lake.read(DataLayer.BRONZE, "prices", "RELIANCE")
    silver = tmp_lake.read(DataLayer.SILVER, "prices", "RELIANCE")
    assert len(bronze) == 2
    assert len(silver) == 2
    assert "close_adj" in silver.columns


def test_ingest_symbol_prices_handles_empty_fetch_gracefully(tmp_lake, monkeypatch):
    empty = pd.DataFrame(columns=list(PRICE_BRONZE_COLUMNS.keys()))
    monkeypatch.setattr(prices_ingestion.prices_yfinance, "fetch_prices", lambda *a, **k: empty)

    rows = prices_ingestion.ingest_symbol_prices(
        tmp_lake, "DELISTED", dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert rows == 0
