"""Contract test for the yfinance price connector (§5, §22: "each source gets
a contract test that catches silent schema/endpoint drift").

Does NOT hit the network (tests must be deterministic/offline) — instead it
pins the *normalization contract*: given raw output shaped the way
yfinance's `Ticker.history()` actually returns it (DatetimeIndex + Open/High/
Low/Close/Adj Close/Volume columns), fetch_prices must produce a frame
matching PRICE_BRONZE_COLUMNS. If yfinance ever renames a column, this test
doesn't catch that live -- but it does catch us breaking our own contract,
and documents exactly what shape we depend on so a real drift is easy to spot.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.connectors import prices_yfinance
from stockpredictor.connectors.base import PRICE_BRONZE_COLUMNS, validate_schema


def _raw_yfinance_shaped_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=index,
    )


def test_fetch_prices_normalizes_raw_yfinance_shape_to_bronze_contract(monkeypatch):
    monkeypatch.setattr(
        prices_yfinance, "_download_one", lambda ticker, start, end: _raw_yfinance_shaped_frame()
    )

    out = prices_yfinance.fetch_prices(
        ["RELIANCE"], dt.date(2024, 1, 1), dt.date(2024, 1, 2), exchange="NSE"
    )

    validate_schema(out, PRICE_BRONZE_COLUMNS, context="contract")
    assert list(out.columns) == list(PRICE_BRONZE_COLUMNS.keys())
    assert (out["symbol"] == "RELIANCE").all()
    assert (out["source"] == "yfinance").all()
    assert len(out) == 2


def test_fetch_prices_skips_symbol_on_empty_response(monkeypatch):
    monkeypatch.setattr(
        prices_yfinance, "_download_one", lambda ticker, start, end: pd.DataFrame()
    )
    out = prices_yfinance.fetch_prices(
        ["GHOST"], dt.date(2024, 1, 1), dt.date(2024, 1, 2), exchange="NSE"
    )
    assert out.empty


def test_fetch_prices_skips_symbol_on_exception_without_aborting_batch(monkeypatch):
    calls = {"RELIANCE": _raw_yfinance_shaped_frame(), "BADTICKER": None}

    def fake_download(ticker, start, end):
        symbol = ticker.replace(".NS", "")
        if calls[symbol] is None:
            raise RuntimeError("simulated provider failure")
        return calls[symbol]

    monkeypatch.setattr(prices_yfinance, "_download_one", fake_download)

    out = prices_yfinance.fetch_prices(
        ["RELIANCE", "BADTICKER"], dt.date(2024, 1, 1), dt.date(2024, 1, 2), exchange="NSE"
    )
    # BADTICKER's failure is skipped, not fatal -- RELIANCE's rows still land.
    assert set(out["symbol"]) == {"RELIANCE"}


def test_to_provider_ticker_appends_correct_suffix():
    assert prices_yfinance.to_provider_ticker("RELIANCE", "NSE") == "RELIANCE.NS"
    assert prices_yfinance.to_provider_ticker("RELIANCE", "BSE") == "RELIANCE.BO"
