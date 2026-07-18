"""Contract test for the macro/benchmark connector (§5, §22). No network --
pins the normalization contract against a realistic yfinance-shaped raw
frame, same approach as test_prices_yfinance_contract.py.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.connectors import macro_yfinance


def _raw_yfinance_shaped_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="Date")
    return pd.DataFrame({"Close": [21500.0, 21600.0]}, index=index)


def test_fetch_macro_series_normalizes_shape(monkeypatch):
    monkeypatch.setattr(
        macro_yfinance, "_download_one", lambda ticker, start, end: _raw_yfinance_shaped_frame()
    )
    out = macro_yfinance.fetch_macro_series(["NIFTY50"], dt.date(2024, 1, 1), dt.date(2024, 1, 2))
    assert list(out.columns) == macro_yfinance.MACRO_COLUMNS
    assert (out["series"] == "NIFTY50").all()
    assert len(out) == 2


def test_fetch_macro_series_skips_unknown_series_name():
    out = macro_yfinance.fetch_macro_series(
        ["NOT_A_REAL_SERIES"], dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert out.empty


def test_fetch_macro_series_skips_failed_series_without_aborting(monkeypatch):
    def fake_download(ticker, start, end):
        if ticker == macro_yfinance.MACRO_TICKERS["CRUDE"]:
            raise RuntimeError("simulated failure")
        return _raw_yfinance_shaped_frame()

    monkeypatch.setattr(macro_yfinance, "_download_one", fake_download)
    out = macro_yfinance.fetch_macro_series(
        ["NIFTY50", "CRUDE"], dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert set(out["series"]) == {"NIFTY50"}
