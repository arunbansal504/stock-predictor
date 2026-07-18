"""Contract test for the corporate-actions connector (§5, §22). No network --
pins normalization against the real observed shape of yfinance's
`Ticker.splits` / `Ticker.dividends` (tz-aware DatetimeIndex Series), which
was verified live against TCS.NS before this module was written.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from stockpredictor.connectors import corporate_actions_yfinance as ca


def _fake_ticker() -> SimpleNamespace:
    splits = pd.Series(
        [2.0],
        index=pd.DatetimeIndex(["2018-05-31 09:15:00+05:30"], name="Date", tz="Asia/Kolkata"),
        name="Stock Splits",
    )
    dividends = pd.Series(
        [11.0, 11.0],
        index=pd.DatetimeIndex(
            ["2025-07-16 09:15:00+05:30", "2025-10-15 09:15:00+05:30"],
            name="Date",
            tz="Asia/Kolkata",
        ),
        name="Dividends",
    )
    return SimpleNamespace(splits=splits, dividends=dividends)


def test_fetch_corporate_actions_normalizes_splits_and_dividends(monkeypatch):
    monkeypatch.setattr(ca, "_fetch_ticker_actions", lambda ticker: _fake_ticker())

    out = ca.fetch_corporate_actions(["TCS"])
    assert list(out.columns) == ca.CORPORATE_ACTION_COLUMNS
    assert len(out) == 3  # 1 split + 2 dividends

    splits = out[out["action_type"] == "split"]
    assert splits.iloc[0]["ratio"] == 2.0
    assert splits.iloc[0]["ex_date"].isoformat() == "2018-05-31"
    # Conservative PIT stance: knowable_date defaults to ex_date (see module docstring).
    assert splits.iloc[0]["knowable_date"] == splits.iloc[0]["ex_date"]

    dividends = out[out["action_type"] == "dividend"]
    assert len(dividends) == 2
    assert dividends.iloc[0]["value"] == 11.0


def test_fetch_corporate_actions_skips_symbol_on_failure(monkeypatch):
    def fake_fetch(ticker):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(ca, "_fetch_ticker_actions", fake_fetch)
    out = ca.fetch_corporate_actions(["BADSYMBOL"])
    assert out.empty


def test_fetch_corporate_actions_empty_history_returns_empty_frame(monkeypatch):
    empty_ticker = SimpleNamespace(splits=pd.Series(dtype="float64"), dividends=pd.Series(dtype="float64"))
    monkeypatch.setattr(ca, "_fetch_ticker_actions", lambda ticker: empty_ticker)
    out = ca.fetch_corporate_actions(["NOACTIONS"])
    assert out.empty


def test_validate_actions_raises_on_none_splits():
    """Regression test: yfinance returning None (observed live under rapid
    sequential calls across a 40-symbol universe) must not crash with an
    AttributeError on `.items()` -- it must be treated as a failure."""
    t = SimpleNamespace(splits=None, dividends=pd.Series(dtype="float64"))
    with pytest.raises(RuntimeError, match="None splits/dividends"):
        ca._validate_actions(t, "BADTICKER.NS")


def test_validate_actions_raises_on_none_dividends():
    t = SimpleNamespace(splits=pd.Series(dtype="float64"), dividends=None)
    with pytest.raises(RuntimeError, match="None splits/dividends"):
        ca._validate_actions(t, "BADTICKER.NS")


def test_validate_actions_passes_through_valid_ticker():
    t = _fake_ticker()
    assert ca._validate_actions(t, "TCS.NS") is t


def test_fetch_corporate_actions_skips_symbol_when_validation_fails(monkeypatch):
    """End-to-end: a None-splits ticker is skipped gracefully (like any
    other per-symbol failure, §3 NFR), not a pipeline crash."""

    def fake_fetch(ticker):
        return ca._validate_actions(SimpleNamespace(splits=None, dividends=None), ticker)

    monkeypatch.setattr(ca, "_fetch_ticker_actions", fake_fetch)
    out = ca.fetch_corporate_actions(["FLAKY"])
    assert out.empty
