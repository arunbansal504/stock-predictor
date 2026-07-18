from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.features.technical import compute_atr
from stockpredictor.portfolio.service import _latest_atr_by_symbol


def _price_frame(symbol: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "close_adj": close,
            "volume": 100_000,
        }
    )


def test_latest_atr_by_symbol_matches_compute_atr_directly():
    """Regression proof that computing ATR on demand from silver prices
    (portfolio/service.py) gives the exact same answer as the batch feature
    pipeline's own compute_atr -- this is the whole point of reusing that
    function rather than reimplementing ATR (see _latest_atr_by_symbol's
    docstring on the split-adjustment pitfall of a naive reimplementation)."""
    prices = _price_frame("AAA", 60, seed=1)
    result = _latest_atr_by_symbol(prices)

    expected = compute_atr(prices.sort_values("date"))["atr_14"].dropna().iloc[-1]
    assert result["AAA"] == pytest.approx(expected)


def test_latest_atr_by_symbol_does_not_blend_across_symbol_boundaries():
    """compute_atr's rolling/ewm calculations are not multi-symbol-safe on
    their own -- feeding it a concatenated multi-symbol frame without
    grouping would let symbol B's first row see symbol A's last close as
    its "previous close". This is the specific bug _latest_atr_by_symbol
    must avoid by grouping before calling compute_atr."""
    aaa = _price_frame("AAA", 60, seed=1)
    bbb = _price_frame("BBB", 60, seed=2)
    combined = pd.concat([aaa, bbb], ignore_index=True)

    result = _latest_atr_by_symbol(combined)

    expected_aaa = compute_atr(aaa.sort_values("date"))["atr_14"].dropna().iloc[-1]
    expected_bbb = compute_atr(bbb.sort_values("date"))["atr_14"].dropna().iloc[-1]
    assert result["AAA"] == pytest.approx(expected_aaa)
    assert result["BBB"] == pytest.approx(expected_bbb)


def test_latest_atr_by_symbol_empty_prices_returns_empty_series():
    result = _latest_atr_by_symbol(pd.DataFrame())
    assert result.empty


def test_latest_atr_by_symbol_excludes_symbol_with_insufficient_history():
    # ATR-14 needs at least 14 rows (min_periods=period in compute_atr's ewm).
    short = _price_frame("SHORT", 5, seed=3)
    result = _latest_atr_by_symbol(short)
    assert "SHORT" not in result.index
