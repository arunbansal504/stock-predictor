from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.common.types import RiskProfile
from stockpredictor.features.technical import compute_atr
from stockpredictor.portfolio import service as service_module
from stockpredictor.portfolio.service import _latest_atr_by_symbol, construct_portfolio_from_lake


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


class _FakeSession:
    """Stands in for a SQLAlchemy Session: no securities/sectors, which is
    fine for these tests since they only assert on the investment_amount
    pass-through, not on sector data."""

    def execute(self, *args, **kwargs):
        return self

    def scalars(self):
        return self

    def all(self):
        return []

    def close(self):
        pass


def _fake_session_factory():
    return _FakeSession()


def _multi_symbol_prices(symbols, n=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    frames = []
    for i, s in enumerate(symbols):
        close = 100 + np.cumsum(rng.normal(0, 1 + i * 0.2, n))
        high = close + rng.uniform(0.5, 2.0, n)
        low = close - rng.uniform(0.5, 2.0, n)
        open_ = close + rng.normal(0, 0.5, n)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": s,
                    "date": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "close_adj": close,
                    "volume": 100_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_construct_portfolio_from_lake_threads_investment_amount(monkeypatch):
    """investment_amount is a plain pass-through from the API/UI-facing
    construct_portfolio_from_lake down to the pure construct_portfolio --
    this confirms the wiring, not the allocation math itself (already
    covered by test_portfolio_constructor.py)."""
    symbols = [f"S{i}" for i in range(6)]
    ranked = pd.DataFrame({"symbol": symbols, "rank": range(1, 7), "score": np.linspace(0.7, 0.5, 6)})
    prices = _multi_symbol_prices(symbols)
    calib = pd.DataFrame(
        {
            "decile": [0, 1],
            "score_min": [0.0, 0.5],
            "score_max": [0.49, 1.0],
            "mean_return": [0.01, 0.04],
            "median_return": [0.01, 0.04],
            "n_obs": [10, 10],
        }
    )

    monkeypatch.setattr(service_module, "read_latest_rankings", lambda lake, horizon: ranked)
    monkeypatch.setattr(service_module, "_read_for_symbols", lambda lake, layer, domain, syms: prices)
    monkeypatch.setattr(service_module, "read_latest_return_calibration", lambda lake, strategy_id, horizon: calib)

    pf = construct_portfolio_from_lake(
        lake=None,
        session_factory=_fake_session_factory,
        horizon="5d",
        risk_profile=RiskProfile.BALANCED,
        top_n=6,
        investment_amount=20_000.0,
    )

    assert pf is not None
    assert pf.expected_return is not None
    assert pf.expected_final_value == pytest.approx(20_000.0 * (1 + pf.expected_return))
    for p in pf.positions:
        assert p.allocated_amount == pytest.approx(p.weight * 20_000.0)


def test_construct_portfolio_from_lake_investment_amount_none_leaves_amounts_none(monkeypatch):
    symbols = [f"S{i}" for i in range(6)]
    ranked = pd.DataFrame({"symbol": symbols, "rank": range(1, 7), "score": np.linspace(0.7, 0.5, 6)})
    prices = _multi_symbol_prices(symbols)
    calib = pd.DataFrame(
        {
            "decile": [0, 1],
            "score_min": [0.0, 0.5],
            "score_max": [0.49, 1.0],
            "mean_return": [0.01, 0.04],
            "median_return": [0.01, 0.04],
            "n_obs": [10, 10],
        }
    )

    monkeypatch.setattr(service_module, "read_latest_rankings", lambda lake, horizon: ranked)
    monkeypatch.setattr(service_module, "_read_for_symbols", lambda lake, layer, domain, syms: prices)
    monkeypatch.setattr(service_module, "read_latest_return_calibration", lambda lake, strategy_id, horizon: calib)

    pf = construct_portfolio_from_lake(
        lake=None,
        session_factory=_fake_session_factory,
        horizon="5d",
        risk_profile=RiskProfile.BALANCED,
        top_n=6,
    )

    assert pf is not None
    assert pf.expected_final_value is None
    assert pf.expected_return_amount is None
    for p in pf.positions:
        assert p.allocated_amount is None
        assert p.expected_return_amount is None
