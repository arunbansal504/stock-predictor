from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stockpredictor.common.types import RiskProfile
from stockpredictor.features.technical import compute_atr
from stockpredictor.portfolio import service as service_module
from stockpredictor.portfolio.service import MAX_AUTO_EXPAND_POOL, _latest_atr_by_symbol, construct_portfolio_from_lake


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


class _SectorFakeSession:
    """Like _FakeSession, but returns real per-symbol sector assignments --
    needed to test pool auto-expansion, where sector diversity determines
    whether adding more names can actually relax the sector cap."""

    def __init__(self, sector_by_symbol: dict[str, str]):
        self._sector_by_symbol = sector_by_symbol

    def execute(self, *args, **kwargs):
        return self

    def scalars(self):
        return self

    def all(self):
        return [SimpleNamespace(symbol=s, sector=sec) for s, sec in self._sector_by_symbol.items()]

    def close(self):
        pass


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
    # expected_final_value only credits the actually-*allocated* fraction
    # of investment_amount with expected_return (see
    # test_portfolio_constructor.py's
    # test_construct_portfolio_expected_final_value_scales_by_allocated_weight
    # for the regression this guards) -- this test is about wiring, not
    # re-deriving that formula, so just confirm it matches whatever
    # construct_portfolio itself computed.
    invested = 20_000.0 * pf.total_allocated_weight
    assert pf.expected_final_value == pytest.approx(20_000.0 + invested * pf.expected_return)
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


def test_construct_portfolio_from_lake_expands_pool_to_fully_deploy_capital(monkeypatch):
    """Regression test for a real UX gap: with too few candidates, a tight
    risk profile's position cap creates a hard ceiling on how much capital
    can ever be deployed regardless of investment_amount -- e.g.
    Conservative's 10% position cap means 5 names can never absorb more
    than 50%, no matter how confident the model is. construct_portfolio
    (constructor.py) correctly reports that as an honest shortfall, but
    construct_portfolio_from_lake must not just accept it when more ranked
    candidates exist -- it should pull in more (sufficiently
    sector-diverse) names to actually deploy the investor's stated amount,
    only falling back to a real shortfall once MAX_AUTO_EXPAND_POOL is
    exhausted (see the next test)."""
    n_symbols = 20
    symbols = [f"S{i}" for i in range(n_symbols)]
    ranked = pd.DataFrame(
        {"symbol": symbols, "rank": range(1, n_symbols + 1), "score": np.linspace(0.9, 0.5, n_symbols)}
    )
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
    # 5 distinct sectors, cycling -- diverse enough that neither cap has a
    # hard low ceiling once enough names are pulled in.
    sector_by_symbol = {s: f"Sector{i % 5}" for i, s in enumerate(symbols)}

    monkeypatch.setattr(service_module, "read_latest_rankings", lambda lake, horizon: ranked)
    monkeypatch.setattr(
        service_module, "_read_for_symbols", lambda lake, layer, domain, syms: prices[prices["symbol"].isin(syms)]
    )
    monkeypatch.setattr(service_module, "read_latest_return_calibration", lambda lake, strategy_id, horizon: calib)

    pf = construct_portfolio_from_lake(
        lake=None,
        session_factory=lambda: _SectorFakeSession(sector_by_symbol),
        horizon="5d",
        risk_profile=RiskProfile.CONSERVATIVE,
        top_n=5,
        investment_amount=100_000.0,
    )

    assert pf is not None
    # Requesting only 5 names under Conservative's 10% position cap can
    # never exceed 50% allocated -- if the pool never expanded, this would
    # be capped there. Confirm it actually pulled in more names and beat
    # that ceiling.
    assert len(pf.positions) > 5
    assert pf.total_allocated_weight > 0.55
    assert pf.diversification_warning is not None
    assert "expanded to" in pf.diversification_warning.lower()
    assert f"requested top 5" in pf.diversification_warning.lower()


def test_construct_portfolio_from_lake_still_reports_honest_shortfall_when_expansion_cant_help(monkeypatch):
    """If every available candidate shares one sector (or expansion is
    otherwise structurally incapable of relaxing the binding cap), the
    auto-expand loop must terminate at MAX_AUTO_EXPAND_POOL -- not spin
    forever -- and still surface the genuine shortfall, rather than
    pretending it deployed more than it did."""
    n_symbols = MAX_AUTO_EXPAND_POOL + 20  # more than the expansion ceiling can ever use
    symbols = [f"S{i}" for i in range(n_symbols)]
    ranked = pd.DataFrame(
        {"symbol": symbols, "rank": range(1, n_symbols + 1), "score": np.linspace(0.9, 0.5, n_symbols)}
    )
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
    # Every candidate in the SAME sector: Balanced's 35% sector cap is a
    # hard ceiling no amount of additional same-sector names can relax.
    sector_by_symbol = {s: "OnlySector" for s in symbols}

    monkeypatch.setattr(service_module, "read_latest_rankings", lambda lake, horizon: ranked)
    monkeypatch.setattr(
        service_module, "_read_for_symbols", lambda lake, layer, domain, syms: prices[prices["symbol"].isin(syms)]
    )
    monkeypatch.setattr(service_module, "read_latest_return_calibration", lambda lake, strategy_id, horizon: calib)

    pf = construct_portfolio_from_lake(
        lake=None,
        session_factory=lambda: _SectorFakeSession(sector_by_symbol),
        horizon="5d",
        risk_profile=RiskProfile.BALANCED,
        top_n=5,
        investment_amount=100_000.0,
    )

    assert pf is not None
    assert pf.total_allocated_weight <= 0.35 + 1e-6, "single-sector cap must still hold even after expansion"
    assert pf.diversification_warning is not None
    assert len(pf.positions) <= MAX_AUTO_EXPAND_POOL
