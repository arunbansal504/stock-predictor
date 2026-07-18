from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.common.types import RiskProfile
from stockpredictor.portfolio.constructor import construct_portfolio


def _synthetic_scenario(symbols, seed=0, n=100):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    frames = []
    for i, s in enumerate(symbols):
        closes = 100 + np.cumsum(rng.normal(0, 1 + i * 0.2, n))
        frames.append(pd.DataFrame({"symbol": s, "date": dates, "close_adj": closes}))
    prices = pd.concat(frames, ignore_index=True)

    ranked = pd.DataFrame(
        {"symbol": symbols, "rank": range(1, len(symbols) + 1), "score": np.linspace(0.7, 0.5, len(symbols))}
    )
    atr = pd.Series({s: 2.0 for s in symbols})
    sectors = pd.Series({s: "IT" if i % 2 == 0 else "Financials" for i, s in enumerate(symbols)})
    calib = pd.DataFrame(
        {"decile": [0, 1], "score_min": [0.0, 0.5], "score_max": [0.49, 1.0], "mean_return": [0.01, 0.04], "median_return": [0.01, 0.04], "n_obs": [10, 10]}
    )
    return ranked, prices, atr, sectors, calib


def test_construct_portfolio_weights_sum_to_one_with_enough_names():
    symbols = [f"S{i}" for i in range(10)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.AGGRESSIVE, "5d", top_n=10)
    assert pf.total_allocated_weight == pytest.approx(1.0, abs=1e-6)
    assert sum(p.weight for p in pf.positions) == pytest.approx(1.0, abs=1e-6)


def test_construct_portfolio_flags_diversification_shortfall_with_too_few_names():
    symbols = [f"S{i}" for i in range(3)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.CONSERVATIVE, "5d", top_n=3)
    assert pf.diversification_warning is not None
    assert "10" in pf.diversification_warning  # conservative min_positions
    assert pf.total_allocated_weight < 1.0


def test_construct_portfolio_no_warning_when_min_positions_met():
    # Needs enough distinct SECTORS too, not just names: conservative's
    # max_sector_weight=0.25 means only 2 sectors (as the shared helper's
    # alternating IT/Financials split gives) creates its own hard ceiling
    # of 2*0.25=0.50 regardless of name count -- a separate constraint from
    # min_positions, and this test is specifically about the latter.
    symbols = [f"S{i}" for i in range(10)]
    ranked, prices, atr, _, calib = _synthetic_scenario(symbols)
    sectors = pd.Series({s: f"Sector{i % 5}" for i, s in enumerate(symbols)})
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.CONSERVATIVE, "5d", top_n=10)
    assert pf.diversification_warning is None


def test_construct_portfolio_respects_position_cap():
    symbols = [f"S{i}" for i in range(10)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.CONSERVATIVE, "5d", top_n=10)
    for p in pf.positions:
        assert p.weight <= 0.10 + 1e-9  # conservative max_position_weight


def test_construct_portfolio_respects_sector_cap():
    symbols = [f"S{i}" for i in range(10)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.BALANCED, "5d", top_n=10)
    sector_totals: dict[str, float] = {}
    for p in pf.positions:
        sector_totals[p.sector] = sector_totals.get(p.sector, 0.0) + p.weight
    for total in sector_totals.values():
        assert total <= 0.35 + 1e-6  # balanced max_sector_weight


def test_construct_portfolio_each_position_has_stop_below_and_target_above_entry():
    symbols = [f"S{i}" for i in range(6)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.BALANCED, "5d", top_n=6)
    for p in pf.positions:
        assert p.stop_loss < p.entry_price < p.target_price


def test_construct_portfolio_entry_price_skips_a_null_latest_row():
    """Regression test: observed live, a free-data-source gap left a row
    present for the most recent date with a null close_adj (not a missing
    row -- an existing row with a NaN value), which a naive `.tail(1)`
    would pick up and propagate as a NaN entry price into every downstream
    stop-loss/target calculation. Entry price must fall back to the last
    *valid* price, not the chronologically last row regardless of validity."""
    symbols = [f"S{i}" for i in range(6)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)

    last_date = prices["date"].max()
    second_last_date = prices["date"].drop_duplicates().sort_values().iloc[-2]
    target_symbol = symbols[0]
    last_valid_price = prices.loc[
        (prices["symbol"] == target_symbol) & (prices["date"] == second_last_date), "close_adj"
    ].iloc[0]
    prices.loc[(prices["symbol"] == target_symbol) & (prices["date"] == last_date), "close_adj"] = float("nan")

    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.BALANCED, "5d", top_n=6)
    position = next(p for p in pf.positions if p.symbol == target_symbol)
    assert position.entry_price == pytest.approx(last_valid_price)
    assert not pd.isna(position.stop_loss)
    assert not pd.isna(position.target_price)


def test_construct_portfolio_expected_return_uses_calibration():
    symbols = [f"S{i}" for i in range(6)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.BALANCED, "5d", top_n=6)
    assert pf.expected_return is not None
    assert 0.0 < pf.expected_return < 0.05  # within the calibration table's observed range


def test_construct_portfolio_expected_volatility_is_positive():
    symbols = [f"S{i}" for i in range(6)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.BALANCED, "5d", top_n=6)
    assert pf.expected_volatility > 0


def test_construct_portfolio_excludes_symbols_missing_price_history():
    symbols = ["AAA", "BBB"]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols, n=100)
    # Add a candidate with no price history at all.
    ranked = pd.concat([ranked, pd.DataFrame({"symbol": ["NODATA"], "rank": [3], "score": [0.4]})], ignore_index=True)

    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.AGGRESSIVE, "5d", top_n=3)
    assert "NODATA" in pf.excluded_symbols
    assert "NODATA" not in [p.symbol for p in pf.positions]


def test_construct_portfolio_empty_when_no_candidates_have_price_history():
    ranked = pd.DataFrame({"symbol": ["NODATA"], "rank": [1], "score": [0.5]})
    prices = pd.DataFrame(columns=["symbol", "date", "close_adj"])
    atr = pd.Series(dtype="float64")
    sectors = pd.Series(dtype="object")
    calib = pd.DataFrame()

    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.AGGRESSIVE, "5d", top_n=1)
    assert pf.positions == []
    assert pf.total_allocated_weight == 0.0
    assert pf.expected_return is None
    assert pf.expected_volatility is None


def test_construct_portfolio_disclaimer_present():
    symbols = [f"S{i}" for i in range(6)]
    ranked, prices, atr, sectors, calib = _synthetic_scenario(symbols)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.BALANCED, "5d", top_n=6)
    assert "not investment advice" in pf.disclaimer.lower()
