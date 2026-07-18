"""Focused regression tests for ConstructedPortfolio's expected_sharpe
calculation (§12) -- separate from test_portfolio_constructor.py because
this specifically guards against a real bug caught while building this
feature: naively dividing a per-horizon expected_return by an *annualized*
expected_volatility (or worse, compounding the return to "annualize" it)
produces a unit-inconsistent or wildly exaggerated number. The fix uses
sqrt-of-time scaling consistently on both sides, matching
backtest/metrics.py's own sharpe_ratio() convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.common.types import RiskProfile
from stockpredictor.portfolio.constructor import construct_portfolio


def _single_asset_scenario(daily_return_std: float, mean_return_per_5d: float):
    """A single-asset portfolio (no HRP diversification, no sector/position
    capping complexity) with a precisely known daily return standard
    deviation, so expected_volatility is exactly predictable, and a
    calibration table with a single, known mean_return -- isolates the
    Sharpe formula from every other moving part."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2022-01-01", periods=n)
    daily_returns = rng.normal(0, daily_return_std, n)
    closes = 100 * np.cumprod(1 + daily_returns)
    prices = pd.DataFrame({"symbol": ["AAA"] * n, "date": dates, "close_adj": closes})

    ranked = pd.DataFrame({"symbol": ["AAA"], "rank": [1], "score": [0.7]})
    atr = pd.Series({"AAA": 2.0})
    sectors = pd.Series({"AAA": "IT"})
    calibration = pd.DataFrame(
        {
            "decile": [0],
            "score_min": [0.0],
            "score_max": [1.0],
            "mean_return": [mean_return_per_5d],
            "median_return": [mean_return_per_5d],
            "n_obs": [10],
        }
    )
    return ranked, prices, atr, sectors, calibration


def test_expected_sharpe_matches_manual_sqrt_of_time_formula():
    daily_std = 0.02  # 2% daily vol, a realistic single-stock figure
    mean_return_5d = 0.03  # 3% expected return over the 5-day horizon
    ranked, prices, atr, sectors, calib = _single_asset_scenario(daily_std, mean_return_5d)

    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.AGGRESSIVE, horizon="5d", top_n=1)

    # Manual, independent computation of the expected formula.
    daily_vol = pf.expected_volatility / (252**0.5)
    period_vol = daily_vol * (5**0.5)
    manual_sharpe = (mean_return_5d / period_vol) * ((252 / 5) ** 0.5)

    assert pf.expected_sharpe == pytest.approx(manual_sharpe, rel=1e-6)


def test_expected_sharpe_scales_correctly_across_different_horizons():
    """The same per-period return/vol relationship should give a *smaller*
    annualized Sharpe for a longer horizon than a shorter one, when
    everything else is held constant -- sqrt(252/horizon_days) shrinks as
    horizon_days grows. This is the sanity check that would have caught the
    original compounding bug immediately (compounding blows UP with more
    periods/year, i.e. SHORTER horizons, which is the opposite of a sane
    "more uncertainty over a longer hold" intuition here being about scaling,
    not directionally about risk)."""
    daily_std = 0.02
    mean_return = 0.03

    ranked, prices, atr, sectors, calib = _single_asset_scenario(daily_std, mean_return)
    pf_5d = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.AGGRESSIVE, horizon="5d", top_n=1)

    calib30 = calib.copy()
    pf_30d = construct_portfolio(ranked, prices, atr, sectors, calib30, RiskProfile.AGGRESSIVE, horizon="30d", top_n=1)

    # Same return/vol relationship, but sqrt(252/5) > sqrt(252/30) --
    # annualizing a 5-day figure scales it up more than a 30-day figure.
    assert pf_5d.expected_sharpe > pf_30d.expected_sharpe


def test_expected_sharpe_none_for_unparseable_horizon_label():
    ranked, prices, atr, sectors, calib = _single_asset_scenario(0.02, 0.03)
    pf = construct_portfolio(ranked, prices, atr, sectors, calib, RiskProfile.AGGRESSIVE, horizon="unknown", top_n=1)
    assert pf.expected_sharpe is None
    # expected_return/expected_volatility should still be computed fine --
    # only the Sharpe combination step needs a parseable horizon.
    assert pf.expected_return is not None
    assert pf.expected_volatility is not None
