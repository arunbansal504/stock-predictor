from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.backtest.metrics import (
    cagr,
    calmar_ratio,
    hit_rate_by_decile,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def test_cagr_matches_manual_compounding():
    returns = pd.Series([0.1] * 10)
    horizon_days = 126  # 2 periods/year
    # growth = 1.1^10, years = 10 / 2 = 5 -> cagr = 1.1^10^(1/5) - 1 = 1.1^2 - 1 = 0.21
    assert cagr(returns, horizon_days) == pytest.approx(0.21, rel=1e-6)


def test_cagr_empty_series_is_nan():
    assert np.isnan(cagr(pd.Series(dtype="float64"), 21))


def test_sharpe_ratio_matches_manual_formula():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0])
    horizon_days = 21
    ppy = 252 / horizon_days
    expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(ppy)
    assert sharpe_ratio(returns, horizon_days) == pytest.approx(expected)


def test_sharpe_ratio_nan_for_zero_variance():
    returns = pd.Series([0.01, 0.01, 0.01])
    assert np.isnan(sharpe_ratio(returns, 21))


def test_sharpe_ratio_nan_for_insufficient_data():
    assert np.isnan(sharpe_ratio(pd.Series([0.01]), 21))


def test_sortino_ratio_matches_manual_formula_using_downside_only():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    horizon_days = 21
    ppy = 252 / horizon_days
    downside = returns[returns < 0]
    expected = (returns.mean() / downside.std(ddof=1)) * np.sqrt(ppy)
    assert sortino_ratio(returns, horizon_days) == pytest.approx(expected)


def test_sortino_ratio_nan_when_no_downside_periods():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert np.isnan(sortino_ratio(returns, 21))


def test_max_drawdown_known_scenario():
    returns = pd.Series([0.10, -0.20, 0.05])
    # equity: 1.10, 0.88, 0.924 ; running max: 1.10, 1.10, 1.10
    # drawdowns: 0, -0.2, -0.16 -> min = -0.2
    assert max_drawdown(returns) == pytest.approx(-0.20, rel=1e-6)


def test_max_drawdown_empty_is_nan():
    assert np.isnan(max_drawdown(pd.Series(dtype="float64")))


def test_calmar_ratio_is_cagr_over_abs_max_drawdown():
    # Needs a real drawdown (not all-positive returns, which would make
    # max_drawdown 0 and the ratio undefined) for a meaningful comparison.
    returns = pd.Series([0.1, 0.1, -0.2, 0.1, 0.1, -0.15, 0.1, 0.1, 0.1, 0.1])
    horizon_days = 126
    expected = cagr(returns, horizon_days) / abs(max_drawdown(returns))
    assert calmar_ratio(returns, horizon_days) == pytest.approx(expected)


def test_calmar_ratio_nan_when_no_drawdown():
    returns = pd.Series([0.1] * 10)  # monotonically up -> zero drawdown -> undefined ratio
    assert np.isnan(calmar_ratio(returns, 126))


def test_win_rate_basic():
    returns = pd.Series([0.1, -0.1, 0.2, -0.05, 0.0])
    assert win_rate(returns) == pytest.approx(0.4)  # 2 of 5 strictly positive


def test_hit_rate_by_decile_is_monotonic_for_an_informative_score():
    scores = pd.Series(np.arange(100, dtype="float64"))
    outcomes = (scores >= 50).astype(int)
    result = hit_rate_by_decile(scores, outcomes)
    assert result.loc[0] == pytest.approx(0.0)  # lowest-score decile: all negative outcomes
    assert result.loc[9] == pytest.approx(1.0)  # highest-score decile: all positive outcomes
    assert result.is_monotonic_increasing


def test_hit_rate_by_decile_empty_input():
    assert hit_rate_by_decile(pd.Series(dtype="float64"), pd.Series(dtype="float64")).empty


def test_information_coefficient_perfect_positive_correlation():
    scores = pd.Series([1, 2, 3, 4, 5])
    forward_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    assert information_coefficient(scores, forward_returns) == pytest.approx(1.0)


def test_information_coefficient_perfect_negative_correlation():
    scores = pd.Series([1, 2, 3, 4, 5])
    forward_returns = pd.Series([0.05, 0.04, 0.03, 0.02, 0.01])
    assert information_coefficient(scores, forward_returns) == pytest.approx(-1.0)


def test_information_coefficient_nan_with_insufficient_data():
    assert np.isnan(information_coefficient(pd.Series([1, 2]), pd.Series([0.1, 0.2])))
