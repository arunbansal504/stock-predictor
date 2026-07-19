from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.backtest.costs import CostModel
from stockpredictor.backtest.engine import select_rebalance_dates, simulate_top_k_strategy


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "score": [3, 1, 2, 1, 3, 2],
            "forward_return": [0.05, 0.01, 0.03, 0.02, 0.06, 0.04],
            "benchmark_forward_return": [0.02, 0.02, 0.02, 0.01, 0.01, 0.01],
        }
    )


ZERO_COST = CostModel(brokerage_bps=0, stt_bps=0, exchange_txn_bps=0, gst_bps=0, stamp_duty_bps=0, slippage_bps=0)


def test_simulate_top_k_selects_highest_scored_and_averages_returns():
    result = simulate_top_k_strategy(_frame(), horizon_days=5, top_k=2, cost_model=ZERO_COST)

    # d1: top-2 by score are A(3) and C(2) -> mean(0.05, 0.03) = 0.04
    # d2: top-2 by score are B(3) and C(2) -> mean(0.06, 0.04) = 0.05
    assert result.per_period_returns.loc["d1"] == pytest.approx(0.04)
    assert result.per_period_returns.loc["d2"] == pytest.approx(0.05)
    assert result.benchmark_returns.loc["d1"] == pytest.approx(0.02)
    assert result.benchmark_returns.loc["d2"] == pytest.approx(0.01)


def test_simulate_applies_transaction_costs():
    cost_model = CostModel(brokerage_bps=0, stt_bps=0, exchange_txn_bps=0, gst_bps=0, stamp_duty_bps=0, slippage_bps=50.0)
    result = simulate_top_k_strategy(_frame(), horizon_days=5, top_k=2, cost_model=cost_model)
    # round trip = 100bps = 1% drag
    assert result.per_period_returns.loc["d1"] == pytest.approx(0.04 - 0.01)


def test_simulate_ic_is_perfect_when_score_perfectly_ranks_outcomes():
    result = simulate_top_k_strategy(_frame(), horizon_days=5, top_k=2, cost_model=ZERO_COST)
    assert result.ic_by_date.loc["d1"] == pytest.approx(1.0)
    assert result.ic_by_date.loc["d2"] == pytest.approx(1.0)


def test_simulate_drops_rows_with_unresolved_outcomes():
    df = _frame()
    extra = pd.DataFrame(
        [{"date": "d1", "symbol": "D", "score": 99, "forward_return": np.nan, "benchmark_forward_return": 0.02}]
    )
    df = pd.concat([df, extra], ignore_index=True)

    result = simulate_top_k_strategy(df, horizon_days=5, top_k=2, cost_model=ZERO_COST)
    # D had the highest score but an unresolved return -- must not be selected,
    # so d1's result is unchanged from the baseline (still A and C).
    assert result.per_period_returns.loc["d1"] == pytest.approx(0.04)


def test_simulate_metrics_dict_has_expected_keys():
    result = simulate_top_k_strategy(_frame(), horizon_days=5, top_k=2, cost_model=ZERO_COST)
    expected_keys = {"cagr", "sharpe", "sortino", "calmar", "max_drawdown", "win_rate", "n_periods"}
    assert expected_keys.issubset(result.metrics.keys())
    assert expected_keys.issubset(result.benchmark_metrics.keys())
    assert result.metrics["n_periods"] == 2


def test_simulate_empty_frame_returns_empty_result_not_a_crash():
    empty = pd.DataFrame(columns=["date", "symbol", "score", "forward_return", "benchmark_forward_return"])
    result = simulate_top_k_strategy(empty, horizon_days=5, top_k=2, cost_model=ZERO_COST)
    assert result.per_period_returns.empty
    assert result.metrics["n_periods"] == 0
    assert np.isnan(result.metrics["sharpe"])


def test_turnover_is_full_on_first_period_and_partial_when_holdings_overlap():
    df = pd.DataFrame(
        {
            "date": ["d1", "d1", "d1", "d1", "d2", "d2", "d2", "d2"],
            "symbol": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "score": [4, 3, 2, 1, 1, 4, 3, 2],
            "forward_return": [0.10, 0.08, 0.06, 0.04, 0.01, 0.09, 0.07, 0.05],
            "benchmark_forward_return": [0.02] * 8,
        }
    )
    cost_model = CostModel(brokerage_bps=0, stt_bps=0, exchange_txn_bps=0, gst_bps=0, stamp_duty_bps=0, slippage_bps=50.0)
    result = simulate_top_k_strategy(df, horizon_days=5, top_k=2, cost_model=cost_model)

    # d1: top-2 = {A, B} (first period, no prior holdings) -> turnover=1.0.
    assert result.turnover_by_date.loc["d1"] == pytest.approx(1.0)
    assert result.per_period_returns.loc["d1"] == pytest.approx(0.09 - 0.01)

    # d2: top-2 by score = {B, C}; overlap with {A, B} is {B} -> turnover=0.5.
    assert result.turnover_by_date.loc["d2"] == pytest.approx(0.5)
    assert result.per_period_returns.loc["d2"] == pytest.approx(0.08 - 0.5 * 0.01)


def test_universe_returns_is_mean_of_all_resolved_symbols_not_just_top_k():
    result = simulate_top_k_strategy(_frame(), horizon_days=5, top_k=2, cost_model=ZERO_COST)
    # d1: mean(0.05, 0.01, 0.03) across all 3 symbols, not just the top-2.
    assert result.universe_returns.loc["d1"] == pytest.approx((0.05 + 0.01 + 0.03) / 3)
    assert result.universe_returns.loc["d2"] == pytest.approx((0.02 + 0.06 + 0.04) / 3)


def test_hysteresis_band_keeps_an_existing_holding_outside_the_strict_top_k():
    df = pd.DataFrame(
        {
            "date": ["d1"] * 5 + ["d2"] * 5,
            "symbol": ["A", "B", "C", "D", "E"] * 2,
            "score": [5, 4, 3, 2, 1, 3, 5, 4, 2, 1],  # d2: A drops to 3rd place
            "forward_return": [0.10, 0.08, 0.06, 0.04, 0.02, 0.03, 0.09, 0.07, 0.02, 0.01],
            "benchmark_forward_return": [0.01] * 10,
        }
    )

    strict = simulate_top_k_strategy(df, horizon_days=5, top_k=1, cost_model=ZERO_COST)
    # Without hysteresis, d2's strict top-1 by score is B -> return 0.09.
    assert strict.per_period_returns.loc["d2"] == pytest.approx(0.09)

    with_band = simulate_top_k_strategy(
        df, horizon_days=5, top_k=1, cost_model=ZERO_COST, hysteresis_band=3.0
    )
    # With hysteresis_band=3.0, A (held from d1, and still within the top-3
    # on d2) is kept instead of being sold for B -> return is A's, 0.03.
    assert with_band.per_period_returns.loc["d2"] == pytest.approx(0.03)


def test_select_rebalance_dates_keeps_every_nth_trading_day():
    df = pd.DataFrame({"date": list(range(10)), "value": range(10)})
    out = select_rebalance_dates(df, every_n_trading_days=5)
    assert sorted(out["date"].unique()) == [0, 5]


def test_select_rebalance_dates_keeps_all_rows_on_kept_dates():
    # 6 distinct trading days (0..5) so every-5th selection lands on both 0 and 5.
    dates = [0, 0, 1, 2, 3, 4, 5, 5]
    symbols = ["A", "B", "A", "A", "A", "A", "A", "B"]
    df = pd.DataFrame({"date": dates, "symbol": symbols})
    out = select_rebalance_dates(df, every_n_trading_days=5)
    assert set(out["date"].unique()) == {0, 5}
    assert len(out) == 4  # both symbols on date 0 and date 5, dates 1-4 dropped


def test_select_rebalance_dates_prevents_overlapping_window_double_counting():
    """The bug this exists to prevent: daily rows carrying an overlapping
    5-day forward return, fed straight into simulate_top_k_strategy, would
    compound the same underlying price move up to 5 times. Subsampling to
    every 5th trading day fixes it."""
    dates = list(range(20))
    # Same symbol, same score every day; forward_return also constant --
    # if the same economic move were being recompounded across 5 overlapping
    # windows, daily vs subsampled results would differ sharply.
    daily = pd.DataFrame(
        {
            "date": dates,
            "score": [1.0] * 20,
            "forward_return": [0.05] * 20,
            "benchmark_forward_return": [0.02] * 20,
        }
    )
    subsampled = select_rebalance_dates(daily, every_n_trading_days=5)
    assert len(subsampled) == 4  # dates 0, 5, 10, 15

    result = simulate_top_k_strategy(subsampled, horizon_days=5, top_k=1, cost_model=ZERO_COST)
    assert result.metrics["n_periods"] == 4  # not 20 -- overlap correctly removed
