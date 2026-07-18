"""Walk-forward backtest engine (§25, §27 step 9).

Simulates a simple, honest strategy -- equal-weight Top-K by predicted
score, rebalanced at each horizon's cadence -- over a scored test frame
(typically the concatenation of every walk-forward fold's test predictions,
see models/walk_forward.py). This intentionally stops short of the full
Portfolio Optimizer (HRP, risk profiles, sector caps -- §12, Phase 2):
Phase 1's job is proving the *ranking* has real out-of-sample skill before
building allocation logic on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpredictor.backtest.costs import CostModel, net_of_costs
from stockpredictor.backtest.metrics import (
    cagr,
    calmar_ratio,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


METRIC_NAMES: tuple[str, ...] = ("cagr", "sharpe", "sortino", "calmar", "max_drawdown", "win_rate", "n_periods")


def select_rebalance_dates(
    df: pd.DataFrame, every_n_trading_days: int, date_col: str = "date"
) -> pd.DataFrame:
    """Filter to non-overlapping rebalance dates, spaced `every_n_trading_days`
    apart among the trading days present in `df`.

    Required before compounding metrics whenever `df` was built from daily
    rows carrying an *overlapping* forward-return window (e.g. a 5-day label
    computed fresh at every trading day -- see labels/returns.py). Feeding
    every daily row straight into `simulate_top_k_strategy` would compound
    the same underlying price move into the equity curve up to
    `every_n_trading_days` times, wildly distorting drawdown/Sharpe/CAGR --
    not a subtle effect, closer to 5x-overstated volatility for a 5-day
    horizon. This is what enforces config/backtest.yaml's
    `rebalance.cadence: horizon` in Phase 1's simplest strategy simulation.
    Intentionally a separate, explicit call rather than a hidden default
    inside `simulate_top_k_strategy` -- a genuinely daily-rebalanced
    strategy should NOT have this subsampling applied.
    """
    unique_dates = sorted(pd.unique(df[date_col]))
    rebalance_dates = set(unique_dates[::every_n_trading_days])
    return df[df[date_col].isin(rebalance_dates)]


@dataclass
class BacktestResult:
    per_period_returns: pd.Series  # net-of-cost strategy returns, one per rebalance date
    benchmark_returns: pd.Series
    ic_by_date: pd.Series
    metrics: dict
    benchmark_metrics: dict


def _compute_metrics(returns: pd.Series, horizon_days: int) -> dict:
    return {
        "cagr": cagr(returns, horizon_days),
        "sharpe": sharpe_ratio(returns, horizon_days),
        "sortino": sortino_ratio(returns, horizon_days),
        "calmar": calmar_ratio(returns, horizon_days),
        "max_drawdown": max_drawdown(returns),
        "win_rate": win_rate(returns),
        "n_periods": int(len(returns)),
    }


def simulate_top_k_strategy(
    scored_test_frame: pd.DataFrame,
    horizon_days: int,
    top_k: int = 10,
    cost_model: CostModel | None = None,
    score_col: str = "score",
    return_col: str = "forward_return",
    benchmark_return_col: str = "benchmark_forward_return",
    date_col: str = "date",
) -> BacktestResult:
    """`scored_test_frame` must have one row per (symbol, date) with a
    predicted `score_col` and the realized `return_col` / `benchmark_return_col`
    (see labels/returns.py). Rows with an unresolved outcome (NaN) are
    dropped per rebalance date before that date is scored -- an honest gap
    in history, not something to impute a value for.
    """
    cost_model = cost_model or CostModel()

    period_returns: list[float] = []
    bench_returns: list[float] = []
    ic_values: list[float] = []
    dates: list = []

    for date, group in scored_test_frame.groupby(date_col):
        group = group.dropna(subset=[score_col, return_col, benchmark_return_col])
        if group.empty:
            continue

        dates.append(date)
        ic_values.append(information_coefficient(group[score_col], group[return_col]))

        top = group.nlargest(top_k, score_col)
        gross_return = top[return_col].mean() if not top.empty else np.nan
        net_return = net_of_costs(gross_return, cost_model) if not np.isnan(gross_return) else np.nan
        period_returns.append(net_return)
        bench_returns.append(group[benchmark_return_col].mean())

    index = pd.Index(dates, name=date_col)
    per_period_raw = pd.Series(period_returns, index=index)
    bench_raw = pd.Series(bench_returns, index=index)
    ic_series = pd.Series(ic_values, index=index)

    per_period = per_period_raw.dropna()
    bench = bench_raw.reindex(per_period.index)

    return BacktestResult(
        per_period_returns=per_period,
        benchmark_returns=bench,
        ic_by_date=ic_series,
        metrics=_compute_metrics(per_period, horizon_days),
        benchmark_metrics=_compute_metrics(bench, horizon_days),
    )
