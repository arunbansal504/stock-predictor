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
    # Equal-weight mean return across every resolved symbol in the universe
    # that date (not just the Top-K) -- the "just hold everything" baseline.
    # Top-K equal-weight vs. a cap-weighted benchmark index bakes in a
    # concentration/size bet that has nothing to do with ranking skill; this
    # isolates that from "did the ranking actually add value over holding
    # the whole eligible universe."
    universe_returns: pd.Series
    ic_by_date: pd.Series
    # Fraction of the Top-K portfolio that changed vs. the prior period
    # (see simulate_top_k_strategy's docstring) -- what net_of_costs
    # actually charged each period, not assumed.
    turnover_by_date: pd.Series
    metrics: dict
    benchmark_metrics: dict
    universe_metrics: dict


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
    symbol_col: str = "symbol",
    hysteresis_band: float | None = None,
) -> BacktestResult:
    """`scored_test_frame` must have one row per (symbol, date) with a
    predicted `score_col` and the realized `return_col` / `benchmark_return_col`
    (see labels/returns.py). Rows with an unresolved outcome (NaN) are
    dropped per rebalance date before that date is scored -- an honest gap
    in history, not something to impute a value for.

    Turnover-aware costs: the round-trip cost charged each period is scaled
    by that period's turnover -- the fraction of the Top-K portfolio that
    actually changed vs. the prior period's holdings (0 = identical
    holdings, 1 = completely different) -- rather than a full round trip
    every period regardless of overlap. A symbol that stays selected across
    two consecutive rebalances isn't sold and rebought, so it shouldn't pay
    a round-trip cost again. The first period always has turnover=1.0
    (starting from cash, nothing to hold over), matching the previous
    unconditional-full-cost behavior. Requires `symbol_col` to identify
    which positions persist across periods; if absent, turnover is always
    1.0 (the old behavior) since holdings can't be tracked.

    `hysteresis_band`, when given (e.g. 2.0), lets an already-held position
    stay in the portfolio as long as its rank that period is within
    `top_k * band` (not just the strict Top-K), filling any remaining slots
    with the next-highest-scored new names. This deliberately trades a
    small amount of ranking purity for materially lower turnover -- a stock
    hovering just outside the Top-K on score noise alone doesn't need to be
    sold and immediately different money re-bought next period. Disabled
    (strict Top-K every period) by default.
    """
    cost_model = cost_model or CostModel()

    period_returns: list[float] = []
    bench_returns: list[float] = []
    universe_mean_returns: list[float] = []
    ic_values: list[float] = []
    turnovers: list[float] = []
    dates: list = []

    prior_holdings: set = set()

    for date, group in scored_test_frame.groupby(date_col):
        group = group.dropna(subset=[score_col, return_col, benchmark_return_col])
        if group.empty:
            continue

        dates.append(date)
        # Spearman rank correlation is invariant to subtracting a per-date
        # constant (the benchmark's forward return is the same value for
        # every row this date), so IC computed against raw forward_return
        # is already numerically identical to IC against excess_return --
        # nothing to "align" here despite that sounding like a plausible fix.
        ic_values.append(information_coefficient(group[score_col], group[return_col]))
        universe_mean_returns.append(group[return_col].mean())

        ranked = group.sort_values(score_col, ascending=False, kind="stable")
        top = ranked.head(top_k)

        if hysteresis_band is not None and prior_holdings and symbol_col in ranked.columns:
            band_size = max(top_k, int(top_k * hysteresis_band))
            eligible = ranked.head(band_size)
            kept = eligible[eligible[symbol_col].isin(prior_holdings)]
            new_names = ranked[~ranked[symbol_col].isin(set(kept[symbol_col]))].head(
                max(top_k - len(kept), 0)
            )
            top = pd.concat([kept, new_names]).head(top_k)

        gross_return = top[return_col].mean() if not top.empty else np.nan

        current_holdings = set(top[symbol_col]) if symbol_col in top.columns else set()
        if current_holdings or prior_holdings:
            denom = len(current_holdings) or top_k
            overlap = len(current_holdings & prior_holdings)
            turnover = (len(current_holdings) - overlap) / denom
        else:
            turnover = 1.0
        turnovers.append(turnover)
        prior_holdings = current_holdings

        net_return = (
            gross_return - turnover * (cost_model.round_trip_cost_bps() / 10_000.0)
            if not np.isnan(gross_return)
            else np.nan
        )
        period_returns.append(net_return)
        bench_returns.append(group[benchmark_return_col].mean())

    index = pd.Index(dates, name=date_col)
    per_period_raw = pd.Series(period_returns, index=index)
    bench_raw = pd.Series(bench_returns, index=index)
    universe_raw = pd.Series(universe_mean_returns, index=index)
    ic_series = pd.Series(ic_values, index=index)
    turnover_series = pd.Series(turnovers, index=index)

    per_period = per_period_raw.dropna()
    bench = bench_raw.reindex(per_period.index)
    universe = universe_raw.reindex(per_period.index)

    return BacktestResult(
        per_period_returns=per_period,
        benchmark_returns=bench,
        universe_returns=universe,
        ic_by_date=ic_series,
        turnover_by_date=turnover_series.reindex(per_period.index),
        metrics=_compute_metrics(per_period, horizon_days),
        benchmark_metrics=_compute_metrics(bench, horizon_days),
        universe_metrics=_compute_metrics(universe, horizon_days),
    )
