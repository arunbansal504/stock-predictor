"""Backtest performance metrics (§25).

All ratio metrics (Sharpe/Sortino/Calmar) are computed on a per-rebalance
return series and annualized using `periods_per_year` (derived from the
horizon's trading-day length) rather than assuming daily returns -- reusing
a "daily Sharpe" formula on a multi-day holding-period strategy is a common,
silent source of wildly wrong ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def periods_per_year(horizon_days: int) -> float:
    return TRADING_DAYS_PER_YEAR / horizon_days


def cagr(returns: pd.Series, horizon_days: int) -> float:
    """Compound annual growth rate from a series of per-rebalance returns."""
    if len(returns) == 0:
        return float("nan")
    growth = (1 + returns).prod()
    years = len(returns) / periods_per_year(horizon_days)
    if years <= 0 or growth <= 0:
        return float("nan")
    return growth ** (1 / years) - 1


def sharpe_ratio(returns: pd.Series, horizon_days: int, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / periods_per_year(horizon_days)
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return (excess.mean() / std) * np.sqrt(periods_per_year(horizon_days))


def sortino_ratio(returns: pd.Series, horizon_days: int, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / periods_per_year(horizon_days)
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year(horizon_days))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the cumulative equity curve, as a
    negative fraction (e.g. -0.25 for a 25% drawdown)."""
    if len(returns) == 0:
        return float("nan")
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def max_gain(returns: pd.Series) -> float:
    """Maximum trough-to-peak run-up of the cumulative equity curve, as a
    positive fraction -- the mirror of `max_drawdown`, same equity-curve
    construction, just tracking the running minimum instead of the running
    maximum."""
    if len(returns) == 0:
        return float("nan")
    equity = (1 + returns).cumprod()
    running_min = equity.cummin()
    runup = equity / running_min - 1.0
    return float(runup.max())


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, horizon_days: int) -> float:
    """Annualized mean/std of active return (`returns - benchmark_returns`)
    -- mathematically `sharpe_ratio` applied to the active-return series
    with a zero risk-free rate, which is exactly what IR is."""
    active = (returns - benchmark_returns).dropna()
    return sharpe_ratio(active, horizon_days, risk_free_rate=0.0)


def calmar_ratio(returns: pd.Series, horizon_days: int) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    c = cagr(returns, horizon_days)
    if np.isnan(c):
        return float("nan")
    return c / abs(mdd)


def win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).mean())


def hit_rate_by_decile(scores: pd.Series, outcomes: pd.Series) -> pd.Series:
    """Fraction of positive outcomes within each score decile -- a sanity
    check, not just a metric (§25, §30): top deciles should show a higher
    hit rate than bottom deciles if the model has real signal. If every
    decile looks equally great, that is a leakage red flag, not a win.
    """
    df = pd.DataFrame({"score": scores, "outcome": outcomes}).dropna()
    if df.empty:
        return pd.Series(dtype="float64")
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")
    return df.groupby("decile")["outcome"].mean()


def information_coefficient(scores: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman rank correlation between predicted score and realized
    forward return -- the standard "IC" metric for a ranking model's skill
    on a single date's cross-section."""
    df = pd.DataFrame({"score": scores, "forward_return": forward_returns}).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df["score"].corr(df["forward_return"], method="spearman"))
