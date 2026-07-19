"""Ranking engine (§10): turns per-symbol scores into a ranked table, after
liquidity / data-completeness / anomaly filters remove names that shouldn't
be recommended regardless of score.

Phase 1 keeps these filters simple and rule-based -- Isolation Forest-based
anomaly detection is explicitly deferred to Phase 3 in the model verdict
table (§8: "not for ranking -- for data-quality/anomaly flags"). A
rule-based liquidity/anomaly floor is what "basic" ranking-engine filtering
means for this phase; it should be replaced, not just supplemented, once a
real anomaly model exists.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.common.pit import filter_as_of
from stockpredictor.common.types import DataLayer
from stockpredictor.storage.lake import Lake

# INR, trailing-20-trading-day median daily turnover (close_adj * volume).
# Deliberately low for the Phase 1 large-cap seed universe -- this is a floor
# to catch genuinely illiquid/broken data, not a real institutional liquidity
# threshold (which would be sized against the strategy's intended capital).
DEFAULT_MIN_TURNOVER = 1_000_000.0
DEFAULT_MAX_ABS_DAILY_RETURN = 0.20  # flag a >20% single-day move as a likely data anomaly


def compute_liquidity_and_anomaly_flags(
    lake: Lake, window: int = 20, as_of: dt.date | None = None
) -> pd.DataFrame:
    """Per-symbol, as-of-latest-date liquidity (median turnover), a same-day
    anomaly flag, and the raw closing price (`close_price`) -- derived
    purely from silver price data, and the one place this pipeline attaches
    an actual quoted market price to a ranked/scored row (score/rank alone
    don't say what price the stock was ranked at). Deliberately the raw
    `close`, not `close_adj`: a user checking this against a live quote
    wants the actual traded price, not the backward-adjusted series the
    model trains on internally.

    `as_of`, when given, drops price rows after that date first -- same
    rationale as features/registry.py's `as_of` (common/trading_calendar.py)."""
    prices = lake.read_all(DataLayer.SILVER, "prices")
    if prices.empty:
        return pd.DataFrame()
    if as_of is not None:
        prices = filter_as_of(prices, pd.Timestamp(as_of))
        if prices.empty:
            return pd.DataFrame()

    rows = []
    for symbol, group in prices.groupby("symbol"):
        group = group.sort_values("date")
        turnover = group["close_adj"] * group["volume"]
        median_turnover = turnover.tail(window).median()

        daily_return = group["close_adj"].pct_change().iloc[-1] if len(group) > 1 else float("nan")
        is_anomalous = bool(abs(daily_return) > DEFAULT_MAX_ABS_DAILY_RETURN) if pd.notna(daily_return) else False

        rows.append(
            {
                "symbol": symbol,
                "close_price": float(group["close"].iloc[-1]),
                "median_turnover_20d": median_turnover,
                "latest_daily_return": daily_return,
                "is_price_anomaly": is_anomalous,
            }
        )
    return pd.DataFrame(rows)


def apply_ranking_filters(
    scored: pd.DataFrame,
    flags: pd.DataFrame,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
) -> pd.DataFrame:
    """Join the scored universe with liquidity/anomaly flags and drop
    symbols that fail the liquidity floor or show a same-day price anomaly.
    Symbols missing from `flags` (no price history) are dropped too -- a
    data-completeness floor by construction (an inner join, not left)."""
    merged = scored.merge(flags, on="symbol", how="inner")
    passed = merged[(merged["median_turnover_20d"] >= min_turnover) & (~merged["is_price_anomaly"])]
    return passed.reset_index(drop=True)


def rank_universe(
    filtered: pd.DataFrame, score_col: str = "score", tiebreak_col: str | None = "meta_score"
) -> pd.DataFrame:
    """Assign rank 1 = best (highest score).

    Ties on `score_col` are broken by `tiebreak_col` (when present), not by
    row order. `score` is calibrated via centered-isotonic interpolation
    (models/calibration.py), so exact ties are now the exception rather
    than the rule -- but they still happen for rows with identical raw
    model output, or rows clamped flat outside the fitted calibration
    range. Breaking those ties by row order would produce an ordering that
    looks meaningful but carries zero real information. `meta_score`
    (models/ensemble.py's pre-calibration, continuous meta-learner output)
    stays differentiated in exactly those cases, so use it instead. Falls
    back to row order only if `tiebreak_col` isn't present in `filtered` --
    callers that don't have it (e.g. tests using synthetic already-distinct
    scores) still work, they just don't need a tiebreak."""
    if filtered.empty:
        out = filtered.copy()
        out["rank"] = pd.Series(dtype="int64")
        return out
    out = filtered.copy()
    sort_cols = [score_col]
    if tiebreak_col and tiebreak_col in out.columns:
        sort_cols.append(tiebreak_col)
    out = out.sort_values(sort_cols, ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def top_n(ranked: pd.DataFrame, n: int) -> pd.DataFrame:
    return ranked[ranked["rank"] <= n].reset_index(drop=True)
