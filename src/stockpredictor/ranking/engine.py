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

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.storage.lake import Lake

# INR, trailing-20-trading-day median daily turnover (close_adj * volume).
# Deliberately low for the Phase 1 large-cap seed universe -- this is a floor
# to catch genuinely illiquid/broken data, not a real institutional liquidity
# threshold (which would be sized against the strategy's intended capital).
DEFAULT_MIN_TURNOVER = 1_000_000.0
DEFAULT_MAX_ABS_DAILY_RETURN = 0.20  # flag a >20% single-day move as a likely data anomaly


def compute_liquidity_and_anomaly_flags(lake: Lake, window: int = 20) -> pd.DataFrame:
    """Per-symbol, as-of-latest-date liquidity (median turnover) and a
    same-day anomaly flag, derived purely from silver price data."""
    prices = lake.read_all(DataLayer.SILVER, "prices")
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


def rank_universe(filtered: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Assign rank 1 = best (highest score); ties broken by row order."""
    if filtered.empty:
        out = filtered.copy()
        out["rank"] = pd.Series(dtype="int64")
        return out
    out = filtered.copy()
    out["rank"] = out[score_col].rank(method="first", ascending=False).astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def top_n(ranked: pd.DataFrame, n: int) -> pd.DataFrame:
    return ranked[ranked["rank"] <= n].reset_index(drop=True)
