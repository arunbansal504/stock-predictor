"""Fundamental/Quality feature block (§7), joining PIT-aligned annual
fundamental snapshots against daily prices.

PE and PB are price-dependent and change daily even though the underlying
earnings/book value only updates ~once a year -- computed here, not stored
as static fields on the fundamentals connector output.

`revenue_growth_yoy`/`eps_growth_yoy` are deliberately NOT in
FUNDAMENTAL_FEATURE_COLUMNS (excluded from the model-facing feature set),
though `compute_fundamental_ratios` still computes them. Empirically, not
theoretically: a live 500-stock/5-year backtest with them included showed
mean IC drop from 0.037 (technical-only) to 0.015 -- and yfinance's annual
statements only go back ~5 years, almost exactly the backtest window
itself, so YoY growth (which needs 2 prior years) was entirely NaN across
most of the early walk-forward folds (confirmed via a live sklearn
imputation warning: "no non-missing value" for these columns in those
folds). Feeding a model mostly-empty columns diluted its broad-universe
discrimination rather than adding signal. Revisit once fundamentals history
is deep enough to actually populate them -- this is a data-depth problem,
not a reason growth ratios are inherently unhelpful.

Uses `pd.merge_asof` (not a per-row Python loop calling an as-of lookup) for
the PIT join -- a symbol has ~4-5 years of fundamental snapshots but ~1250
daily price rows over the same window; a naive row-by-row join across a
500-symbol universe would mean hundreds of thousands of individual lookups
and take minutes-to-hours. `merge_asof` is the vectorized, C-implemented
equivalent of "most recent knowable row <= this date" and runs in
milliseconds per symbol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNDAMENTAL_FEATURE_COLUMNS: list[str] = [
    "pe_ratio",
    "pb_ratio",
    "roe",
    "roa",
    "debt_to_equity",
    "net_margin",
]

_SNAPSHOT_COLUMNS = [
    "knowable_date",
    "eps",
    "book_value_per_share",
    "roe",
    "roa",
    "debt_to_equity",
    "net_margin",
]


def compute_fundamental_ratios(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ratio columns to a single symbol's fundamentals frame
    (one row per fiscal year, sorted by period_end). Growth ratios need the
    prior year's row, hence computed here across the whole per-symbol
    series rather than per-row in isolation."""
    df = fundamentals_df.sort_values("period_end").reset_index(drop=True)
    df["book_value_per_share"] = df["total_equity"] / df["shares_outstanding"]
    df["roe"] = df["net_income"] / df["total_equity"]
    df["roa"] = df["net_income"] / df["total_assets"]
    df["debt_to_equity"] = df["total_debt"] / df["total_equity"]
    df["net_margin"] = df["net_income"] / df["revenue"]
    df["revenue_growth_yoy"] = df["revenue"].pct_change()
    df["eps_growth_yoy"] = df["eps"].pct_change()
    return df


def build_fundamental_features_for_symbol(
    prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame
) -> pd.DataFrame:
    """For one symbol: PIT-align annual fundamentals to every trading date
    in `prices_df` via an as-of merge, then compute price-dependent ratios
    (PE, PB) daily. Dates before the first fundamental snapshot was
    knowable get NaN fundamental features -- an honest start-of-history
    gap, not a fabricated placeholder."""
    base = prices_df[["symbol", "date", "close_adj"]].sort_values("date").reset_index(drop=True)
    # pandas 3.0 can produce different default datetime64 resolutions
    # depending on source (e.g. Parquet round-trip vs. pd.to_datetime on
    # plain date objects) -- merge_asof requires matching resolutions, not
    # just "both datetime64". Normalize explicitly rather than let the
    # resolution be whatever each source happened to pick.
    base["date"] = base["date"].astype("datetime64[ns]")

    if fundamentals_df.empty:
        out = base[["symbol", "date"]].copy()
        for col in FUNDAMENTAL_FEATURE_COLUMNS:
            out[col] = np.nan
        return out

    ratios = compute_fundamental_ratios(fundamentals_df)
    ratios["knowable_date"] = pd.to_datetime(ratios["knowable_date"]).astype("datetime64[ns]")
    ratios_sorted = ratios[_SNAPSHOT_COLUMNS].sort_values("knowable_date").reset_index(drop=True)

    merged = pd.merge_asof(
        base,
        ratios_sorted,
        left_on="date",
        right_on="knowable_date",
        direction="backward",
    )

    merged["pe_ratio"] = merged["close_adj"] / merged["eps"].replace(0, np.nan)
    merged["pb_ratio"] = merged["close_adj"] / merged["book_value_per_share"].replace(0, np.nan)

    return merged[["symbol", "date"] + FUNDAMENTAL_FEATURE_COLUMNS]
