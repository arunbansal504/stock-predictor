"""Forward-return label construction (§6, §27 Phase 1 step 7).

Labels are built from FUTURE price data by definition -- that's what makes
them labels. The discipline that matters (§25; Lopez de Prado's "embargo"
concept) is keeping that future-knowledge cleanly separated from what a
model is allowed to train on as of any given date: a label decided on date T
with horizon H is not actually *resolved* (fully knowable) until trading day
T+H. `label_valid_date` records that resolution date so the model layer's
walk-forward CV can enforce an embargo -- never training on a label whose
label_valid_date falls after the training cutoff, even though the label's
*decision* date T is in the past. Skipping this is a classic, subtle way
walk-forward backtests still leak.

Primary target definition (§6): cross-sectional excess return vs. a
benchmark index, computed as stock forward return minus benchmark forward
return over the same window. This is a simple market-relative return, not a
full beta-neutralized alpha -- a documented simplification, not a hidden one.
"""

from __future__ import annotations

import pandas as pd


def compute_forward_return(
    df: pd.DataFrame, horizon_days: int, price_col: str = "close_adj"
) -> pd.Series:
    """Forward return over `horizon_days` *trading days* (row position, not
    calendar days -- consistent with features/technical.py), computed on a
    single series sorted ascending by date."""
    return df[price_col].shift(-horizon_days) / df[price_col] - 1.0


def build_labels_for_symbol(
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    horizons: dict[str, int],
    price_col: str = "close_adj",
    benchmark_price_col: str = "close",
) -> pd.DataFrame:
    """Build forward-return / excess-return labels for one symbol across
    multiple named horizons (e.g. {"5d": 5, "30d": 30}), aligned against a
    benchmark series (see ingestion/macro.py) on trading date.

    Rows within `horizon_days` of the end of history get NaN labels (not
    enough future data to resolve them yet) -- an honest gap, not a
    fabricated value.
    """
    price_df = price_df.sort_values("date").reset_index(drop=True)
    bench_df = (
        benchmark_df.sort_values("date")
        .reset_index(drop=True)[["date", benchmark_price_col]]
        .rename(columns={benchmark_price_col: "_bench_price"})
    )
    merged = price_df.merge(bench_df, on="date", how="inner")

    blocks = []
    for name, h in horizons.items():
        stock_fwd = compute_forward_return(merged, h, price_col=price_col)
        bench_fwd = compute_forward_return(merged, h, price_col="_bench_price")
        excess = stock_fwd - bench_fwd
        outperform = excess.gt(0).astype("boolean").mask(excess.isna())
        label_valid_date = merged["date"].shift(-h)  # NaT if beyond available history

        blocks.append(
            pd.DataFrame(
                {
                    "symbol": merged["symbol"],
                    "date": merged["date"],
                    "horizon": name,
                    "horizon_days": h,
                    "forward_return": stock_fwd,
                    "benchmark_forward_return": bench_fwd,
                    "excess_return": excess,
                    "outperform": outperform,
                    "label_valid_date": label_valid_date,
                }
            )
        )

    return pd.concat(blocks, ignore_index=True)
