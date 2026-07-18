"""Cross-sectional transforms (§7): "for most raw features, add the
cross-sectional rank / z-score on each date -- this is what makes it a
*relative* ranking model and removes market-wide drift. This single
discipline matters more than adding 200 raw indicators."

A raw RSI of 70 means something different in a raging bull market (everything
is overbought) vs. a selloff (only real strength shows RSI 70). Ranking each
feature against the rest of the universe *on that same date* neutralizes the
market-wide level and leaves only the relative signal, which is what a
cross-sectional ranking model (§6, §10) actually needs.
"""

from __future__ import annotations

import pandas as pd


def add_cross_sectional_rank(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    suffix: str = "_xrank",
) -> pd.DataFrame:
    """Add a `{col}{suffix}` percentile rank (0-1, NaN-safe) for each feature,
    computed within each `date_col` group across the whole universe present
    in `df`. `df` must already contain every symbol for a given date for the
    rank to be meaningful (see storage/lake.py Lake.read_all)."""
    out = df.copy()
    grouped = out.groupby(date_col)[feature_cols]
    ranks = grouped.rank(pct=True, na_option="keep")
    ranks.columns = [f"{c}{suffix}" for c in feature_cols]
    return pd.concat([out, ranks], axis=1)


def add_cross_sectional_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    suffix: str = "_xz",
) -> pd.DataFrame:
    """Add a `{col}{suffix}` z-score for each feature, computed within each
    `date_col` group. Rows are the unit of the transform (one date's
    cross-section), not the whole time series -- do not call this per-symbol."""
    out = df.copy()
    grouped = out.groupby(date_col)[feature_cols]
    means = grouped.transform("mean")
    stds = grouped.transform("std")
    z = (out[feature_cols] - means) / stds.replace(0, pd.NA)
    z.columns = [f"{c}{suffix}" for c in feature_cols]
    return pd.concat([out, z], axis=1)
