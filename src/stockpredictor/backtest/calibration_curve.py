"""Score -> realized-return calibration (§12): the honest source for a
portfolio's "expected return" figure.

Our model is a classifier (calibrated probability of outperformance),
never fit or calibrated to predict return MAGNITUDE -- so fabricating an
"expected return %" directly from the score would be false precision. This
computes what CAN be said honestly: given the walk-forward backtest's own
out-of-fold history, what was the *average realized* forward return for
stocks whose score fell in roughly the same range as today's candidate?
That's real evidence, not invented, and it's exactly what
portfolio/targets.py uses to label a "target price" as historically
informed, not guaranteed.
"""

from __future__ import annotations

import pandas as pd

CALIBRATION_COLUMNS = ["decile", "score_min", "score_max", "mean_return", "median_return", "n_obs"]


def compute_decile_return_calibration(
    scores: pd.Series, forward_returns: pd.Series, n_deciles: int = 10
) -> pd.DataFrame:
    """Bucket historical (score, forward_return) pairs into deciles by
    score and report each decile's realized return statistics. Returns a
    frame with columns CALIBRATION_COLUMNS, sorted by decile ascending
    (0 = lowest scores)."""
    df = pd.DataFrame({"score": scores, "forward_return": forward_returns}).dropna()
    if df.empty:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)

    df["decile"] = pd.qcut(df["score"], n_deciles, labels=False, duplicates="drop")
    grouped = df.groupby("decile").agg(
        score_min=("score", "min"),
        score_max=("score", "max"),
        mean_return=("forward_return", "mean"),
        median_return=("forward_return", "median"),
        n_obs=("forward_return", "count"),
    )
    return grouped.reset_index()[CALIBRATION_COLUMNS]


def lookup_expected_return(score: float, calibration: pd.DataFrame) -> float | None:
    """Map a live score to the calibration table's decile whose historical
    score range contains it. Falls back to the nearest decile boundary if
    the live score is outside every observed range (extrapolation handled
    by nearest-edge lookup, not a fabricated interpolation). Returns None
    if no calibration data is available at all."""
    if calibration.empty:
        return None

    within = calibration[(calibration["score_min"] <= score) & (score <= calibration["score_max"])]
    if not within.empty:
        return float(within.iloc[0]["mean_return"])

    calibration = calibration.copy()
    calibration["_dist"] = calibration.apply(
        lambda r: min(abs(score - r["score_min"]), abs(score - r["score_max"])), axis=1
    )
    nearest = calibration.loc[calibration["_dist"].idxmin()]
    return float(nearest["mean_return"])
