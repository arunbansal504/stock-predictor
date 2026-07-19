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

Fits isotonic regression (non-decreasing) directly on raw (score,
forward_return) pairs -- NOT a fixed decile pre-binning (an earlier
version of this module used `pd.qcut(..., 10)`). Two real problems with
fixed deciles, both observed live: (1) per-decile means computed
independently of each other can be non-monotonic from pure sampling
noise -- one 90d backtest run showed the LOWEST-scored decile with a
HIGHER historical mean return (20.4%) than the HIGHEST-scored decile
(16.9%), which silently made a more-diversified, lower-conviction
portfolio show a HIGHER expected return than a concentrated,
high-conviction one -- backwards from what the model's own confidence
ranking should imply. (2) a fixed bucket count claims a fixed resolution
regardless of what the data supports -- when most candidates cluster in
one wide top decile (e.g. covering scores 0.52-0.96), every one of them
gets the exact same historical average, which looks identical to the
score-collapse bug fixed in models/calibration.py even though the
mechanism here is different.

Isotonic regression's own block-finding (Pool Adjacent Violators) instead
produces as many or as few blocks as the real relationship in this
backtest's history actually supports -- fewer, wider blocks where it's
weak/noisy, more where it's genuinely informative -- while GUARANTEEING
mean_return is non-decreasing block to block, which must hold
structurally if the score means anything at all. `lookup_expected_return`
then interpolates continuously between block centers (same technique as
models/calibration.py's IsotonicCalibrator.transform), so scores within a
wide block aren't all flattened to one identical value the way a step
lookup would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

CALIBRATION_COLUMNS = ["block", "score_min", "score_max", "mean_return", "median_return", "n_obs"]


def compute_return_calibration(scores: pd.Series, forward_returns: pd.Series) -> pd.DataFrame:
    """Bucket historical (score, forward_return) pairs into isotonic
    (non-decreasing) blocks and report each block's return statistics.
    `mean_return` is the isotonic-fitted value for that block (guaranteed
    non-decreasing across rows, sorted by score_min) -- this is what
    `lookup_expected_return` actually uses. `median_return` is the block's
    raw (unconstrained) median, reported alongside for transparency only --
    a heavily skewed block's raw median can differ meaningfully from its
    isotonic mean, and hiding that would itself be a kind of false
    precision."""
    df = pd.DataFrame({"score": scores, "forward_return": forward_returns}).dropna()
    if df.empty:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)

    raw = df["score"].to_numpy(dtype=float)
    y = df["forward_return"].to_numpy(dtype=float)

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(raw, y)

    order = np.argsort(raw, kind="stable")
    raw_sorted = raw[order]
    y_sorted = y[order]
    fitted_sorted = iso.predict(raw_sorted)

    # A new block starts wherever the isotonic-fitted value changes --
    # PAVA's pooled ("flat") regions are exactly the runs where it doesn't.
    # Same convention as models/calibration.py's IsotonicCalibrator.
    block_id = np.concatenate(([0], np.cumsum(np.diff(fitted_sorted) != 0)))

    block_df = pd.DataFrame({"score": raw_sorted, "fitted": fitted_sorted, "y": y_sorted, "block": block_id})
    grouped = block_df.groupby("block", sort=True)

    result = pd.DataFrame(
        {
            "block": grouped.size().index.to_numpy(),
            "score_min": grouped["score"].min().to_numpy(),
            "score_max": grouped["score"].max().to_numpy(),
            "mean_return": grouped["fitted"].first().to_numpy(),
            "median_return": grouped["y"].median().to_numpy(),
            "n_obs": grouped["y"].size().to_numpy(),
        }
    )
    return result[CALIBRATION_COLUMNS].sort_values("score_min").reset_index(drop=True)


def lookup_expected_return(score: float, calibration: pd.DataFrame) -> float | None:
    """Interpolates a live score against a fitted return-calibration table
    (see compute_return_calibration): linear interpolation between each
    block's center -- the midpoint of that block's score_min/score_max,
    a simple and honest-enough anchor that avoids needing an extra
    persisted column -- and its (already isotonic, non-decreasing)
    mean_return, clamped to the nearest block's mean_return outside the
    fitted range (extrapolation via nearest edge, not fabricated beyond
    the data). Because mean_return is non-decreasing by construction,
    interpolating between blocks can never invert the model's own
    confidence ranking the way a raw per-decile lookup could. Returns
    None if no calibration data is available at all."""
    if calibration.empty:
        return None
    ordered = calibration.sort_values("score_min")
    centers = ((ordered["score_min"] + ordered["score_max"]) / 2).to_numpy()
    values = ordered["mean_return"].to_numpy()
    if len(centers) == 0:
        return None
    return float(np.interp(score, centers, values))
