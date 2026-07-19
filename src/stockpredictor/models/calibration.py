"""Probability calibration (§6, §8): converts a classifier's raw scores into
honest probabilities via isotonic regression.

Must be fit on out-of-sample predictions -- never on the same rows a base
model trained on, or the calibration curve inherits the base model's
in-sample overconfidence and the whole point (§30: "reliability curve shows
predicted probabilities ~= realized frequencies") is lost.

`transform` does NOT return raw PAVA step-function output. Plain isotonic
regression pools ties into flat blocks, which -- for a weak signal -- can
collapse dozens of genuinely different raw scores onto one identical
calibrated value (see `_block_stats`/`separation_info` for that same block
structure, still used for confidence reporting). To keep the *displayed*
score differentiated while staying calibration-honest, `transform` instead
places each PAVA block's empirical rate at that block's mean raw score
("center") and linearly interpolates between adjacent block centers
(centered isotonic regression). Every knot value is still a real empirical
outcome rate -- nothing is fabricated -- but a row landing between two
block centers gets a distinct, honestly-interpolated value instead of being
flattened to its block's rate. `separation_info`'s `empirical_rate` remains
the pure block anchor, so `transform(raw)` and `separation_info(raw)`'s
`empirical_rate` are no longer identical by construction -- that divergence
is intentional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

# Two-sided significance level for "is this calibration block's empirical
# outperform rate distinguishable from this horizon's own base rate".
# Deliberately stricter than the conventional 0.05: isotonic's Pool Adjacent
# Violators routinely produces blocks with tens of thousands of calibration
# rows (see models/ensemble.py's meta_score docstring), and at that sample
# size a 0.05 threshold flags practically meaningless deviations as
# "significant" purely from sample size, not real separation.
SEPARATION_ALPHA = 0.01

# `separation_direction` values. The test is two-sided (H0: rate ==
# base_rate), so "statistically significant" alone does NOT mean "confirmed
# to outperform" -- a block can be just as significantly BELOW base_rate as
# above it, and that's a confirmed negative signal, not a weaker positive
# one. Collapsing both into one boolean (as an earlier version of this
# module did) made a significantly-underperforming block indistinguishable
# from a significantly-outperforming one to callers, which the UI then
# rendered with the same positive/green treatment -- a real bug, not a
# display nitpick. Keep the sign explicit everywhere downstream instead.
SEPARATION_OUTPERFORM = "outperform"
SEPARATION_UNDERPERFORM = "underperform"
SEPARATION_NONE = "none"


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False
        # One row per pooled calibration block (contiguous run of raw scores
        # mapped to the same calibrated value), sorted by raw score range.
        # Built once at fit time from the *calibration* set itself -- see
        # `_compute_block_stats` -- so "separation_direction" answers "was
        # there real historical evidence for this exact score band, and
        # which way did it point", not just "does the score look high".
        self._block_stats: pd.DataFrame | None = None
        # This horizon's own global outperform rate across the whole
        # calibration set -- the actual null hypothesis for
        # `separation_direction`, NOT a fixed 0.5. A cap-weighted benchmark's
        # constituents don't split 50/50 around it by construction (the
        # index return is pulled up by its largest names), so testing every
        # block against a hardcoded coin-flip conflates "beats this
        # horizon's typical constituent" with "beats an arbitrary 50%" --
        # the latter can flag nearly the entire universe as
        # "underperforming" purely because the population itself sits below
        # 50%, not because any individual block has real negative edge.
        # Recomputed on every `fit` call from that call's own calibration
        # set, never frozen across refits.
        self.base_rate: float | None = None
        # Interpolation knots for `transform` -- block centers (x) and each
        # block's empirical rate (y), derived from `_block_stats` at fit
        # time. See module docstring for why `transform` interpolates
        # instead of returning the raw PAVA step value.
        self._interp_x: np.ndarray | None = None
        self._interp_y: np.ndarray | None = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        raw_scores = np.asarray(raw_scores, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        self._iso.fit(raw_scores, y_true)
        self._fitted = True
        self.base_rate = float(y_true.mean())
        self._block_stats = self._compute_block_stats(raw_scores, y_true, self.base_rate)

        # Blocks are disjoint, ordered raw-score ranges (adjacent blocks
        # differ in calibrated value by construction -- see block_id below),
        # so centers are strictly increasing except in float-degenerate
        # edge cases; dedupe defensively rather than let `np.interp` receive
        # a non-increasing `xp`.
        centers = self._block_stats["raw_center"].to_numpy()
        rates = self._block_stats["calibrated_score"].to_numpy()
        centers_unique, first_idx = np.unique(centers, return_index=True)
        self._interp_x = centers_unique
        self._interp_y = rates[first_idx]
        return self

    def _compute_block_stats(self, raw_scores: np.ndarray, y_true: np.ndarray, base_rate: float) -> pd.DataFrame:
        order = np.argsort(raw_scores, kind="stable")
        raw_sorted = raw_scores[order]
        y_sorted = y_true[order]
        calibrated_sorted = self._iso.predict(raw_sorted)

        # A new block starts wherever the calibrated value changes -- PAVA's
        # pooled ("flat") regions are exactly the runs where it doesn't.
        block_id = np.concatenate(([0], np.cumsum(np.diff(calibrated_sorted) != 0)))

        df = pd.DataFrame(
            {"raw": raw_sorted, "calibrated": calibrated_sorted, "y": y_sorted, "block": block_id}
        )
        grouped = df.groupby("block", sort=True)
        n = grouped["y"].size().to_numpy()
        rate = grouped["y"].mean().to_numpy()
        # Wald test of H0: empirical rate == base_rate, against the
        # two-sided alternative, using the null's own variance
        # base_rate*(1-base_rate) -- not the conservative 0.5 substitute,
        # since the null itself is no longer 0.5.
        se = np.sqrt(base_rate * (1 - base_rate) / n)
        z = np.divide(rate - base_rate, se, out=np.zeros_like(rate), where=se > 0)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
        significant = p_value < SEPARATION_ALPHA
        direction = np.where(
            ~significant,
            SEPARATION_NONE,
            np.where(rate > base_rate, SEPARATION_OUTPERFORM, SEPARATION_UNDERPERFORM),
        )

        return pd.DataFrame(
            {
                "raw_lo": grouped["raw"].min().to_numpy(),
                "raw_hi": grouped["raw"].max().to_numpy(),
                # Block center (mean raw score) -- the interpolation knot
                # `transform` places this block's empirical rate at. Not
                # used by `separation_info` (which reports the block's raw
                # range and rate, not its center).
                "raw_center": grouped["raw"].mean().to_numpy(),
                "calibrated_score": grouped["calibrated"].first().to_numpy(),
                "n": n,
                "empirical_rate": rate,
                "p_value": p_value,
                "separation_direction": direction,
            }
        ).sort_values("raw_lo").reset_index(drop=True)

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """Centered-isotonic calibrated probability for each raw score --
        linear interpolation between PAVA block centers (see module
        docstring), clamped to each block's rate outside the outer centers
        and to [0, 1] overall. NOT the same as looking up each row's PAVA
        block value directly (that's `separation_info`'s `empirical_rate`)."""
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fit before transform")
        raw_scores = np.asarray(raw_scores, dtype=float)
        out = np.interp(raw_scores, self._interp_x, self._interp_y)
        return np.clip(out, 0.0, 1.0)

    @property
    def block_stats(self) -> pd.DataFrame:
        """Copy of the fitted PAVA block table (raw_lo/raw_hi/raw_center/
        calibrated_score/n/empirical_rate/p_value/separation_direction) --
        for diagnostics/inspection, not consumed by `transform` callers."""
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fit before block_stats")
        return self._block_stats.copy()

    def fit_transform(self, raw_scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(raw_scores, y_true).transform(raw_scores)

    def separation_info(self, raw_scores: np.ndarray) -> pd.DataFrame:
        """Per-row calibration-block membership: how many calibration-set
        rows shared this exact score band (`n`), what fraction of them
        actually outperformed (`empirical_rate`), this horizon's own global
        base rate (`base_rate` -- the actual null, repeated per row for
        convenience), and whether `empirical_rate` is statistically
        distinguishable from `base_rate` at `SEPARATION_ALPHA` -- and if so,
        in which direction (`separation_direction`, one of
        `SEPARATION_OUTPERFORM`/`SEPARATION_UNDERPERFORM`/`SEPARATION_NONE`).
        A significantly-below-base_rate block is a confirmed *negative*
        signal, not a weaker positive one -- callers must branch on
        direction, not just treat any significant block as good news. Each
        query row is matched to the block whose raw-score range it falls
        into (or the nearest one above it, if it lands in a gap between two
        singleton blocks -- isotonic's pooled/unpooled regions don't leave
        gaps in coverage, only in exact value matches for scores that
        weren't seen during calibration fitting)."""
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fit before separation_info")
        raw_scores = np.asarray(raw_scores, dtype=float)
        raw_hi = self._block_stats["raw_hi"].to_numpy()
        idx = np.searchsorted(raw_hi, raw_scores, side="left")
        idx = np.clip(idx, 0, len(raw_hi) - 1)
        matched = self._block_stats.iloc[idx].reset_index(drop=True)
        matched = matched[["n", "empirical_rate", "p_value", "separation_direction"]].copy()
        matched["base_rate"] = self.base_rate
        return matched

    @staticmethod
    def separation_badge(direction: str, empirical_rate: float, n: int, base_rate: float) -> dict:
        """Human-readable label + UI style for one row's `separation_direction`.
        Centralized here (not re-derived independently in each UI surface)
        specifically so a significant below-base_rate block cannot end up
        labeled or styled as positive in one place while correctly handled
        in another -- see test_calibration.py's direction-blindness
        regression test for exactly the bug this guards against.

        The label always states `empirical_rate` relative to `base_rate`,
        not the raw rate alone -- since base_rate isn't 0.5, a rate like
        47% can be a confirmed GOOD result (if base_rate is 40%) or a
        confirmed BAD one (if base_rate is 50%), and the raw number by
        itself no longer tells a reader which."""
        comparison = f"{empirical_rate:.1%} vs {base_rate:.1%} horizon base rate"
        if direction == SEPARATION_OUTPERFORM:
            return {
                "style": "positive",
                "label": f"Statistically confirmed outperformance historically ({comparison}, n={n})",
            }
        if direction == SEPARATION_UNDERPERFORM:
            return {
                "style": "negative",
                "label": f"Statistically confirmed underperformance historically ({comparison}, n={n})",
            }
        return {
            "style": "neutral",
            "label": f"Low separation -- not statistically distinguishable from this horizon's "
            f"{base_rate:.1%} base rate ({comparison}, n={n})",
        }
