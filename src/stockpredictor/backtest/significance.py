"""Statistical significance and robustness checks on the backtest's
Information Coefficient (§25, §30: "a too-good backtest result should be
treated as a leakage bug, not a win" — the flip side of that discipline is
that a *modest* result also needs to be checked, not just accepted at face
value because the sign happens to be positive).

`backtest/metrics.py`'s `information_coefficient` gives one IC value per
rebalance date; the backtest result's `mean_ic` (persisted by
backtest/registry.py) is just the average of that series. A positive mean
of a handful of numbers proves nothing on its own -- it could easily be
noise. This module answers three separate honest questions about that
series, not just one:

1. **Is the mean distinguishable from zero at all**, given how few
   independent periods we actually have (`ic_ttest`)? With ~195 periods
   (a realistic number this early), a small positive mean can still be
   statistically indistinguishable from luck.
2. **Does that conclusion survive without the t-test's normality
   assumption** (`ic_bootstrap_ci`) -- financial signal distributions are
   rarely textbook-normal, so a purely parametric answer alone would be
   overconfident.
3. **Is the edge coming from the whole period, or one lucky stretch**
   (`ic_subperiod_stability`)? A mean IC of +0.02 built from +0.08 in the
   first half and -0.04 in the second half is a much weaker result than a
   steady +0.02 throughout, even though the headline number is identical.

`ic_autocorrelation` exists to sanity-check the independence assumption
the t-test leans on: `select_rebalance_dates` (backtest/engine.py) already
makes the underlying return *windows* non-overlapping, but the market
regime driving IC from one period to the next can still be autocorrelated
-- a high lag-1 autocorrelation means the "195 periods" are worth fewer
than 195 independent observations, and the t-test's p-value should be read
as optimistic, not corrected for it here (correcting properly needs a
longer, more elaborate model than this system's data currently justifies
building -- see the module docstring philosophy elsewhere in this
codebase: earn complexity, don't pre-build it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_BOOTSTRAP_SAMPLES = 5000
DEFAULT_CONFIDENCE = 0.95


def ic_ttest(ic_series: pd.Series) -> dict:
    """One-sample t-test of H0: mean IC == 0, against the two-sided
    alternative. Returns NaN/None fields (not a raised error) when there
    are too few periods to say anything -- fewer than 2 observations can't
    produce a standard deviation, and the whole point of this function is
    to be honest about insufficient evidence, not to force an answer."""
    values = ic_series.dropna().to_numpy()
    n = len(values)
    if n < 2:
        return {
            "n_periods": n,
            "mean_ic": float(values[0]) if n == 1 else float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "significant_at_5pct": False,
        }

    mean = float(np.mean(values))
    result = stats.ttest_1samp(values, popmean=0.0)
    t_stat, p_value = float(result.statistic), float(result.pvalue)

    sem = stats.sem(values)
    margin = sem * stats.t.ppf((1 + DEFAULT_CONFIDENCE) / 2, df=n - 1)

    return {
        "n_periods": n,
        "mean_ic": mean,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "significant_at_5pct": bool(p_value < 0.05),
    }


def ic_autocorrelation(ic_series: pd.Series, lag: int = 1) -> float:
    """Lag-`lag` autocorrelation of the per-date IC series -- see module
    docstring on why this matters for reading the t-test's p-value
    honestly. NaN (not 0.0) when there isn't enough history to compute it,
    since "no correlation" and "couldn't measure it" are different facts."""
    values = ic_series.dropna()
    if len(values) < lag + 2:
        return float("nan")
    return float(values.autocorr(lag=lag))


def ic_bootstrap_ci(
    ic_series: pd.Series,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | None = 0,
) -> dict:
    """Percentile-bootstrap confidence interval on the mean IC -- a
    distribution-free complement to `ic_ttest`'s normal-theory interval.
    Also reports the fraction of bootstrap resamples with a mean IC <= 0,
    a bootstrap-native analogue of a one-sided p-value that doesn't lean on
    the t-distribution at all.

    Deliberately a fixed default seed (reproducible reports -- the same
    persisted IC series should reproduce the same CI when re-run, matching
    this project's reproducibility discipline elsewhere), overridable by
    the caller if they explicitly want resampling variance across runs."""
    values = ic_series.dropna().to_numpy()
    n = len(values)
    if n < 2:
        return {
            "n_periods": n,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "fraction_non_positive": float("nan"),
        }

    rng = np.random.default_rng(seed)
    resample_means = rng.choice(values, size=(n_bootstrap, n), replace=True).mean(axis=1)

    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "n_periods": n,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "fraction_non_positive": float((resample_means <= 0).mean()),
    }


def ic_subperiod_stability(ic_series: pd.Series, n_splits: int = 2) -> pd.DataFrame:
    """Splits the date-sorted IC series into `n_splits` contiguous, roughly
    equal-sized chunks and reports each chunk's mean/std/n -- is the edge
    coming from the whole period, or one lucky stretch? An empty frame
    (not an error) when there isn't enough history to split meaningfully."""
    values = ic_series.dropna().sort_index()
    if len(values) < n_splits * 2:
        return pd.DataFrame(columns=["period", "start", "end", "mean_ic", "std_ic", "n_periods"])

    # np.array_split on a Series returns plain ndarrays, losing the date
    # index each chunk's "start"/"end" needs -- split by position instead
    # and re-slice the Series so each chunk keeps its own index.
    split_points = np.array_split(np.arange(len(values)), n_splits)
    chunks = [values.iloc[idx] for idx in split_points]
    rows = []
    for i, chunk in enumerate(chunks):
        rows.append(
            {
                "period": i + 1,
                "start": chunk.index[0],
                "end": chunk.index[-1],
                "mean_ic": float(chunk.mean()),
                "std_ic": float(chunk.std(ddof=1)) if len(chunk) > 1 else float("nan"),
                "n_periods": len(chunk),
            }
        )
    return pd.DataFrame(rows)


def run_significance_report(ic_series: pd.Series) -> dict:
    """Bundles all of the above into one report -- the single entry point
    scripts/the UI actually call. `consistent_sign` is a plain-language
    summary of the sub-period check: True only if every sub-period's mean
    IC has the same sign as the overall mean (excluding sub-periods with
    too few points to have a meaningful sign, which don't count against
    consistency but also don't count for it)."""
    ttest = ic_ttest(ic_series)
    bootstrap = ic_bootstrap_ci(ic_series)
    subperiods = ic_subperiod_stability(ic_series)
    autocorr = ic_autocorrelation(ic_series)

    if subperiods.empty or pd.isna(ttest["mean_ic"]):
        consistent_sign = None
    else:
        overall_sign = np.sign(ttest["mean_ic"])
        consistent_sign = bool((np.sign(subperiods["mean_ic"]) == overall_sign).all())

    return {
        "ttest": ttest,
        "bootstrap": bootstrap,
        "subperiods": subperiods,
        "lag1_autocorrelation": autocorr,
        "consistent_sign_across_subperiods": consistent_sign,
    }
