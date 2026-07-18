"""Lightweight feature-drift baseline (§23: "feature drift, concept drift").

Not a full Evidently/statistical-test integration -- architecture doc Truth
3 says a heavier dependency must earn its place before it's added, and a
basic mean/std shift check against a stored baseline is sufficient to catch
the two failure modes that actually matter early: a source silently
changing its data (e.g. re-scaled units) or a bug shifting an entire
feature's distribution. Promote to a fuller drift-detection library only if
this proves insufficient in practice.
"""

from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.storage.lake import Lake

DRIFT_DOMAIN = "drift_baseline"
Z_SCORE_ALERT_THRESHOLD = 3.0


def compute_feature_stats(features_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Mean/std per feature across the given frame (typically the latest
    date's cross-section)."""
    stats = features_df[feature_cols].agg(["mean", "std"]).T
    stats.index.name = "feature"
    return stats.reset_index()


def load_baseline(lake: Lake) -> pd.DataFrame:
    return lake.read(DataLayer.GOLD, DRIFT_DOMAIN, "baseline")


def save_baseline(lake: Lake, stats: pd.DataFrame) -> int:
    return lake.write(stats, DataLayer.GOLD, DRIFT_DOMAIN, "baseline", key_cols=["feature"])


def check_drift(
    current_stats: pd.DataFrame,
    baseline_stats: pd.DataFrame,
    z_threshold: float = Z_SCORE_ALERT_THRESHOLD,
) -> pd.DataFrame:
    """For each feature present in both, compute how many baseline standard
    deviations the current mean has moved. Features missing from the
    baseline (e.g. newly added) are reported but not flagged -- there's
    nothing to compare against yet."""
    merged = current_stats.merge(baseline_stats, on="feature", how="left", suffixes=("_current", "_baseline"))
    baseline_std_safe = merged["std_baseline"].replace(0, float("nan"))
    merged["z_shift"] = (merged["mean_current"] - merged["mean_baseline"]).abs() / baseline_std_safe
    merged["drifted"] = merged["z_shift"].fillna(0) > z_threshold
    return merged


def check_and_update_baseline(
    lake: Lake,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    z_threshold: float = Z_SCORE_ALERT_THRESHOLD,
) -> pd.DataFrame | None:
    """Convenience wrapper for orchestration: compute current stats, compare
    against any existing baseline (returns None if no baseline exists yet --
    the first run always just establishes one), and refresh the stored
    baseline to the current stats either way. A rolling baseline, not a
    frozen reference -- appropriate for markets, which genuinely do shift
    regime over time (§24: "drift-triggered retraining")."""
    current_stats = compute_feature_stats(features_df, feature_cols)
    baseline = load_baseline(lake)

    result = check_drift(current_stats, baseline, z_threshold) if not baseline.empty else None
    save_baseline(lake, current_stats)
    return result
