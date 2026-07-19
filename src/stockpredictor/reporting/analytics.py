"""Performance analytics (ML Review Board spec Part 3): historical stats
computed from resolved `ValidationResult`s, feeding both the HTML dashboard
(reporting/dashboard.py) and the monthly review (reporting/review.py).

Every stat here is grouped by `prediction_date` (the "vintage" the
recommendation was issued in), not `validated_at` (when it happened to be
checked) -- evaluating "how good were the picks made in March" is the
meaningful question; "how many validations ran in March" isn't.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from stockpredictor.common.logging import get_logger
from stockpredictor.monitoring.drift import load_baseline
from stockpredictor.storage.lake import Lake
from stockpredictor.storage.models import PublishedPrediction, ValidationResult

logger = get_logger(__name__)

_HORIZON_TRADING_DAYS = {"5d": 5, "30d": 30, "90d": 90}


def _empty_frame(dtypes: dict[str, str]) -> pd.DataFrame:
    """An empty DataFrame with real per-column dtypes, not `pd.DataFrame([])`'s
    all-`object` columns. That distinction matters here: pandas'
    `df[some_object_dtype_series]` is ambiguous and gets treated as *column*
    selection rather than boolean row-filtering when the series isn't
    actually `bool`-dtype -- an empty `hit_or_miss` column built without an
    explicit `bool` dtype silently collapses `df[df["hit_or_miss"]]` to zero
    *columns*, not zero rows, which then raises a confusing KeyError several
    lines later at whatever column is referenced next."""
    return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})


_PUBLISHED_DTYPES = {
    "prediction_id": "object", "prediction_date": "datetime64[ns]", "prediction_horizon": "object",
    "stock_symbol": "object", "rank": "int64", "prediction_probability": "float64",
}


def _load_published(session: Session) -> pd.DataFrame:
    rows = session.execute(select(PublishedPrediction)).scalars().all()
    if not rows:
        return _empty_frame(_PUBLISHED_DTYPES)
    return pd.DataFrame(
        [
            {
                "prediction_id": r.prediction_id,
                "prediction_date": pd.Timestamp(r.prediction_date),
                "prediction_horizon": r.prediction_horizon,
                "stock_symbol": r.stock_symbol,
                "rank": r.rank,
                "prediction_probability": float(r.prediction_probability),
            }
            for r in rows
        ]
    )


_RESOLVED_DTYPES = {
    "prediction_id": "object", "prediction_date": "datetime64[ns]", "prediction_horizon": "object",
    "stock_symbol": "object", "rank": "int64", "prediction_probability": "float64",
    "actual_return": "float64", "benchmark_return": "float64", "alpha": "float64", "hit_or_miss": "bool",
}


def load_resolved_predictions(session: Session) -> pd.DataFrame:
    """`PublishedPrediction` joined to its `ValidationResult`, one row per
    resolved prediction -- shared by reporting/review.py as well as this
    module. Empty result still carries the full column schema *and real
    dtypes* (see `_empty_frame`) so callers can safely index by column name
    -- e.g. `frame["prediction_date"]` or boolean-mask on `hit_or_miss` --
    even when nothing has resolved yet, which is the real state of a
    freshly-deployed system."""
    stmt = select(PublishedPrediction, ValidationResult).join(
        ValidationResult, ValidationResult.prediction_id == PublishedPrediction.prediction_id
    )
    matches = session.execute(stmt).all()
    if not matches:
        return _empty_frame(_RESOLVED_DTYPES)
    records = []
    for pred, val in matches:
        records.append(
            {
                "prediction_id": pred.prediction_id,
                "prediction_date": pd.Timestamp(pred.prediction_date),
                "prediction_horizon": pred.prediction_horizon,
                "stock_symbol": pred.stock_symbol,
                "rank": pred.rank,
                "prediction_probability": float(pred.prediction_probability),
                "actual_return": float(val.actual_return),
                "benchmark_return": float(val.benchmark_return),
                "alpha": float(val.alpha),
                "hit_or_miss": bool(val.hit_or_miss),
            }
        )
    return pd.DataFrame(records)


def _accuracy(resolved: pd.DataFrame, max_rank: int | None = None) -> float | None:
    subset = resolved if max_rank is None else resolved[resolved["rank"] <= max_rank]
    if subset.empty:
        return None
    return float(subset["hit_or_miss"].mean())


def _period_stats(resolved: pd.DataFrame, freq: str) -> dict:
    if resolved.empty:
        return {}
    grouped = resolved.set_index("prediction_date").groupby(pd.Grouper(freq=freq))
    out = {}
    for period, group in grouped:
        if group.empty:
            continue
        # Label from the Timestamp directly rather than Timestamp.to_period(freq):
        # pandas 2.2+ Grouper/offset aliases ("ME"/"QE") aren't valid Period
        # frequency strings ("M"/"Q"), so `.to_period(freq)` raises here.
        label = period.strftime("%Y-%m") if freq == "ME" else f"{period.year}Q{period.quarter}"
        out[label] = {
            "n": int(len(group)),
            "hit_rate": float(group["hit_or_miss"].mean()),
            "avg_alpha": float(group["alpha"].mean()),
            "avg_return": float(group["actual_return"].mean()),
        }
    return out


def _rolling_window(resolved: pd.DataFrame, months: int) -> dict | None:
    if resolved.empty:
        return None
    latest = resolved["prediction_date"].max()
    cutoff = latest - pd.DateOffset(months=months)
    window = resolved[resolved["prediction_date"] > cutoff]
    if window.empty:
        return None
    return {
        "n": int(len(window)),
        "hit_rate": float(window["hit_or_miss"].mean()),
        "avg_alpha": float(window["alpha"].mean()),
        "avg_return": float(window["actual_return"].mean()),
    }


def _probability_histogram(published: pd.DataFrame) -> dict:
    """10-bin histogram of published `prediction_probability`, with plain
    string bin labels (an Interval-keyed dict isn't JSON-serializable, which
    reporting/dashboard.py and reporting/review.py both need).

    Decimal precision is derived from the bin width rather than fixed at 3
    places: real calibrated scores can cluster in a narrow band (see
    models/calibration.py's centered-isotonic interpolation -- confirmed
    live, not hypothetical, dry-running this against real data), and a
    fixed-3-decimal label collapses distinct bins into identical-looking
    strings like "0.525-0.525"."""
    if published.empty:
        return {}
    counts = published["prediction_probability"].value_counts(bins=10, sort=False)
    bin_width = counts.index[0].length if len(counts) else 0.0
    decimals = max(3, -math.floor(math.log10(bin_width)) + 1) if bin_width > 0 else 3
    return {
        f"{interval.left:.{decimals}f}-{interval.right:.{decimals}f}": int(count) for interval, count in counts.items()
    }


def compute_performance_analytics(lake: Lake, session_factory: sessionmaker[Session]) -> dict:
    """Every number a human (or reporting/review.py) needs to judge model
    performance so far, computed fresh from `published_predictions` +
    `validation_results` -- no cached/stale summary tables."""
    session = session_factory()
    try:
        published = _load_published(session)
        resolved = load_resolved_predictions(session)
    finally:
        session.close()

    if resolved.empty:
        return {
            "n_published": int(len(published)),
            "n_resolved": 0,
            "note": "No predictions have resolved (completed their horizon) yet.",
        }

    # Ratio metrics reuse backtest/metrics.py's per-rebalance-return-series
    # functions on the *actual* resolved-return sequence, one horizon at a
    # time -- exactly the "per-rebalance return series" they were designed
    # for (annualization needs a single horizon_days, so horizons that mix
    # different lengths are reported separately, not pooled).
    by_horizon_ratios = {}
    for horizon, group in resolved.groupby("prediction_horizon"):
        h_days = _HORIZON_TRADING_DAYS.get(horizon)
        returns = group.sort_values("prediction_date")["actual_return"]
        if h_days is None or len(returns) < 2:
            continue
        by_horizon_ratios[horizon] = {
            "n": int(len(returns)),
            "cagr": cagr(returns, h_days),
            "sharpe_ratio": sharpe_ratio(returns, h_days),
            "sortino_ratio": sortino_ratio(returns, h_days),
            "max_drawdown": max_drawdown(returns),
        }

    baseline = load_baseline(lake)
    feature_drift = (
        {"as_of": "most recent nightly run", "features": baseline.set_index("feature").to_dict("index")}
        if not baseline.empty
        else None
    )

    return {
        "n_published": int(len(published)),
        "n_resolved": int(len(resolved)),
        "overall_accuracy": _accuracy(resolved),
        "top_5_accuracy": _accuracy(resolved, max_rank=5),
        "top_10_accuracy": _accuracy(resolved, max_rank=10),
        "avg_return": float(resolved["actual_return"].mean()),
        "median_return": float(resolved["actual_return"].median()),
        "avg_alpha": float(resolved["alpha"].mean()),
        "win_rate": win_rate(resolved["actual_return"]),
        "loss_rate": 1.0 - win_rate(resolved["actual_return"]),
        "avg_holding_return": float(resolved["actual_return"].mean()),
        "by_horizon_ratios": by_horizon_ratios,
        "monthly_stats": _period_stats(resolved, "ME"),
        "quarterly_stats": _period_stats(resolved, "QE"),
        "rolling_6m": _rolling_window(resolved, 6),
        "rolling_12m": _rolling_window(resolved, 12),
        "probability_distribution": _probability_histogram(published),
        "feature_drift": feature_drift,
    }
