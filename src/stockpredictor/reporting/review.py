"""Monthly ML Review Board report (spec Parts 4-7): a fully data-driven
template, not an LLM-authored narrative -- decided with the user (see the
plan): every table/number below is computed from real
`published_predictions` / `validation_results` / `explanations` history;
anything that calls for qualitative judgment is left as an explicit
`<!-- ANALYSIS: fill in during monthly review -->` placeholder rather than
fabricated prose. Recommendations (Part 5) and the model-change proposal
(Part 6) are rule-based off the same computed evidence, not free text.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.backtest.metrics import information_coefficient
from stockpredictor.backtest.registry import read_latest_backtest_result
from stockpredictor.common.config import REPO_ROOT
from stockpredictor.common.logging import get_logger
from stockpredictor.explain.registry import read_explanations
from stockpredictor.monitoring.accuracy import compute_accuracy
from stockpredictor.reporting.analytics import load_resolved_predictions
from stockpredictor.storage.lake import Lake
from stockpredictor.storage.models import Security

logger = get_logger(__name__)

REPORTS_DIR = REPO_ROOT / "reports"

# Must match scripts/run_backtests.py's STRATEGY_ID. Not cross-imported --
# scripts/ depends on the package, not the reverse -- so this is kept in
# sync by hand; a mismatch just means the backtest-comparison section
# degrades to "no committed backtest found," not a crash.
_BACKTEST_STRATEGY_ID = "top_k_technical_fundamental_v1"
_BACKTEST_HORIZON = "90d"

_LOW_IMPORTANCE_THRESHOLD = 0.003  # 0.3%, matching the spec's own worked example
_STANDING_FEATURE_IDEAS = [
    ("Options Open Interest", "Derivatives positioning often leads spot price moves."),
    ("FII/DII activity", "Institutional flow is a well-documented Indian-market factor."),
    ("Quarterly earnings surprise", "Post-earnings drift is a documented anomaly."),
    ("Mutual fund holdings", "Ownership concentration/changes can precede re-ratings."),
    ("Insider trading", "SAST disclosures are a public, point-in-time-safe signal source."),
    ("Analyst upgrades/downgrades", "Sell-side revisions are a fast-moving sentiment proxy."),
    ("Alternative news providers", "Broadens beyond Google News RSS's no-historical-backfill limit."),
    ("Macroeconomic indicators", "Rate/inflation/currency regime context, currently unused."),
]


# ---------------------------------------------------------------- data ----


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01")
    return start, start + pd.offsets.MonthEnd(1)


def _with_sector(session: Session, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        frame["sector"] = pd.Series(dtype="object")
        return frame
    symbols = frame["stock_symbol"].unique().tolist()
    rows = session.execute(select(Security.symbol, Security.sector).where(Security.symbol.in_(symbols))).all()
    sector_map = {s: (sec or "Unknown") for s, sec in rows}
    frame["sector"] = frame["stock_symbol"].map(sector_map).fillna("Unknown")
    return frame


def _sector_performance(resolved: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    if resolved.empty:
        # Columns must match the populated case -- callers (both here and in
        # _build_proposal_markdown) index into this by column name (e.g.
        # `sector_perf["avg_alpha"]`) unconditionally, not just after a
        # `.empty` check, so a bare `pd.DataFrame()` would KeyError there.
        return pd.DataFrame(columns=["n", "hit_rate", "avg_alpha"])
    grouped = resolved.groupby("sector").agg(
        n=("alpha", "size"), hit_rate=("hit_or_miss", "mean"), avg_alpha=("alpha", "mean")
    )
    return grouped[grouped["n"] >= min_n].sort_values("avg_alpha", ascending=False)


def _feature_contributions(lake: Lake, resolved: pd.DataFrame) -> pd.DataFrame:
    """Mean SHAP contribution per feature across `resolved`, joined against
    nightly's already-committed per-horizon explanations
    (explain/registry.py) on (symbol, date, horizon) -- no SHAP is
    recomputed here."""
    empty = pd.DataFrame(columns=["mean_contribution", "mean_abs_contribution", "n"])
    if resolved.empty:
        return empty
    records = []
    for horizon, group in resolved.groupby("prediction_horizon"):
        explanations = read_explanations(lake, horizon)
        if explanations.empty:
            continue
        explanations = explanations.copy()
        explanations["date"] = pd.to_datetime(explanations["date"]).dt.normalize()
        merged = group.merge(
            explanations, left_on=["stock_symbol", "prediction_date"], right_on=["symbol", "date"], how="inner"
        )
        for _, row in merged.iterrows():
            for signal in list(row["top_positive_signals"]) + list(row["top_negative_signals"]):
                records.append({"feature": signal["feature"], "contribution": signal["contribution"]})
    if not records:
        return empty
    df = pd.DataFrame(records)
    df["abs_contribution"] = df["contribution"].abs()
    result = df.groupby("feature").agg(
        mean_contribution=("contribution", "mean"),
        mean_abs_contribution=("abs_contribution", "mean"),
        n=("contribution", "size"),
    )
    return result.sort_values("mean_abs_contribution", ascending=False)


def _rolling_ic(resolved: pd.DataFrame, months: int = 6) -> list[tuple[str, float]]:
    if resolved.empty:
        return []
    tmp = resolved.copy()
    tmp["month"] = tmp["prediction_date"].dt.to_period("M")
    out = []
    for period, group in tmp.groupby("month"):
        if len(group) < 3:
            continue
        ic = information_coefficient(group["prediction_probability"], group["actual_return"])
        out.append((str(period), ic))
    return sorted(out)[-months:]


def _repeat_offenders(resolved: pd.DataFrame, min_misses: int = 2) -> pd.DataFrame:
    if resolved.empty:
        return pd.DataFrame()
    misses = resolved[~resolved["hit_or_miss"]]
    counts = misses.groupby("stock_symbol").size().rename("miss_count").sort_values(ascending=False)
    return counts[counts >= min_misses].to_frame()


# ------------------------------------------------------------ rendering ----

_ANALYSIS_PLACEHOLDER = "<!-- ANALYSIS: fill in during monthly review -->"


def _md_table(headers: list[str], rows: list[list], empty_note: str = "No data available yet.") -> str:
    if not rows:
        return f"_{empty_note}_\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value * 100:.2f}%"


def _num(value: float | None, decimals: int = 4) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{decimals}f}"


def _recommendation(
    priority: str, impact: str, complexity: str, confidence: str, title: str, reason: str, evidence: str
) -> str:
    return (
        f"### [{priority}] {title}\n\n"
        f"- **Expected Impact:** {impact}\n"
        f"- **Implementation Complexity:** {complexity}\n"
        f"- **Confidence:** {confidence}\n"
        f"- **Reasoning:** {reason}\n"
        f"- **Supporting Evidence:** {evidence}\n"
    )


def _build_review_markdown(
    month: str,
    month_resolved: pd.DataFrame,
    history_resolved: pd.DataFrame,
    feature_contrib: pd.DataFrame,
    sector_perf: pd.DataFrame,
    calibration: dict | None,
    ic_trend: list[tuple[str, float]],
    repeat_offenders: pd.DataFrame,
    backtest_row: dict | None,
) -> str:
    wins = month_resolved[month_resolved["hit_or_miss"]].sort_values("alpha", ascending=False)
    losses = month_resolved[~month_resolved["hit_or_miss"]].sort_values("alpha")

    positive_features = feature_contrib[feature_contrib["mean_contribution"] > 0].head(10)
    low_importance = feature_contrib[feature_contrib["mean_abs_contribution"] < _LOW_IMPORTANCE_THRESHOLD]

    top_sectors = sector_perf.head(5)
    bottom_sectors = sector_perf.tail(5).iloc[::-1]

    decile = calibration["hit_rate_by_score_decile"] if calibration else {}
    ic_declining = len(ic_trend) >= 3 and ic_trend[-1][1] < ic_trend[0][1]

    sections = [
        f"# ML Review -- {month}\n",
        f"Generated from `published_predictions` and `validation_results` history. "
        f"{len(month_resolved)} predictions resolved this month; {len(history_resolved)} resolved all-time.\n",
        "## 1. Which predictions succeeded?\n",
        _md_table(
            ["Symbol", "Horizon", "Rank", "Alpha", "Actual return"],
            [[r.stock_symbol, r.prediction_horizon, r.rank, _pct(r.alpha), _pct(r.actual_return)] for r in wins.itertuples()],
        ),
        "## 2. Which predictions failed?\n",
        _md_table(
            ["Symbol", "Horizon", "Rank", "Alpha", "Actual return"],
            [[r.stock_symbol, r.prediction_horizon, r.rank, _pct(r.alpha), _pct(r.actual_return)] for r in losses.itertuples()],
        ),
        "## 3. Which sectors consistently outperform?\n",
        "(All-time history, sectors with >= 3 resolved predictions.)\n\n"
        + _md_table(
            ["Sector", "N", "Hit rate", "Avg alpha"],
            [[sec, int(row["n"]), _pct(row["hit_rate"]), _pct(row["avg_alpha"])] for sec, row in top_sectors.iterrows()],
        ),
        "## 4. Which sectors consistently underperform?\n",
        _md_table(
            ["Sector", "N", "Hit rate", "Avg alpha"],
            [[sec, int(row["n"]), _pct(row["hit_rate"]), _pct(row["avg_alpha"])] for sec, row in bottom_sectors.iterrows()],
        ),
        "## 5. Which features contributed positively?\n",
        "(Mean SHAP contribution across this month's resolved predictions' Top-N explanations.)\n\n"
        + _md_table(
            ["Feature", "Mean contribution", "N observations"],
            [[f, _num(row["mean_contribution"]), int(row["n"])] for f, row in positive_features.iterrows()],
        ),
        "## 6. Which features had little or no predictive power?\n",
        f"(Mean |SHAP contribution| below {_LOW_IMPORTANCE_THRESHOLD:.1%}.)\n\n"
        + _md_table(
            ["Feature", "Mean |contribution|", "N observations"],
            [[f, _num(row["mean_abs_contribution"]), int(row["n"])] for f, row in low_importance.iterrows()],
        ),
        "## 7. Did news sentiment improve predictions?\n",
        "Sentiment features (`features/sentiment.py`) are computed and stored but are **not yet part of "
        "`ALL_FEATURE_COLUMNS`** -- the news connector has no historical backfill, so real day-by-day "
        "coverage only started accumulating from nightly ingestion (see the project's standing "
        "sentiment-readiness follow-up). This question cannot be honestly answered from live model "
        "behavior until sentiment is actually folded into the feature set and evaluated out-of-sample.\n",
        "## 8. Did technical indicators improve predictions?\n",
        "(Mean SHAP contribution by factor block -- see explain/factors.py -- for this month's resolved "
        "predictions; a positive mean contribution suggests the block helped rather than hurt.)\n\n"
        + f"{_ANALYSIS_PLACEHOLDER}\n",
        "## 9. Were false positives concentrated in specific industries?\n",
        "(Sector distribution of this month's misses.)\n\n"
        + _md_table(
            ["Sector", "Misses"],
            [[sec, int(n)] for sec, n in losses["sector"].value_counts().items()] if "sector" in losses.columns else [],
        ),
        "## 10. Were false negatives concentrated in specific industries?\n",
        "This system only publishes a Top-N recommendation list -- it never records an explicit \"avoid this "
        "stock\" prediction, so there is no natural false-negative population to measure here. If this "
        "question matters going forward, consider also tracking a Bottom-N list (see Part 6, Risk Metrics "
        "to Add).\n",
        "## 11. Is the model becoming stale?\n",
        "(Information Coefficient -- prediction probability vs. realized return -- by month, last "
        f"{len(ic_trend)} months with >=3 resolved predictions.)\n\n"
        + _md_table(["Month", "IC"], [[m, _num(ic)] for m, ic in ic_trend])
        + ("\n**Flag: trailing IC is lower than the earliest month in this window.**\n" if ic_declining else "\n"),
        "## 12. Is there evidence of overfitting?\n",
        (
            f"Latest committed walk-forward backtest (`{_BACKTEST_STRATEGY_ID}`, {_BACKTEST_HORIZON}): "
            f"mean IC = {_num(backtest_row.get('mean_ic'))}, strategy Sharpe = "
            f"{_num(backtest_row.get('strategy_sharpe'))}. Compare against this month's live figures "
            f"above -- a live IC persistently far above the backtested mean IC across many months would be "
            f"the overfitting red flag; a single month is not enough evidence either way.\n"
            if backtest_row
            else "No committed backtest result found (`scripts/run_backtests.py` output) to compare against.\n"
        ),
        "## 13. Is prediction confidence well calibrated?\n",
        "(Hit rate by predicted-probability decile, all-time resolved history -- monitoring/accuracy.py.)\n\n"
        + _md_table(["Decile", "Hit rate"], [[d, _pct(v)] for d, v in sorted(decile.items())]),
        "## 14. Which stocks repeatedly fooled the model?\n",
        "(All-time, symbols with >= 2 misses.)\n\n"
        + _md_table(
            ["Symbol", "Miss count"],
            [[sym, int(row["miss_count"])] for sym, row in repeat_offenders.iterrows()],
        ),
        "## 15. Which indicators appear redundant?\n",
        "Candidates are the same low-importance features from Question 6, cross-checked against each other "
        "for near-duplicate signal (e.g. two moving-average variants moving together) -- that cross-check is "
        f"a judgment call, not computable from SHAP alone.\n\n{_ANALYSIS_PLACEHOLDER}\n",
        "## 16. Which new features could improve prediction quality?\n",
        _md_table(["Candidate", "Rationale"], [[name, why] for name, why in _STANDING_FEATURE_IDEAS]),
    ]
    return "\n".join(sections)


def _build_proposal_markdown(
    month: str,
    feature_contrib: pd.DataFrame,
    sector_perf: pd.DataFrame,
    ic_declining: bool,
    calibration: dict | None,
) -> str:
    low_importance = feature_contrib[feature_contrib["mean_abs_contribution"] < _LOW_IMPORTANCE_THRESHOLD]
    worst_sectors = sector_perf[sector_perf["avg_alpha"] < 0]

    recs = []
    if ic_declining:
        recs.append(
            _recommendation(
                "HIGH", "High", "Medium", "Medium",
                "Investigate model staleness / evaluate a retrain",
                "The trailing months' Information Coefficient is lower than the start of the rolling window.",
                "See ML-Review.md Question 11's IC-by-month table.",
            )
        )
    for feature, row in low_importance.iterrows():
        recs.append(
            _recommendation(
                "LOW", "Low", "Low", "Medium",
                f"Consider removing indicator `{feature}`",
                f"Mean |SHAP contribution| ({row['mean_abs_contribution']:.4f}) is below the "
                f"{_LOW_IMPORTANCE_THRESHOLD:.1%} threshold across {int(row['n'])} observations.",
                "See ML-Review.md Question 6.",
            )
        )
    for sector, row in worst_sectors.iterrows():
        recs.append(
            _recommendation(
                "MEDIUM", "Medium", "Low", "Low",
                f"Review model behavior in the {sector} sector",
                f"Average alpha across {int(row['n'])} resolved predictions in this sector is negative "
                f"({row['avg_alpha']:.2%}).",
                "See ML-Review.md Question 4.",
            )
        )
    if calibration:
        deciles = calibration["hit_rate_by_score_decile"]
        if deciles and max(deciles.values()) - min(deciles.values()) < 0.10:
            recs.append(
                _recommendation(
                    "MEDIUM", "Medium", "Medium", "Medium",
                    "Investigate weak probability calibration",
                    "Hit rate barely differs between the top and bottom score deciles -- the model may not "
                    "be adding much ranking value yet.",
                    f"Decile hit rates: {deciles}.",
                )
            )
    if not recs:
        recs.append(
            _recommendation(
                "LOW", "Low", "n/a", "Low",
                "No rule-triggered recommendations this month",
                "None of the standing thresholds (feature importance, sector alpha, IC trend, calibration "
                "spread) were tripped by this month's data.",
                "See ML-Review.md for the underlying tables.",
            )
        )

    sections = [
        f"# Improvement Proposal -- {month}\n",
        "## Recommendations\n",
        "\n---\n\n".join(recs),
        "\n## Model Change Proposal\n",
        "### Features to Add\n",
        _md_table(["Candidate", "Rationale"], [[name, why] for name, why in _STANDING_FEATURE_IDEAS]),
        "### Features to Remove\n",
        _md_table(
            ["Feature", "Mean |contribution|"],
            [[f, f"{row['mean_abs_contribution']:.4f}"] for f, row in low_importance.iterrows()],
        ),
        "### Hyperparameters to Tune\n",
        f"{_ANALYSIS_PLACEHOLDER} -- see `config/model.yaml` for the current base-learner/calibration "
        "configuration; no automated hyperparameter search exists yet.\n",
        "### Models Worth Testing\n",
        "- Gradient-boosted alternatives to LightGBM (XGBoost, CatBoost) as an additional base learner.\n"
        f"{_ANALYSIS_PLACEHOLDER}\n",
        "### Feature Engineering Improvements\n",
        f"{_ANALYSIS_PLACEHOLDER}\n",
        "### Data Quality Improvements\n",
        "- Sentiment history depth -- see the project's standing sentiment-readiness follow-up.\n"
        "- Fundamentals remain sparse/laggy by nature of the free data source (see features/fundamental.py).\n",
        "### Potential Bugs Detected\n",
        f"None newly detected by this report. Check `run_metadata` (orchestration/run_tracking.py) for any "
        f"failed pipeline stages this month.\n",
        "### Ranking Improvements\n",
        f"{_ANALYSIS_PLACEHOLDER}\n",
        "### Probability Calibration Improvements\n",
        f"{_ANALYSIS_PLACEHOLDER} -- see ML-Review.md Question 13's decile table.\n",
        "### Risk Metrics to Add\n",
        "- Sector/industry concentration limits across the published Top-N.\n"
        "- A Bottom-N \"avoid\" list, to make false-negative analysis (Question 10) possible.\n",
        "### Expected Benefit / Confidence Score\n",
        f"{_ANALYSIS_PLACEHOLDER}\n",
    ]
    return "\n".join(sections)


def generate_monthly_review(
    lake: Lake, session_factory: sessionmaker[Session], month: str | None = None
) -> tuple[str, str]:
    """Generate the ML Review (`reports/{month}-ML-Review.md`) and
    Improvement Proposal (`reports/{month}-Improvement-Proposal.md`) for
    `month` (`YYYY-MM`, defaults to the current calendar month). Refuses to
    overwrite an existing month's report pair. Returns (review_path,
    proposal_path) as strings."""
    month = month or dt.date.today().strftime("%Y-%m")
    review_path = REPORTS_DIR / f"{month}-ML-Review.md"
    proposal_path = REPORTS_DIR / f"{month}-Improvement-Proposal.md"
    if review_path.exists() or proposal_path.exists():
        raise FileExistsError(f"reports/{month}-*.md already exist -- refusing to overwrite a previous review.")

    start, end = _month_bounds(month)
    session = session_factory()
    try:
        history_resolved = load_resolved_predictions(session)
        history_resolved = _with_sector(session, history_resolved)
    finally:
        session.close()

    month_resolved = history_resolved[
        (history_resolved["prediction_date"] >= start) & (history_resolved["prediction_date"] <= end)
    ]

    feature_contrib = _feature_contributions(lake, month_resolved if not month_resolved.empty else history_resolved)
    sector_perf = _sector_performance(history_resolved)
    ic_trend = _rolling_ic(history_resolved)
    ic_declining = len(ic_trend) >= 3 and ic_trend[-1][1] < ic_trend[0][1]
    repeat_offenders = _repeat_offenders(history_resolved)

    calibration = None
    for horizon in history_resolved["prediction_horizon"].unique() if not history_resolved.empty else []:
        result = compute_accuracy(lake, horizon)
        if result:
            calibration = result
            break

    backtest_row = read_latest_backtest_result(lake, _BACKTEST_STRATEGY_ID, _BACKTEST_HORIZON)

    review_md = _build_review_markdown(
        month, month_resolved, history_resolved, feature_contrib, sector_perf,
        calibration, ic_trend, repeat_offenders, backtest_row,
    )
    proposal_md = _build_proposal_markdown(month, feature_contrib, sector_perf, ic_declining, calibration)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    review_path.write_text(review_md, encoding="utf-8")
    proposal_path.write_text(proposal_md, encoding="utf-8")
    logger.info("Wrote %s and %s", review_path, proposal_path)
    return str(review_path), str(proposal_path)
