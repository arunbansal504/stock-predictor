"""MVP dashboard (§15 "MVP shortcut": Streamlit stands in for Top Picks,
Stock Detail, Backtest Lab, and Model Transparency until the Next.js UI
lands in Phase 2).

Reads directly from the lake -- the same pre-computed Gold-layer artifacts
the API serves (§4: "the UI never triggers model inference on demand").
Running this only requires the nightly pipeline (or scripts/run_phase1_smoke.py
for the backtest tab) to have been run at least once; it does not need the
FastAPI server running alongside it.

Usage: .venv/Scripts/python.exe -m streamlit run apps/streamlit_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import streamlit as st

from stockpredictor.backtest.engine import METRIC_NAMES
from stockpredictor.backtest.registry import (
    read_latest_backtest_result,
    read_latest_ic_series,
    read_latest_return_calibration,
)
from stockpredictor.backtest.significance import run_significance_report
from stockpredictor.common.types import DataLayer, RiskProfile
from stockpredictor.explain.registry import read_explanations
from stockpredictor.features.sentiment import latest_sentiment_snapshot
from stockpredictor.models.calibration import SEPARATION_ALPHA, IsotonicCalibrator
from stockpredictor.monitoring.accuracy import compute_accuracy
from stockpredictor.portfolio.constructor import parse_horizon_days
from stockpredictor.portfolio.service import DEFAULT_STRATEGY_ID, construct_portfolio_from_lake
from stockpredictor.portfolio.targets import estimate_return_for_days, extrapolation_warning
from stockpredictor.ranking.registry import read_latest_rankings
from stockpredictor.reporting.analytics import compute_performance_analytics
from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake

DISCLAIMER = (
    "**For research/educational purposes only. Not investment advice.** "
    "Markets carry risk; past performance does not guarantee future results."
)
HORIZONS = ["5d", "30d", "90d"]
RISK_PROFILES = [RiskProfile.CONSERVATIVE, RiskProfile.BALANCED, RiskProfile.AGGRESSIVE]


@st.cache_resource
def get_lake() -> Lake:
    return Lake()


@st.cache_resource
def get_sessionmaker():
    # init_db is idempotent (CREATE TABLE IF NOT EXISTS, effectively -- see
    # storage/db.py) and every other entrypoint in this codebase
    # (orchestration/nightly_flow.py, the reporting/ scripts) already calls
    # it before touching the DB. The app previously didn't, which worked by
    # accident as long as app.db happened to predate a schema addition --
    # it broke the moment the Track Record tab queried the new
    # published_predictions/validation_results tables on a DB that hadn't
    # run any of the new scripts yet.
    engine = make_engine()
    init_db(engine)
    return make_sessionmaker(engine)


st.set_page_config(page_title="Stock Predictor (Research)", layout="wide")
st.title("AI Stock Research Dashboard")
st.markdown(
    "[![Nightly pipeline](https://github.com/arunbansal504/stock-predictor/actions/workflows/nightly.yml/badge.svg)]"
    "(https://github.com/arunbansal504/stock-predictor/actions/workflows/nightly.yml)"
)
st.warning(DISCLAIMER)

lake = get_lake()
sessionmaker_ = get_sessionmaker()

with st.sidebar:
    st.header("Controls")
    horizon = st.selectbox("Investment horizon", HORIZONS, index=0)
    top_n = st.selectbox("Top N", [5, 10, 20, 50, "Custom"], index=1)
    if top_n == "Custom":
        top_n = st.number_input("Custom N", min_value=1, max_value=500, value=15)
    strategy_id = st.text_input("Backtest strategy id", value=DEFAULT_STRATEGY_ID)
    risk_profile = st.selectbox(
        "Risk profile", RISK_PROFILES, index=1, format_func=lambda p: p.value.capitalize()
    )
    investment_amount = st.number_input("Investment amount (₹)", min_value=0.0, value=0.0)

tab_picks, tab_detail, tab_portfolio, tab_backtest, tab_transparency, tab_track_record = st.tabs(
    ["Top Picks", "Stock Detail", "Portfolio Constructor", "Backtest Lab", "Model Transparency", "Track Record"]
)

with tab_picks:
    st.subheader(f"Top {top_n} -- horizon {horizon}")
    ranked = read_latest_rankings(lake, horizon)
    if ranked.empty:
        st.info(
            "No rankings yet for this horizon. Run the nightly pipeline first: "
            "`prefect` flow `nightly_pipeline` in orchestration/nightly_flow.py, "
            "or scripts/run_phase1_smoke.py for a quick backtest-only run."
        )
    else:
        top = ranked[ranked["rank"] <= top_n].copy()
        st.caption(f"As of {top['date'].max().date()}")
        if "close_price" in top.columns:
            st.caption(
                "`price` is the raw closing price on the date above (not a live quote) -- "
                "the same price the ranking was computed from."
            )

        # `score` stays the primary ranking/display key (rank_universe already
        # sorts by it) -- everything below is context ON TOP of score, never
        # a replacement for it. Column order matters here: the compact,
        # glance-at-a-time columns (rank/symbol/price/score/disagreement)
        # come first so they're visible without scrolling; the verbose
        # `confidence` sentence and its supporting rate columns -- useful
        # detail, but wide and secondary to `score` -- go last, so only
        # someone who wants that detail needs to scroll for it.
        display_cols = ["rank", "symbol"]
        has_close_price = "close_price" in top.columns
        if has_close_price:
            display_cols.append("close_price")
        display_cols += ["score", "disagreement"]
        has_meta_score = "meta_score" in top.columns
        if has_meta_score:
            display_cols.append("meta_score")
        has_separation = "separation_direction" in top.columns
        # Background colors keyed by style name, applied to the `confidence`
        # cell below -- direction-blind styling (treating any "significant"
        # block as good news regardless of sign) was a real bug here, so the
        # color is driven by the same `separation_direction`/`separation_badge`
        # every other surface uses, not re-derived from the boolean alone.
        BADGE_COLORS = {"positive": "#c6efce", "negative": "#ffc7ce", "neutral": "#f2f2f2"}
        # Compact label for the table cell -- the full sentence
        # (separation_badge's `label`) truncates in a dataframe column no
        # matter where it's placed; the full wording is still used for the
        # Stock Detail tab's banner below, which has room for it.
        SHORT_DIRECTION = {"outperform": "Outperform", "underperform": "Underperform", "none": "Low separation"}
        if has_separation:
            badges = [
                IsotonicCalibrator.separation_badge(
                    row.separation_direction, row.separation_empirical_rate, int(row.separation_n), row.separation_base_rate
                )
                for row in top.itertuples()
            ]
            top["confidence"] = [
                f"{SHORT_DIRECTION[row.separation_direction]} "
                f"({row.separation_empirical_rate:.1%} vs {row.separation_base_rate:.1%})"
                for row in top.itertuples()
            ]
            top["_confidence_style"] = [b["style"] for b in badges]
            top["empirical_outperform_rate"] = top["separation_empirical_rate"]
            top["horizon_base_rate"] = top["separation_base_rate"]
            display_cols += ["empirical_outperform_rate", "horizon_base_rate", "confidence"]
        display = (
            top[display_cols]
            .rename(columns={"meta_score": "relative_strength", "close_price": "price"})
            .set_index("rank")
        )
        if has_separation:
            styles = top.set_index("rank")["_confidence_style"]
            styled = display.style.apply(
                lambda col: [f"background-color: {BADGE_COLORS[styles[idx]]}" for idx in col.index]
                if col.name == "confidence"
                else ["" for _ in col.index],
                axis=0,
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(display, use_container_width=True)
        st.bar_chart(top.set_index("symbol")["score"])

        if has_separation:
            n_out = int((top["separation_direction"] == "outperform").sum())
            n_under = int((top["separation_direction"] == "underperform").sum())
            n_none = len(top) - n_out - n_under
            horizon_base_rate = float(top["separation_base_rate"].iloc[0])
            st.caption(
                f"**Confidence** reports whether a stock's score sits in a calibration band "
                f"with a historically real (statistically significant, two-sided p < "
                f"{SEPARATION_ALPHA}) departure from this horizon's own base rate -- "
                f"currently **{horizon_base_rate:.1%}** (the fraction of all historical "
                "calibration-set rows, across the whole universe, whose forward return beat "
                "that same date's universe median stock; close to but not exactly 50% -- a "
                "median split isn't perfectly even once ties and unresolved rows are dropped "
                "-- not a fixed coin flip, and in which direction. Green = confirmed "
                "historical outperformance *relative to that base rate*; red = confirmed "
                "historical *under*performance relative to it (a negative signal, not a "
                "weaker positive one); grey = not statistically distinguishable from it. "
                f"Of the {len(top)} shown here: {n_out} outperform, {n_under} underperform, "
                f"{n_none} low separation. `score` remains the primary, outcome-calibrated "
                "ranking signal regardless of this flag. Note: block boundaries are "
                "data-derived from the calibration set itself, not fixed in advance -- "
                "treat these p-values as indicative of relative edge, not a formal "
                "family-wise guarantee."
            )
        if has_meta_score:
            st.caption(
                "`relative_strength` is **not** a probability and has **not** been validated "
                "against forward returns -- it's the model's raw, uncalibrated signal. "
                "`score` is calibrated via interpolation between historical evidence bands "
                "(see USER_GUIDE.md), so it and `empirical_outperform_rate` are no longer the "
                "same number -- `score` reflects this stock's interpolated position, while "
                "`empirical_outperform_rate` is the anchor band's own historical rate. "
                "`relative_strength` breaks any remaining exact ties in `rank` (identical raw "
                "scores, or scores clamped flat outside the calibration range), but it is not "
                "a second opinion of equal weight to `score`."
            )

with tab_detail:
    ranked = read_latest_rankings(lake, horizon)
    if ranked.empty:
        st.info("No rankings yet -- see the Top Picks tab.")
    else:
        symbol = st.selectbox("Symbol", sorted(ranked["symbol"].unique()))
        row = ranked[ranked["symbol"] == symbol].iloc[0]
        has_close_price = "close_price" in row.index and pd.notna(row.get("close_price"))
        cols = st.columns(4 if has_close_price else 3)
        cols[0].metric("Rank", int(row["rank"]))
        if has_close_price:
            cols[1].metric("Price (close)", f"{row['close_price']:.2f}")
            cols[2].metric("Score (calibrated prob. of outperformance)", f"{row['score']:.3f}")
            cols[3].metric("Ensemble disagreement", f"{row['disagreement']:.3f}")
        else:
            cols[1].metric("Score (calibrated prob. of outperformance)", f"{row['score']:.3f}")
            cols[2].metric("Ensemble disagreement", f"{row['disagreement']:.3f}")
        if "separation_direction" in row.index and pd.notna(row.get("separation_direction")):
            badge = IsotonicCalibrator.separation_badge(
                row["separation_direction"],
                row["separation_empirical_rate"],
                int(row["separation_n"]),
                row["separation_base_rate"],
            )
            banner_text = f"{badge['label']} (two-sided p < {SEPARATION_ALPHA})."
            if badge["style"] == "positive":
                st.success(banner_text)
            elif badge["style"] == "negative":
                st.error(banner_text)
            else:
                st.info(banner_text)
        if "meta_score" in row.index and pd.notna(row.get("meta_score")):
            st.caption(
                f"Relative strength: {row['meta_score']:.3f} -- an uncalibrated tie-break "
                "signal that has **not** been validated against forward returns, not a "
                "probability. See Top Picks tab for why this exists."
            )

        with st.expander("What if calculator (custom horizon)"):
            whatif_amount = st.number_input("Amount to invest (₹)", min_value=0.0, value=10000.0, key="whatif_amount")
            whatif_days = st.number_input("Days", min_value=1, value=30, step=1, key="whatif_days")
            reference_horizon_days = parse_horizon_days(horizon)
            return_calibration = read_latest_return_calibration(lake, strategy_id, horizon)
            expected_return_pct = (
                estimate_return_for_days(float(row["score"]), return_calibration, int(whatif_days), reference_horizon_days)
                if reference_horizon_days is not None
                else None
            )
            if expected_return_pct is None:
                st.info("No return calibration available yet for this strategy/horizon.")
            else:
                warning = (
                    extrapolation_warning(int(whatif_days), reference_horizon_days)
                    if reference_horizon_days is not None
                    else None
                )
                if warning is not None:
                    st.warning(warning)
                expected_return_amount = whatif_amount * expected_return_pct
                expected_final_value = whatif_amount + expected_return_amount
                wc1, wc2 = st.columns(2)
                wc1.metric("Expected return", f"{expected_return_pct:.2%}")
                wc2.metric(
                    "Expected final value (₹)",
                    f"₹{expected_final_value:,.0f}",
                    delta=f"₹{expected_return_amount:,.0f}",
                )
                st.caption(
                    f"Not a forecast -- a linear-in-time extrapolation of this strategy's own "
                    f"historical score-conditional return calibration (from the nearest published "
                    f"horizon, {horizon}) to {int(whatif_days)} day(s), not a dedicated calibration "
                    "for that exact day count -- the further from a published horizon, the less "
                    "this reflects any actual evidence. Derived from historical calibration, "
                    "same as the Portfolio Constructor tab's expected return."
                )

        explanations = read_explanations(lake, horizon)
        exp_row = explanations[explanations["symbol"] == symbol] if not explanations.empty else explanations
        if exp_row.empty:
            st.info("No SHAP explanation computed yet for this symbol/horizon.")
        else:
            exp = exp_row.iloc[0]
            st.markdown("**Factor block attribution** (SHAP, LightGBM base learner)")
            blocks = pd.Series(exp["factor_blocks"]).sort_values(ascending=False)
            st.bar_chart(blocks)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top positive signals**")
                for s in exp["top_positive_signals"]:
                    st.write(f"+ `{s['feature']}` ({s['block']}): {s['contribution']:+.4f}")
            with c2:
                st.markdown("**Top negative signals**")
                for s in exp["top_negative_signals"]:
                    st.write(f"- `{s['feature']}` ({s['block']}): {s['contribution']:+.4f}")

        st.markdown("---")
        st.markdown("**Recent news & sentiment** (last 5 days, FinBERT-scored)")
        news = lake.read(DataLayer.SILVER, "news", symbol)
        snapshot = latest_sentiment_snapshot(news, pd.Timestamp.today())
        if snapshot["article_count"] == 0:
            st.caption("No recent news found for this symbol (or the news pipeline hasn't run yet).")
        else:
            st.metric("Mean sentiment (5d, -1 to +1)", f"{snapshot['mean_sentiment']:+.2f}", help=f"{snapshot['article_count']} article(s)")
            for a in snapshot["articles"]:
                st.markdown(f"[{a['title']}]({a['url']})  \n{a['source']} · {a['published_date'].date()} · {a['sentiment_label']} ({a['sentiment_score']:+.2f})")
        st.caption(
            "Sentiment is scored (FinBERT) but not yet a trained-model feature -- "
            "the free news source has no historical archive, so there isn't enough "
            "history yet to evaluate it out-of-sample honestly. Shown here as live "
            "context only."
        )

with tab_portfolio:
    st.subheader(f"Portfolio Constructor -- Top {top_n}, {horizon}, {risk_profile.value} risk")
    portfolio = construct_portfolio_from_lake(
        lake, sessionmaker_, horizon, risk_profile, int(top_n), strategy_id,
        investment_amount=investment_amount if investment_amount > 0 else None,
    )
    if portfolio is None:
        st.info("No rankings yet for this horizon -- see the Top Picks tab.")
    elif not portfolio.positions:
        st.warning(
            "No candidates had enough price history to build a portfolio "
            f"(excluded: {', '.join(portfolio.excluded_symbols) or 'none'})."
        )
    else:
        if portfolio.diversification_warning:
            st.warning(portfolio.diversification_warning)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Expected return (over {horizon})", f"{portfolio.expected_return:.2%}" if portfolio.expected_return is not None else "N/A")
        c2.metric("Expected volatility (annualized)", f"{portfolio.expected_volatility:.2%}" if portfolio.expected_volatility is not None else "N/A")
        c3.metric("Expected Sharpe", f"{portfolio.expected_sharpe:.2f}" if portfolio.expected_sharpe is not None else "N/A")
        c4.metric("Capital allocated", f"{portfolio.total_allocated_weight:.0%}")

        if portfolio.expected_final_value is not None:
            st.metric(
                "Expected final value (₹)",
                f"₹{portfolio.expected_final_value:,.0f}",
                delta=f"₹{portfolio.expected_return_amount:,.0f}" if portfolio.expected_return_amount is not None else None,
            )

        show_amounts = investment_amount > 0
        positions_df = pd.DataFrame(
            [
                {
                    "symbol": p.symbol,
                    "weight": p.weight,
                    **({"allocated_amount": p.allocated_amount, "expected_return_amount": p.expected_return_amount} if show_amounts else {}),
                    "sector": p.sector,
                    "score": p.score,
                    "entry": p.entry_price,
                    "stop_loss": p.stop_loss,
                    "target": p.target_price,
                    "expected_return": p.expected_return,
                }
                for p in portfolio.positions
            ]
        ).sort_values("weight", ascending=False)
        if show_amounts:
            positions_df["allocated_amount"] = positions_df["allocated_amount"].map(
                lambda v: f"₹{v:,.0f}" if v is not None else "N/A"
            )
            positions_df["expected_return_amount"] = positions_df["expected_return_amount"].map(
                lambda v: f"₹{v:,.0f}" if v is not None else "N/A"
            )
        st.dataframe(positions_df.set_index("symbol"), use_container_width=True)
        st.bar_chart(positions_df.set_index("symbol")["weight"])

        if portfolio.excluded_symbols:
            st.caption(f"Excluded (insufficient price history): {', '.join(portfolio.excluded_symbols)}")

        st.caption(
            "Allocation via Hierarchical Risk Parity, tilted by model confidence and capped by the "
            "selected risk profile's position/sector limits. Stop-loss/target are ATR-based; expected "
            "return comes from this strategy's own historical score-conditional return calibration, not a forecast. "
            + portfolio.disclaimer
        )

with tab_backtest:
    st.subheader(f"Backtest: {strategy_id} -- horizon {horizon}")
    result = read_latest_backtest_result(lake, strategy_id, horizon)
    if result is None:
        st.info(
            "No backtest result yet for this strategy/horizon. Run "
            "`scripts/run_phase1_smoke.py` to produce one."
        )
    else:
        # run_date is stored normalized to midnight (see backtest/registry.py --
        # it's a calendar-date dedup key, not a precise timestamp), so
        # displaying just the date avoids a redundant, misleading "00:00:00".
        st.caption(f"Run date: {pd.Timestamp(result['run_date']).date()}")
        # Pull only the known metric fields (not e.g. "strategy_id", which
        # incidentally also starts with "strategy_" and would otherwise
        # contaminate this table via naive prefix-stripping).
        metrics_df = pd.DataFrame(
            {
                "strategy": {m: result.get(f"strategy_{m}") for m in METRIC_NAMES},
                "benchmark": {m: result.get(f"benchmark_{m}") for m in METRIC_NAMES},
            }
        )
        st.dataframe(metrics_df, use_container_width=True)
        st.caption(f"Mean Information Coefficient across test dates: {result['mean_ic']:.4f}")

        ic_series = read_latest_ic_series(lake, strategy_id, horizon)
        if len(ic_series) >= 2:
            report = run_significance_report(ic_series)
            ttest, bootstrap = report["ttest"], report["bootstrap"]
            st.markdown("**Is that IC actually distinguishable from noise?**")
            c1, c2, c3 = st.columns(3)
            c1.metric("t-test p-value", f"{ttest['p_value']:.3f}")
            c2.metric(
                "95% CI (t-test)",
                f"{ttest['ci_low']:+.4f} to {ttest['ci_high']:+.4f}",
            )
            c3.metric(
                "Bootstrap: resamples ≤ 0",
                f"{bootstrap['fraction_non_positive']:.1%}",
            )
            if not ttest["significant_at_5pct"]:
                st.warning(
                    f"Not statistically significant at the 5% level (p={ttest['p_value']:.3f}, "
                    f"n={ttest['n_periods']} periods) -- the confidence interval straddles zero, "
                    "so this mean IC could plausibly be noise rather than real signal."
                )
            else:
                st.success(
                    f"Statistically significant at the 5% level (p={ttest['p_value']:.3f}, "
                    f"n={ttest['n_periods']} periods)."
                )

            with st.expander("Robustness detail (sub-period stability, autocorrelation)"):
                subperiods = report["subperiods"]
                if not subperiods.empty:
                    st.markdown("**Sub-period stability** -- is the edge steady, or one lucky stretch?")
                    st.dataframe(subperiods.set_index("period"), use_container_width=True)
                    consistent = report["consistent_sign_across_subperiods"]
                    if consistent is False:
                        st.caption(
                            "Sub-periods disagree on direction -- the overall mean IC is being pulled "
                            "by an inconsistent signal, weaker evidence than a steady edge throughout."
                        )
                    elif consistent is True:
                        st.caption("Every sub-period has the same sign as the overall mean -- a mildly reassuring sign.")
                autocorr = report["lag1_autocorrelation"]
                if not pd.isna(autocorr):
                    st.caption(
                        f"Lag-1 autocorrelation of the IC series: {autocorr:+.3f}. Rebalance dates are "
                        "already non-overlapping, but a high value here means the periods aren't fully "
                        "independent, and the p-value above should be read as optimistic, not corrected for it."
                    )
        else:
            st.caption(
                "Not enough persisted per-date IC history yet to test statistical significance "
                "(needs a backtest run after this feature was added)."
            )

        curve = pd.DataFrame(result["equity_curve"])
        if not curve.empty:
            curve["strategy_equity"] = (1 + curve["strategy_return"]).cumprod()
            curve["benchmark_equity"] = (1 + curve["benchmark_return"]).cumprod()
            st.markdown("**Cumulative equity (out-of-fold, net of estimated costs)**")
            st.line_chart(curve.set_index("date")[["strategy_equity", "benchmark_equity"]])

        st.caption(
            "Reminder: a too-good result here would be treated as a leakage bug, not a win. "
            "This is a small-universe, short-history research backtest -- not a performance claim."
        )

with tab_transparency:
    st.subheader(f"Model Transparency -- horizon {horizon}")
    accuracy = compute_accuracy(lake, horizon)
    if accuracy is None:
        st.info(
            "Not enough resolved prediction history yet. Predictions accumulate "
            "one snapshot per nightly run; run the pipeline for several days/weeks "
            "before this becomes meaningful."
        )
    else:
        st.caption(f"Based on {accuracy['n_resolved_predictions']} resolved predictions")
        deciles = pd.Series(accuracy["hit_rate_by_score_decile"]).sort_index()
        deciles.index = [f"decile {i}" for i in deciles.index]
        st.markdown("**Hit rate by predicted-score decile** (top decile should beat bottom decile)")
        st.bar_chart(deciles)
        st.caption(
            "If every decile looks equally good, that's a leakage red flag, not a win (§30) -- "
            "a real, if modest, edge should show up as a rising staircase from decile 0 to 9."
        )

with tab_track_record:
    st.subheader("Track Record -- published predictions vs. actual outcomes")
    st.caption(
        "This tab is horizon-independent (every published prediction, not just the sidebar's "
        "selected horizon) and reads the same `published_predictions`/`validation_results` "
        "tables as `reports/dashboard/index.html` and the monthly ML Review -- all three "
        "surfaces share reporting/analytics.py's compute_performance_analytics so they can't "
        "silently disagree. Unlike Top Picks (which re-ranks every night), a *published* "
        "prediction is the official weekly Top-10 snapshot, frozen and never edited after the "
        "fact -- see USER_GUIDE.md's \"weekly track record\" glossary section for what that means."
    )
    analytics = compute_performance_analytics(lake, sessionmaker_)
    if analytics["n_resolved"] == 0:
        st.info(
            f"{analytics['n_published']} prediction(s) published so far, none resolved yet -- "
            "a 90-day pick published this week won't resolve for about three months. Check back "
            "once the first horizon completes."
        )
    else:
        st.caption(f"{analytics['n_published']} published, {analytics['n_resolved']} resolved.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Overall accuracy", f"{analytics['overall_accuracy']:.1%}")
        c2.metric("Top-5 accuracy", f"{analytics['top_5_accuracy']:.1%}" if analytics["top_5_accuracy"] is not None else "N/A")
        c3.metric("Top-10 accuracy", f"{analytics['top_10_accuracy']:.1%}" if analytics["top_10_accuracy"] is not None else "N/A")
        c4.metric("Avg alpha (vs. NIFTY 500)", f"{analytics['avg_alpha']:+.2%}")
        c5.metric("Win rate", f"{analytics['win_rate']:.1%}")

        r6, r12 = analytics.get("rolling_6m"), analytics.get("rolling_12m")
        rc1, rc2 = st.columns(2)
        rc1.metric(
            "Rolling 6-month hit rate",
            f"{r6['hit_rate']:.1%} (n={r6['n']})" if r6 else "N/A",
        )
        rc2.metric(
            "Rolling 12-month hit rate",
            f"{r12['hit_rate']:.1%} (n={r12['n']})" if r12 else "N/A",
        )

        if analytics["by_horizon_ratios"]:
            st.markdown("**Risk-adjusted return by horizon** (resolved predictions only)")
            ratios_df = pd.DataFrame(analytics["by_horizon_ratios"]).T
            st.dataframe(ratios_df, use_container_width=True)

        if analytics["monthly_stats"]:
            # Plain st.bar_chart is one flat series color (same as every
            # other chart in this app) -- it can't conditionally color bars
            # by sign the way reports/dashboard/index.html's custom SVG
            # chart does, so this doesn't claim a green/red distinction it
            # can't render. The table right below carries the exact sign.
            st.markdown("**Monthly avg alpha** (positive = beat benchmark that month)")
            monthly_df = pd.DataFrame(analytics["monthly_stats"]).T
            st.bar_chart(monthly_df["avg_alpha"])
            st.dataframe(monthly_df, use_container_width=True)

        if analytics["probability_distribution"]:
            st.markdown("**Published prediction-probability distribution**")
            st.bar_chart(pd.Series(analytics["probability_distribution"]))

        st.caption(
            "n_periods here can still be small -- treat Sharpe/Sortino/CAGR with the same "
            "sample-size caution as the Backtest Lab tab (§ that tab's own n_periods caveat). "
            "The full monthly write-up, with sector/feature-level detail, is in "
            "`reports/YYYY-MM-ML-Review.md`, generated automatically on the 1st of each month."
        )
