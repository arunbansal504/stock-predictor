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
from stockpredictor.backtest.registry import read_latest_backtest_result, read_latest_ic_series
from stockpredictor.backtest.significance import run_significance_report
from stockpredictor.common.types import DataLayer, RiskProfile
from stockpredictor.explain.registry import read_explanations
from stockpredictor.features.sentiment import latest_sentiment_snapshot
from stockpredictor.monitoring.accuracy import compute_accuracy
from stockpredictor.portfolio.service import DEFAULT_STRATEGY_ID, construct_portfolio_from_lake
from stockpredictor.ranking.registry import read_latest_rankings
from stockpredictor.storage.db import make_engine, make_sessionmaker
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
    return make_sessionmaker(make_engine())


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

tab_picks, tab_detail, tab_portfolio, tab_backtest, tab_transparency = st.tabs(
    ["Top Picks", "Stock Detail", "Portfolio Constructor", "Backtest Lab", "Model Transparency"]
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
        st.dataframe(
            top[["rank", "symbol", "score", "disagreement"]].set_index("rank"),
            use_container_width=True,
        )
        st.bar_chart(top.set_index("symbol")["score"])

with tab_detail:
    ranked = read_latest_rankings(lake, horizon)
    if ranked.empty:
        st.info("No rankings yet -- see the Top Picks tab.")
    else:
        symbol = st.selectbox("Symbol", sorted(ranked["symbol"].unique()))
        row = ranked[ranked["symbol"] == symbol].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Rank", int(row["rank"]))
        col2.metric("Score (calibrated prob. of outperformance)", f"{row['score']:.3f}")
        col3.metric("Ensemble disagreement", f"{row['disagreement']:.3f}")

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
        lake, sessionmaker_, horizon, risk_profile, int(top_n), strategy_id
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

        positions_df = pd.DataFrame(
            [
                {
                    "symbol": p.symbol,
                    "weight": p.weight,
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
        st.dataframe(positions_df.set_index("symbol"), use_container_width=True)
        st.bar_chart(positions_df.set_index("symbol")["weight"])

        if portfolio.excluded_symbols:
            st.caption(f"Excluded (insufficient price history): {', '.join(portfolio.excluded_symbols)}")

        st.caption(
            "Allocation via Hierarchical Risk Parity, tilted by model confidence and capped by the "
            "selected risk profile's position/sector limits. Stop-loss/target are ATR-based; expected "
            "return comes from this strategy's own historical score-decile calibration, not a forecast. "
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
