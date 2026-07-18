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
from stockpredictor.backtest.registry import read_latest_backtest_result
from stockpredictor.explain.registry import read_explanations
from stockpredictor.monitoring.accuracy import compute_accuracy
from stockpredictor.ranking.registry import read_latest_rankings
from stockpredictor.storage.lake import Lake

DISCLAIMER = (
    "**For research/educational purposes only. Not investment advice.** "
    "Markets carry risk; past performance does not guarantee future results."
)
HORIZONS = ["5d", "30d", "90d"]
DEFAULT_STRATEGY_ID = "top_k_technical_v1"


@st.cache_resource
def get_lake() -> Lake:
    return Lake()


st.set_page_config(page_title="Stock Predictor (Research)", layout="wide")
st.title("AI Stock Research Dashboard")
st.warning(DISCLAIMER)

lake = get_lake()

with st.sidebar:
    st.header("Controls")
    horizon = st.selectbox("Investment horizon", HORIZONS, index=0)
    top_n = st.selectbox("Top N", [5, 10, 20, 50, "Custom"], index=1)
    if top_n == "Custom":
        top_n = st.number_input("Custom N", min_value=1, max_value=500, value=15)
    strategy_id = st.text_input("Backtest strategy id", value=DEFAULT_STRATEGY_ID)

tab_picks, tab_detail, tab_backtest, tab_transparency = st.tabs(
    ["Top Picks", "Stock Detail", "Backtest Lab", "Model Transparency"]
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

with tab_backtest:
    st.subheader(f"Backtest: {strategy_id} -- horizon {horizon}")
    result = read_latest_backtest_result(lake, strategy_id, horizon)
    if result is None:
        st.info(
            "No backtest result yet for this strategy/horizon. Run "
            "`scripts/run_phase1_smoke.py` to produce one."
        )
    else:
        st.caption(f"Run date: {result['run_date']}")
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
