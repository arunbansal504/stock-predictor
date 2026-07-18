# Stock Predictor — AI Stock Research Platform (NSE/BSE)

> **Research/education tool. Not investment advice.** Markets carry risk;
> past performance does not guarantee future results. See the full
> architecture rationale (including *why* this framing, not a "buy/sell"
> advisory product) in the design doc referenced below.

An institutional-inspired, personal-scale system that ranks NSE-listed
stocks by a **calibrated probability of forward outperformance** vs. a
benchmark, across multiple horizons — with full explainability (SHAP factor
attribution), walk-forward backtesting, and honest calibration reporting.
Built lean and free-tier by design: `yfinance` for data, DuckDB/Parquet +
SQLite for storage, LightGBM + a linear baseline for modeling, Prefect for
orchestration, FastAPI + Streamlit for serving.

This is **Phase 1** of a phased roadmap (see the architecture document) —
an "Honest EOD Ranker": end-of-day batch, technical features only, real
NIFTY 500 universe (fetched live from NSE each run — see "Universe" below).
Fundamentals, news/sentiment, options data, and the full Portfolio
Optimizer are later phases, deliberately deferred rather than half-built.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"          # Windows
# .venv/bin/pip install -e ".[dev]"             # macOS/Linux

# Run the tests (no network required — everything's mocked)
.venv/Scripts/python -m pytest -q

# Run the full nightly pipeline against live data (universe -> prices ->
# features -> labels -> predict -> rank -> explain). Takes a few minutes.
.venv/Scripts/python -c "from stockpredictor.orchestration.nightly_flow import nightly_pipeline; nightly_pipeline()"

# Or run a walk-forward backtest validation (separate from the pipeline above —
# proves the *ranking* has out-of-sample skill, doesn't publish a live ranking):
.venv/Scripts/python scripts/run_phase1_smoke.py

# Serve the dashboard (reads whatever the pipeline last published)
.venv/Scripts/python -m streamlit run apps/streamlit_app/app.py

# Or the API
.venv/Scripts/python -m uvicorn stockpredictor.api.app:app --reload
```

No configuration is required for local dev — a fresh checkout defaults to
SQLite (`data/app.db`) and a local DuckDB/Parquet lake (`data/`). Copy
`.env.example` to `.env` only if you want Postgres, paid data sources, an
LLM, or Telegram alerts (see comments in that file).

## Architecture

- **Bronze/Silver/Gold** medallion lake (`data/`) via `storage/lake.py`,
  queryable through DuckDB; small relational reference data (securities,
  corporate actions, run audit trail) in Postgres/SQLite via
  `storage/db.py` + `storage/models.py`.
- **Point-in-time discipline** (`common/pit.py`) is the backbone of the
  whole pipeline: every dataset distinguishes an event date from a
  *knowable* date, and `models/walk_forward.py` enforces a label-resolution
  embargo on top of that — see `tests/leakage/` for what this catches.
- **Ensemble**: LightGBM + a regularized-linear honesty baseline, stacked
  via a meta-learner trained on a held-out chronological split (not
  refit-on-full-data, to avoid a train/predict distribution mismatch), then
  isotonic-calibrated (`models/ensemble.py`, `models/calibration.py`).
- **Backtesting** (`backtest/`): walk-forward with an India-specific
  transaction-cost model, non-overlapping rebalance-date subsampling
  (`select_rebalance_dates`), and standard metrics (CAGR, Sharpe, Sortino,
  Calmar, max drawdown, hit-rate-by-decile, information coefficient).
- **Explainability** (`explain/`): SHAP on the LightGBM base learner,
  aggregated into factor blocks (Momentum/Trend, Oscillators,
  Volatility/Risk, Volume/Liquidity) with top positive/negative signal
  lists per stock.
- **Orchestration** (`orchestration/nightly_flow.py`): a Prefect flow
  wiring ingestion → features → labels → predict/rank/explain, with
  data-quality gates, an audit trail in `run_metadata`, and freshness/drift
  monitoring (`monitoring/`).
- **Serving**: FastAPI (`api/app.py`) and a Streamlit dashboard
  (`apps/streamlit_app/app.py`) both read pre-computed Gold-layer
  artifacts — neither ever triggers model inference on demand.

Every design decision above has a documented rationale in the module
docstrings — start there before changing behavior, not just the code.

## Universe

The pipeline fetches NSE's real, current NIFTY 500 constituent list live
(`connectors/universe_nse.py`, `ind_nifty500list.csv` from NSE's archives)
at the start of every run — not a hand-maintained approximation. If that
fetch fails, it falls back to the small 40-symbol seed CSV
(`config/universe_seed.csv`) and logs/alerts about it; that CSV is
scaffolding, kept for offline tests and as a documented fallback, not
meant to be the primary universe. Newer/smaller constituents will
naturally have less price history than large-caps — per-symbol partial
history is handled gracefully, not treated as a failure.

## Honesty notes (read before trusting any output)

- A too-good backtest result should be treated as a leakage bug, not a win.
- Calibration and out-of-sample skill (Model Transparency tab / `/accuracy`
  endpoint) only become meaningful after the pipeline has run for enough
  days that predictions have actually resolved.

## Project layout

See `src/stockpredictor/` for the package (`connectors/`, `ingestion/`,
`features/`, `labels/`, `models/`, `prediction/`, `ranking/`, `explain/`,
`backtest/`, `orchestration/`, `monitoring/`, `api/`, `storage/`,
`common/`), `apps/streamlit_app/` for the dashboard, `scripts/` for
one-off/research entrypoints, and `tests/{unit,contract,leakage,integration}/`
for the test suite.
