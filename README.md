# Stock Predictor — AI Stock Research Platform (NSE/BSE)

> **Research/education tool. Not investment advice.** Markets carry risk;
> past performance does not guarantee future results. See the full
> architecture rationale (including *why* this framing, not a "buy/sell"
> advisory product) in the design doc referenced below.

**New to trading terminology?** Start with [USER_GUIDE.md](USER_GUIDE.md)
— a plain-language glossary (score, ATR, Sharpe ratio, etc.) and a
walkthrough of every screen in the dashboard, written for someone with no
trading background.

An institutional-inspired, personal-scale system that ranks NSE-listed
stocks by a **calibrated probability of forward outperformance** vs. a
benchmark, across multiple horizons — with full explainability (SHAP factor
attribution), walk-forward backtesting, honest calibration reporting, and
a Portfolio Constructor (HRP allocation, risk-profile position/sector caps,
ATR stop-loss/target). Built lean and free-tier by design: `yfinance` for
data, Google News RSS + a local FinBERT model for news/sentiment, DuckDB/Parquet
+ SQLite for storage, LightGBM + a linear baseline for modeling, Prefect for
orchestration, FastAPI + Streamlit for serving.

This is **Phase 1(+)** of a phased roadmap (see the architecture document)
— an "Honest EOD Ranker" over the real, live NIFTY 500 universe (fetched
from NSE each run — see "Universe" below), now also incorporating annual
fundamentals (PE/PB/ROE/ROA/D-E/net margin, PIT-correct) alongside the
technical feature set, plus a Portfolio Constructor on top of the ranking.
News + FinBERT sentiment scoring now runs nightly too (see "News &
Sentiment" below) and powers a live per-stock panel, though it isn't yet a
trained-model feature. Options data remains later-phase, deliberately
deferred rather than half-built.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,sentiment]"    # Windows
# .venv/bin/pip install -e ".[dev,sentiment]"       # macOS/Linux
# `sentiment` pulls in transformers/torch (~1-2GB) for the FinBERT news
# scorer -- only needed to run ingestion; the full test suite and the
# dashboard both run fine without it (`.[dev]`), see pyproject.toml.

# Run the tests (no network required — everything's mocked)
.venv/Scripts/python -m pytest -q

# Run the full nightly pipeline against live data (universe -> prices ->
# features -> labels -> news/sentiment -> predict -> rank -> explain).
# A first run over the full ~500-symbol universe takes well over an hour
# (mostly per-symbol news fetch + FinBERT scoring); see "Scheduling" below
# to run this unattended instead of watching it live.
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
  Volatility/Risk, Volume/Liquidity, Fundamental/Quality) with top
  positive/negative signal lists per stock.
- **Fundamentals** (`connectors/fundamentals_yfinance.py`,
  `features/fundamental.py`): annual statements, PIT-stamped via actual
  historical earnings-announcement dates (falling back to SEBI's 60-day
  filing deadline), joined against daily prices via `pd.merge_asof` for
  PE/PB (price-dependent, computed daily) plus ROE/ROA/D-E/net margin.
  Revenue/EPS YoY growth are computed but deliberately excluded from the
  model-facing feature set — see that module's docstring for the live
  backtest evidence behind that call.
- **Portfolio Constructor** (`portfolio/`): turns a ranked Top-N into an
  illustrative allocation via Hierarchical Risk Parity (robust to noisy
  covariance estimates, unlike Markowitz), tilted by model confidence and
  capped by a Conservative/Balanced/Aggressive risk profile's
  position/sector limits. Per-stock stop-loss/target are ATR-based;
  portfolio "expected return" comes from the backtest's own
  score-decile-conditional historical realized returns
  (`backtest/calibration_curve.py`), never fabricated from the classifier's
  score directly (it was never calibrated for return *magnitude*, only
  probability). Not model inference — a deterministic optimization over
  already-published rankings, safe to compute on demand
  (`POST /portfolio/construct`, Streamlit's "Portfolio Constructor" tab).
- **Orchestration** (`orchestration/nightly_flow.py`): a Prefect flow
  wiring ingestion → features → labels → predict/rank/explain, with
  data-quality gates, an audit trail in `run_metadata`, and freshness/drift
  monitoring (`monitoring/`).
- **Serving**: FastAPI (`api/app.py`) and a Streamlit dashboard
  (`apps/streamlit_app/app.py`) both read pre-computed Gold-layer
  artifacts — neither ever triggers model inference on demand.

Every design decision above has a documented rationale in the module
docstrings — start there before changing behavior, not just the code.

## Scheduling & Deployment

The nightly pipeline (§17, §21: "nightly batch as a scheduled deployment")
is idempotent/resumable by construction, so it's safe to run unattended.
Two ways to schedule it — pick one (GitHub Actions is the primary/
recommended path; local Task Scheduler is a documented alternative if you'd
rather run on your own machine, e.g. for full control over long FinBERT
runs without touching a shared CI runner).

### Option A: GitHub Actions (recommended — runs without your machine on)

`.github/workflows/nightly.yml` runs the pipeline daily at 19:00 IST
(13:30 UTC — after NSE close; adjust the `cron:` line to change it) on a
GitHub-hosted runner, then commits the updated `data/` back to `main`.

**Why commit data back to git**: GitHub Actions runners are ephemeral — a
fresh, empty disk every run. Without persisting `data/` somewhere, the
accumulated news/sentiment history would be destroyed every single night
(Google News RSS has no historical archive to re-fetch from — see
"News & Sentiment" below), and prices/fundamentals would need a full
5-year re-fetch every run instead of a cheap incremental one. Git-as-
storage is the zero-new-infrastructure way to solve that for a personal-
scale project — see `.gitignore`: `data/bronze/` and `data/gold/features/`
stay out of git (both fully regenerated from scratch every run with zero
dependency on a prior committed copy, so persisting them would only add
git history bloat for no benefit — `gold/features` alone was ~250MB
rewritten wholesale every night before this was fixed; ATR, its only
serving-side consumer, is now computed on demand from `silver/prices` via
`portfolio/service.py`'s `_latest_atr_by_symbol`). The rest of `data/silver/`
and `data/gold/`, plus the SQLite app db, are committed.

Setup: push this repo to GitHub (already done if you're reading this from
there) — the workflow runs automatically on the schedule once merged to
`main`. No secrets are required to run; optionally add
`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` as **repo Settings → Secrets and
variables → Actions** secrets to receive failure alerts (`monitoring/
alerts.py` degrades to log-only otherwise). Trigger a run on demand from
the **Actions** tab (`workflow_dispatch`) to test it without waiting for
the schedule. A run has historically taken **~2.5 hours**, dominated by
per-symbol FinBERT scoring across the ~500-symbol universe — the workflow
sets `timeout-minutes: 300` and a concurrency guard so a second scheduled
trigger can't overlap a still-running one.

**Viewing results**: deploy `apps/streamlit_app/app.py` to
[Streamlit Community Cloud](https://share.streamlit.io) (free, connects
directly to this GitHub repo): sign in with GitHub → "New app" → pick this
repo/branch → main file path `apps/streamlit_app/app.py`. It installs from
`requirements.txt` (deliberately lean — no transformers/torch, which the
app never imports; see that file's header comment) and auto-redeploys on
every push, including the workflow's own nightly data commits — so the
hosted dashboard reflects last night's run automatically, no manual step.

### Option B: local Windows Task Scheduler

`scripts/run_nightly.ps1` wraps the same pipeline with what an unattended
*local* run needs instead: a persisted log file
(`data/logs/nightly_<timestamp>.log`, 30-day retention, not versioned) and
a propagated process exit code, so a failed run shows up in Task
Scheduler's own history too.

```powershell
schtasks /create /tn "StockPredictor Nightly" /tr "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"C:\path\to\repo\scripts\run_nightly.ps1\"" /sc daily /st 19:00
```

Adjust `/st` (start time, 24h HH:MM, local time) to whenever your machine
is reliably on — after NSE close (15:30 IST) is the only hard constraint;
weekends/holidays are harmless no-ops (the pipeline tolerates stale gaps,
see "Honesty notes"), not worth excluding via `/sc weekly` unless you'd
rather skip the wasted run. To inspect, run on demand, or remove:

```powershell
schtasks /query /tn "StockPredictor Nightly"
schtasks /run /tn "StockPredictor Nightly"      # trigger immediately, e.g. to test
schtasks /delete /tn "StockPredictor Nightly"   # /f to skip the confirmation prompt
```

## News & Sentiment

Nightly ingestion (`connectors/news_rss.py`, `sentiment/`) fetches each
company's current Google News RSS results, filters them for actual
relevance (`sentiment/relevance.py`), and scores each headline with FinBERT
(`sentiment/classifier.py`, `ProsusAI/finbert`) into a signed
[-1, 1] polarity score. `features/sentiment.py` turns that into PIT-correct
rolling aggregates (5d/20d sentiment, momentum, volume, dispersion).

**Deliberately not yet a trained-model feature** (not in
`ALL_FEATURE_COLUMNS`): unlike prices or annual fundamentals, free news RSS
has no historical archive to backfill from — real news history only starts
accumulating from whenever nightly ingestion first ran. Training on columns
that are almost entirely NaN across a multi-year backtest window already
backfired once for fundamentals' growth ratios (see
`features/fundamental.py`'s docstring: IC dropped from 0.037 to 0.015); the
same failure mode would apply here, worse. So this is intentionally wired
in now to *start accumulating real data*, and shown live in the Streamlit
Stock Detail tab (recent headlines + FinBERT sentiment) as real value today
— folding it into the trained model is a later, evidence-gated step once
enough calendar time has passed to evaluate it out-of-sample honestly.

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
`backtest/`, `portfolio/`, `sentiment/`, `orchestration/`, `monitoring/`,
`api/`, `storage/`, `common/`), `apps/streamlit_app/` for the dashboard,
`scripts/` for one-off/research entrypoints,
`tests/{unit,contract,leakage,integration}/` for the test suite, and
`.github/workflows/nightly.yml` for the scheduled pipeline run (see
"Scheduling & Deployment"). `requirements.txt` exists only for Streamlit
Community Cloud's deploy step — local/CI installs use `pyproject.toml`.
