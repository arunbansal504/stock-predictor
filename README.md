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
stocks by a **calibrated probability of forward outperformance** — beating
that same trading day's *universe median stock*, not a benchmark index (see
"Honesty notes" for why) — across multiple horizons, with full
explainability (SHAP factor attribution), walk-forward backtesting, honest
calibration reporting, and a Portfolio Constructor (HRP allocation,
risk-profile position/sector caps, ATR stop-loss/target). Every nightly (or
manual) run is pinned to the last *completed* NSE session, so re-running the
pipeline on an unchanged day reproduces byte-identical rankings — see
"Honesty notes". Built lean and free-tier by design: `yfinance` for
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
.venv/Scripts/python scripts/run_phase1_smoke.py   # 5d only, full pipeline incl. ingestion
.venv/Scripts/python scripts/run_backtests.py       # 5d/30d/90d, reuses data already in the lake

# Or audit today's predictions stock-by-stock (news, sentiment, technicals,
# feature vector, base-learner/meta/calibrated scores, feature importances,
# calibrator block table) -- read-only, trains in memory, writes nothing:
.venv/Scripts/python scripts/run_prediction_diagnostics.py --horizon 90d --top 10

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
- **Ensemble**: LightGBM (deterministic mode — see below) + a
  regularized-linear honesty baseline, stacked via a meta-learner, then
  isotonic-calibrated -- a three-way *chronological* split (base learners /
  meta-learner / calibrator), not two, so the calibrator is fit on truly
  out-of-sample meta-learner predictions, never on rows the meta-learner
  itself trained on (`models/ensemble.py`, `models/calibration.py`).
- **Backtesting** (`backtest/`): walk-forward with an India-specific,
  turnover-aware transaction-cost model (a position held over from the
  prior rebalance isn't charged a fresh round trip), non-overlapping
  rebalance-date subsampling (`select_rebalance_dates`), and standard
  metrics (CAGR, Sharpe, Sortino, Calmar, max drawdown, hit-rate-by-decile,
  information coefficient) reported for the strategy, the cap-weighted
  benchmark, *and* an equal-weight "hold the whole universe" baseline —
  beating the benchmark alone isn't evidence of ranking skill if simply
  holding everything would have done even better (see "Honesty notes").
  `backtest/significance.py` goes one step further than reporting a mean
  IC: a t-test against "mean IC is actually zero," a distribution-free
  bootstrap confidence interval, sub-period stability (is the edge steady
  or one lucky stretch?), and a lag-1 autocorrelation check (sanity-checks
  the t-test's independence assumption) — surfaced in the Backtest Lab
  tab. Not run by the nightly pipeline (which never re-runs the
  walk-forward backtest at all) — only when a backtest script
  (`scripts/run_phase1_smoke.py` for 5d, `scripts/run_backtests.py` for all
  three published horizons) is run manually.
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
  portfolio "expected return" comes from the backtest's own isotonic
  (monotonic, interpolated) score-conditional historical realized returns
  (`backtest/calibration_curve.py`), never fabricated from the classifier's
  score directly (it was never calibrated for return *magnitude*, only
  probability). Not model inference — a deterministic optimization over
  already-published rankings, safe to compute on demand
  (`POST /portfolio/construct`, Streamlit's "Portfolio Constructor" tab).
  An optional `investment_amount` (₹) threads through to ₹-denominated
  `allocated_amount`/`expected_return_amount`/`expected_final_value` fields
  alongside the existing weights/percentages — pure multiplication of the
  same numbers above, None/skipped when no amount is given so existing
  callers are unaffected. A separate `POST /stocks/{symbol}/whatif`
  endpoint (and a "What if calculator" expander on the Stock Detail tab)
  answers "invest ₹X for N days" for an arbitrary day count that doesn't
  have its own published calibration curve, by scaling the nearest
  published horizon's (5d/30d/90d) calibrated return *linearly* with
  time — deliberately not the sqrt-of-time convention `expected_sharpe`
  uses, since that scales a return/volatility *ratio*, whereas expected
  return itself scales linearly under the same random-walk assumption
  (sqrt-of-time on the raw return would silently understate long-horizon
  extrapolations) — rather than inventing a new return model
  (`portfolio/targets.py`'s `estimate_return_for_days`). A day count more
  than `MAX_REASONABLE_EXTRAPOLATION_MULTIPLE` (10x) away from the selected
  horizon, in either direction, surfaces an explicit
  `extrapolation_warning` in the API response (and a yellow warning banner
  in the UI) — correct scaling math doesn't make a 5-day curve stretched
  to 5000 days meaningful, so that's flagged rather than presented at face
  value (`portfolio/targets.py`'s `extrapolation_warning`).
- **Orchestration** (`orchestration/nightly_flow.py`): a Prefect flow
  wiring ingestion → features → labels → predict/rank/explain, with
  data-quality gates, an audit trail in `run_metadata`, and freshness/drift
  monitoring (`monitoring/`).
- **Serving**: FastAPI (`api/app.py`) and a Streamlit dashboard
  (`apps/streamlit_app/app.py`) both read pre-computed Gold-layer
  artifacts — neither ever triggers model inference on demand. Every
  ranked row carries `close_price` — the raw (not split/dividend-adjusted)
  closing price the ranking was actually computed from
  (`ranking/engine.py`), so a score/rank is never shown without the price
  it corresponds to.

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

**Monthly backtest audit**: `.github/workflows/monthly_backtest.yml` runs
`scripts/run_backtests.py` at 06:00 UTC on the 1st of each month (also
triggerable on demand via `workflow_dispatch`), reusing whatever
prices/features/labels the nightly workflow has already accumulated in the
lake — it does not re-ingest anything, only re-validates. This exists so
the CAGR/Sharpe/IC numbers in "Honesty notes" below and the Backtest Lab
tab don't quietly go stale for months between manual runs; results are
committed back to `data/gold/backtests/` the same way the nightly pipeline
persists `data/`. It shares the nightly workflow's `nightly-pipeline`
concurrency group (by name, across the two separate workflow files) so the
two never race each other's `data/` commit-and-push.

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
rather skip the wasted run. (The three ML Review Board workflows —
weekly publish, daily validation, monthly review, see "ML Review Board" —
are currently GitHub-Actions-only; there's no local-scheduler equivalent
documented for them yet.) To inspect, run on demand, or remove:

```powershell
schtasks /query /tn "StockPredictor Nightly"
schtasks /run /tn "StockPredictor Nightly"      # trigger immediately, e.g. to test
schtasks /delete /tn "StockPredictor Nightly"   # /f to skip the confirmation prompt
```

## ML Review Board

A governance layer on top of the nightly pipeline above, not a replacement
for it: it freezes an official weekly recommendation set, tracks what
actually happened to it, and produces a monthly report a human reads before
deciding whether anything about the model should change. **Nothing in this
layer ever retrains the production model or deploys a change automatically**
— every report ends with recommendations for a human to act on, never an
action taken on their behalf.

- **Weekly publish** (`reporting/publish.py`,
  `.github/workflows/weekly_prediction.yml`, Fridays 15:00 UTC): freezes
  that week's official Top-10 (90d horizon by default) into the
  `published_predictions` table (`storage/models.py`) — full snapshot
  (buy price, calibrated probability, a disagreement-derived `confidence`,
  `relative_strength` — the same `meta_score`-based meaning this term
  already has in the Streamlit UI/USER_GUIDE.md glossary, not a second
  definition — technical/sentiment/model-input feature vectors, model
  version, git commit) — plus
  `predictions/YYYY-MM-DD.csv`/`.json`. Runs as its own job on a fresh
  checkout, so it rebuilds technical features and retrains from committed
  silver/gold data rather than assuming nightly's in-memory state is
  available (`data/gold/features` is deliberately git-ignored — see
  `.gitignore`); the determinism fix (`common/trading_calendar.py`) makes
  that recompute reproduce nightly's own ranking for the same date. Never
  overwrites a previously published date/horizon.
- **Daily validation** (`reporting/validation.py`,
  `.github/workflows/daily_validation.yml`, daily 15:00 UTC): resolves any
  published prediction whose horizon has completed against the Gold
  `labels` domain (reusing the model's own forward-return computation, not
  a re-derivation of it) and stores actual return, benchmark return, alpha,
  hit/miss, max drawdown, max gain, volatility, Sharpe, and information
  ratio into `validation_results`.
- **Performance dashboard** (`reporting/analytics.py` +
  `reporting/dashboard.py`, regenerated at the end of every daily
  validation run): a static, self-contained `reports/dashboard/index.html`
  — accuracy/alpha/Sharpe headline tiles, monthly and rolling 6/12-month
  performance, risk-adjusted return by horizon, and the prediction
  probability distribution. The same `compute_performance_analytics` also
  powers a live **Track Record** tab in the Streamlit app
  (`apps/streamlit_app/app.py`) — both surfaces compute from the identical
  function, so they can't silently disagree.
- **Monthly ML review** (`reporting/review.py`,
  `.github/workflows/monthly_ml_review.yml`, 1st of the month, 07:00 UTC):
  `reports/YYYY-MM-ML-Review.md` and `reports/YYYY-MM-Improvement-
  Proposal.md`. **Fully data-driven, not LLM-authored** — every table and
  number is computed from real `published_predictions`/`validation_results`
  /`explanations` history; sections that call for qualitative judgment are
  left as explicit `<!-- ANALYSIS: fill in during monthly review -->`
  placeholders rather than fabricated prose, and recommendations are
  rule-based off computed thresholds (feature importance, sector alpha, IC
  trend, calibration spread), not free text. The workflow opens a GitHub
  Issue summarizing the month's findings — the trigger for a human to look,
  not an automated decision.
- **Explainability** (Part 8 of the original spec): already covered by the
  existing SHAP pipeline (`explain/`) — nightly persists explanations for
  its top-20 per horizon (a superset of the weekly-published top-10) to the
  git-committed `gold/explanations` domain, joined by `reporting/review.py`
  on (symbol, date, horizon) rather than recomputed.
- **Human approval**: a human reads the two monthly reports and the
  dashboard, then decides — ignore, open/triage the auto-created GitHub
  issue, retrain, add features, or tune hyperparameters. No code path
  anywhere in this layer skips that decision.

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
fetch fails, it falls back to yesterday's already-synced universe
(last-known-good, from the `securities` table) rather than collapsing
straight to the small 40-symbol seed CSV (`config/universe_seed.csv`) — a
sudden ~500 → 40 symbol drop would itself reshuffle every cross-sectional
feature and swing ranks for reasons unrelated to any stock's actual
behavior. The CSV is the last resort, for an empty/fresh database with no
prior sync. Newer/smaller constituents will naturally have less price
history than large-caps — per-symbol partial history is handled
gracefully, not treated as a failure.

Every run also snapshots that day's resolved membership list to the
`universe_membership` gold domain (`ingestion/universe.py`), so a future
backtest can eventually restrict each historical rebalance date to the
universe as it stood point-in-time, instead of applying today's membership
across years of history (survivorship bias). Accrues going forward only —
history before this started recording remains biased; free data provides
no retroactive fix for that.

## Honesty notes (read before trusting any output)

- A too-good backtest result should be treated as a leakage bug, not a win.
- Calibration and out-of-sample skill (Model Transparency tab / `/accuracy`
  endpoint) only become meaningful after the pipeline has run for enough
  days that predictions have actually resolved.
- **Determinism**: every run — nightly or manual — pins its data cutoff to
  the last *completed* NSE session (`common/trading_calendar.py`), so
  re-running the pipeline the same day, before tomorrow's close, with
  nothing else changed reproduces byte-identical rankings. If two runs on
  the same day *do* produce different ranks, that's a regression, not
  expected noise — `tests/integration/test_determinism.py` guards this.
- **`score` vs. `empirical_outperform_rate`**: these are related but not the
  same number. `models/calibration.py`'s `IsotonicCalibrator` still fits
  PAVA blocks internally (grouping raw model output into evidence bands with
  similar historical outcomes — `empirical_outperform_rate`/
  `separation_*` describe those bands directly), but `score` itself is a
  *centered-isotonic* interpolation between each block's own historical rate
  and its neighbors', so it stays honestly differentiated per stock instead
  of flattening every stock in a band to one identical value. Run
  `scripts/run_prediction_diagnostics.py` to see this end to end for real
  stocks — per-symbol feature vectors, base-learner/meta/calibrated scores,
  and the calibrator's block table in one place.
- **The model does not yet demonstrate real ranking skill**, as of the most
  recent `scripts/run_backtests.py` run: on all three published horizons
  (5d/30d/90d), the strategy beats the cap-weighted NIFTY 500 benchmark,
  but *underperforms simply holding the whole equal-weight universe* — on
  CAGR and every risk-adjusted metric. 30d's mean IC was slightly negative.
  Beating the benchmark alone is not evidence of skill here — see the
  `universe (hold-everything)` column `simulate_top_k_strategy` (and the
  Backtest Lab tab) reports specifically to catch this. Don't treat current
  rankings as validated for real allocation decisions; re-run
  `scripts/run_backtests.py` periodically as more history accrues and
  check this comparison yourself before trusting it.

## Project layout

See `src/stockpredictor/` for the package (`connectors/`, `ingestion/`,
`features/`, `labels/`, `models/`, `prediction/`, `ranking/`, `explain/`,
`backtest/`, `portfolio/`, `sentiment/`, `orchestration/`, `monitoring/`,
`api/`, `storage/`, `common/`, `reporting/` — the ML Review Board layer, see
"ML Review Board"), `apps/streamlit_app/` for the dashboard,
`scripts/` for one-off/research entrypoints (plus
`publish_weekly_predictions.py`/`run_daily_validation.py`/
`generate_monthly_review.py`, the ML Review Board's entrypoints),
`tests/{unit,contract,leakage,integration}/` for the test suite,
`predictions/` for the never-overwritten weekly CSV/JSON exports,
`reports/` for the generated dashboard (`reports/dashboard/index.html`) and
monthly `*-ML-Review.md`/`*-Improvement-Proposal.md` pairs, and
`.github/workflows/` for the scheduled runs (`nightly.yml`,
`monthly_backtest.yml`, `weekly_prediction.yml`, `daily_validation.yml`,
`monthly_ml_review.yml` — see "Scheduling & Deployment" and "ML Review
Board"). `requirements.txt` exists only for Streamlit Community Cloud's
deploy step — local/CI installs use `pyproject.toml`.
