# Beginner's Guide — Understanding & Using This App

You don't need any trading background to read this. Everything here is
explained in plain language first, with the exact numbers you'll actually
see on screen.

**Read this section first, seriously:** This app produces **research
output**, not investment advice. It shows you a *calibrated probability*
that a stock will outperform the market over some period — a number between
0 and 1, closer to "this has been a somewhat better bet historically" than
to "this will definitely go up." Nothing here is a guarantee, a tip, or a
recommendation to buy or sell. Every screen in the app repeats this because
it matters: markets carry real risk, and past performance never guarantees
future results. If you ever act on anything from this app, that's your own
informed decision, ideally after learning more and possibly talking to a
qualified financial advisor.

---

## 1. Glossary — every term you'll see, in plain language

### The basics

| Term | What it means here |
|---|---|
| **Symbol / Ticker** | The short code for a company on the exchange, e.g. `RELIANCE`, `TCS`, `INFY`. |
| **NSE / NIFTY 500** | NSE = National Stock Exchange (India's main exchange). NIFTY 500 = the 500 largest, most liquid companies listed there — the universe of stocks this app considers. |
| **Horizon** | How far into the future the prediction looks: `5d` = 5 trading days, `30d` = 30 trading days, `90d` = 90 trading days. A stock can rank very differently on different horizons — a 5-day pick and a 90-day pick are answering different questions. |
| **Benchmark** | The NIFTY 500 index, shown in the Backtest Lab as a comparison point for the *strategy's* overall performance. It is **not** what an individual stock's score is measured against (see Score below) — a common point of confusion, since earlier versions of this app did use the benchmark that way. |
| **Price** | The stock's actual closing price on the date the ranking was computed (shown next to each rank) — not a live/current quote, and not split-or-dividend-adjusted (it's the real number you'd have seen quoted that day). Shown so a score/rank is never presented without the price it corresponds to. |

### The score and how much to trust it

| Term | What it means here |
|---|---|
| **Score** | A number between 0 and 1 — the model's *calibrated probability* that this stock's return over the chosen horizon will beat that same day's **median stock in the universe** (not the benchmark index — see Benchmark above). A score of 0.52 means: historically, when the model gave stocks a score around 0.52, they beat the typical stock that day about 52% of the time. It is **not** "52% expected gain" — it's a probability of being *better than a typical pick*, which is a different (and in a strong market, often harder) bar than just "going up," and a different question from "beating the index." |
| **Rank** | The stock's position when every candidate is sorted by score, best first. Rank 1 = highest score that day. When several stocks share the exact same score (see **Relative strength** below for why that happens honestly), ties are broken by relative strength — not arbitrarily, and not by row order. |
| **Relative strength** | A second number shown alongside score, used only to order stocks that share the identical score. **It is not a probability** — it's the model's raw, uncalibrated internal signal, before the honesty-check step (calibration) is applied. It's shown because calibration can legitimately give many different stocks the *exact same* score (see Calibration below) — without this number, the ranking among those tied stocks would look arbitrary even though it isn't. Use it only to understand *why* one tied stock outranks another, never as a confidence number on its own. |
| **Ensemble disagreement** | This app doesn't use one model — it uses several (a tree-based model and a simpler linear one) and blends them. Disagreement measures how much those models *disagree* with each other on this particular stock. Low disagreement = the models broadly agree, which is a mild trust signal. High disagreement = the models see it differently, which is a reason for extra caution, even if the blended score looks good. |
| **Calibration** | A calibrated score means the number is *honest* — if the model says 0.6, that should really happen about 60% of the time when checked against history. Look at the **Model Transparency** tab to see whether that's actually true for this system right now. Calibration works by grouping raw predictions into bands with similar historical outcomes — which means many different stocks can legitimately land in the same band and get the *exact same* score, especially near the edges where there's less history to work with. That's calibration being honest about the limits of the evidence, not a bug. |

### Reading a stock's "why"

| Term | What it means here |
|---|---|
| **SHAP / factor attribution** | A technique that explains *why* the model gave a stock its score, by showing which inputs pushed the score up and which pushed it down. It's the model's own reasoning, not a separate written opinion. |
| **Factor block** | Individual signals are grouped into readable categories: **Momentum/Trend** (is the stock trending up, relative to its recent history?), **Oscillators** (is it "overbought" or "oversold" short-term?), **Volatility/Risk** (how much does the price swing around?), **Volume/Liquidity** (how much is actually being traded?), **Fundamental/Quality** (valuation and profitability from the company's financial statements — P/E, P/B, ROE, ROA, debt levels, margins), and (shown separately, not yet part of the score) **News/Sentiment**. |
| **Top positive / negative signals** | The specific individual features that most helped or hurt this stock's score, e.g. `+ rsi_14 (Oscillators): +0.021` means the 14-day RSI oscillator reading pushed the score up by 0.021. |
| **News & sentiment panel** | Recent headlines about the company (via Google News), each scored by a finance-tuned AI model (FinBERT) as positive, negative, or neutral, with a numeric score from -1 (very negative) to +1 (very positive). This is shown as live context, but — important — **it does not yet feed into the score above**. See the note on that tab for why (short version: there isn't enough historical news data yet to prove it actually helps, so it isn't used until that's tested honestly). |

### Sector, fundamentals & valuation terms

| Term | What it means here |
|---|---|
| **Sector** | The industry group a company belongs to (IT, Healthcare, Financial Services, Power, etc.) — used to check a portfolio isn't overloaded in one industry. |
| **P/E ratio (pe_ratio)** | Price-to-Earnings: the share price divided by earnings per share. Roughly, "how many years of current profit would it take to earn back what you paid" — lower can mean cheaper (or the market expects trouble), higher can mean expensive (or the market expects growth). |
| **P/B ratio (pb_ratio)** | Price-to-Book: share price divided by book value (net assets) per share. Similar idea to P/E but based on the balance sheet instead of earnings. |
| **ROE / ROA** | Return on Equity / Return on Assets — how efficiently a company turns shareholders' money (ROE) or all its assets (ROA) into profit. Higher generally means more efficient. |
| **Debt-to-Equity** | How much the company has borrowed relative to shareholders' own money. Higher = more financial leverage = more risk if things go wrong. |
| **Net margin** | Profit as a percentage of revenue — how much of every rupee of sales actually becomes profit. |

### The Portfolio Constructor screen

| Term | What it means here |
|---|---|
| **Risk profile** | A preset that controls how concentrated or spread-out the suggested portfolio is: |
| | **Conservative** — at most 10% in any one stock, at most 25% in any one sector, wants at least 10 different positions, tighter stop-losses. |
| | **Balanced** — at most 15% per stock, 35% per sector, at least 6 positions. |
| | **Aggressive** — at most 25% per stock, 50% per sector, only needs 3 positions, wider stops. |
| **Weight / allocation** | What share of the (hypothetical) portfolio's money goes into each stock, e.g. `0.15` = 15% of the portfolio. |
| **HRP (Hierarchical Risk Parity)** | The method used to decide those weights. In plain terms: it groups stocks that tend to move *together* (correlated) and treats them as if they were partly "the same bet," so it doesn't accidentally put too much money into five stocks that all rise and fall together. It leans toward putting more weight on steadier (lower-volatility) stocks. This is considered more robust than older methods (like Markowitz mean-variance optimization) which are notoriously sensitive to small estimation errors. |
| **Confidence tilt** | On top of the risk-based HRP weights, positions get nudged up or down slightly based on the model's score — a higher-scoring stock gets a bit more weight, within the risk profile's caps. |
| **Entry price** | The most recent available closing price — the reference point stop-loss/target are measured from, not a price you're guaranteed to actually get. |
| **ATR (Average True Range)** | A standard measure of how much a stock's price typically moves per day, averaged over the last 14 trading days. A stock with ATR ₹50 tends to swing about ₹50 a day; a stock with ATR ₹5 is much calmer. Used here to size stop-losses/targets to each stock's *own* normal volatility rather than a one-size-fits-all percentage. |
| **Stop-loss** | A suggested price below entry where, if the stock falls that far, the position would be considered to have failed — a discipline device to cap downside, not a guarantee the stock won't fall further before or after that level. Set as `entry - (multiplier × ATR)`, where the multiplier depends on risk profile (1.5x for Conservative, up to 2.5x for Aggressive). |
| **Target price** | A suggested price above entry representing a reasonable place to consider taking profit. Set as a multiple of the stop-loss distance (the "reward" is set to be some multiple of the "risk," e.g. 2:1 for Balanced). |
| **Expected return** | The portfolio's estimated return over the chosen horizon — **not a forecast conjured from the score**. It comes from looking at history: "stocks that scored around this level, historically, went on to return about this much." Honest, but still just a historical average, not a promise. |
| **Expected volatility** | How much the *whole portfolio's* value is estimated to swing around, annualized (scaled to a one-year basis for comparability, even though the actual holding period may be shorter). |
| **Expected Sharpe** | Expected return divided by expected volatility (risk-adjusted, both put on the same annualized time basis). Higher is generally better — it means more return per unit of risk taken. A Sharpe around 1 is considered decent, above 2 is very good, but treat this as a rough historical estimate, not a promise of the future. |
| **Diversification warning** | Shown when the selected number of stocks or the risk profile's caps don't allow for a well-spread portfolio (e.g., too few candidates for how diversified the risk profile expects you to be). |
| **Capital allocated** | What percentage of the (hypothetical) total money actually got assigned to a position. Sometimes this is less than 100% — see the diversification warning if so; it means the position/sector caps mathematically couldn't fit more into fewer names. |

### The Backtest Lab screen

A **backtest** simulates how a strategy *would have* performed on past data it wasn't allowed to see in advance — the main way this system checks itself honestly before you'd ever trust its live picks.

| Term | What it means here |
|---|---|
| **CAGR** | Compound Annual Growth Rate — the smoothed, annualized rate of return, as if it grew steadily every year. |
| **Sharpe ratio** | Return per unit of risk (see above) — shown for the **strategy** (this system's picks), the **benchmark** (just holding the NIFTY 500 index), and the **universe (hold-everything)** baseline (equal-weight, holding every eligible stock) so you can compare all three. |
| **Universe (hold-everything)** | A third comparison column alongside strategy/benchmark: what you'd have gotten by just holding *every* eligible stock in equal amounts, no ranking involved. This is the more important comparison — beating the cap-weighted benchmark index can happen just from broad market exposure having a good run, with the ranking itself adding nothing. If the strategy doesn't beat this column too, the ranking hasn't yet shown it's actually picking better stocks than a coin flip would. |
| **Sortino ratio** | Like Sharpe, but only penalizes *downside* volatility (bad swings), not upside swings — a stock going up a lot fast isn't "risk" in the way this ratio counts it. |
| **Calmar ratio** | Return divided by the worst drawdown experienced — how much return you got per unit of "worst pain endured." |
| **Max drawdown** | The largest peak-to-trough decline over the backtest period, e.g. `-0.21` means a 21% drop from a high point at some stage. This is the "how bad could it have gotten" number. |
| **Win rate** | The fraction of rebalance periods where the strategy had a positive return. |
| **n_periods** | How many independent time periods the backtest actually covers — a small number here means take everything else with extra caution; there just isn't much history yet to be confident in. |
| **Information Coefficient (IC)** | A single number (roughly -1 to +1) measuring how well the score's *ranking* of stocks lined up with what actually happened. In real-world quant finance, an IC around 0.03–0.05 is considered a genuinely useful, if modest, edge — this isn't a game where 0.5+ is realistic; anything close to that would be a red flag for a bug (leaked future data), not a win. |
| **p-value** | The probability of seeing an edge this good (or better) purely by random chance, if there were actually *no* real signal at all. Smaller is stronger evidence — under 0.05 is the usual line for calling something "statistically significant." A mean IC that *looks* positive but has a large p-value (say, 0.4) is a coin flip dressed up as a result. |
| **Statistically significant** | Shorthand for "the p-value cleared that 0.05 bar" — the result probably isn't just noise. It does **not** mean the edge is large, guaranteed, or will persist; a small, real edge and a large, fake one can both clear this bar. |
| **95% confidence interval (CI)** | A range of plausible values for the *true* IC, given the data available. If the range doesn't include zero (e.g. "+0.004 to +0.029"), that's consistent with a real, if modest, edge. This app reports two versions that should roughly agree: one from a standard t-test, and one from bootstrapping (below) — agreement between the two is itself a good sign. |
| **Bootstrap** | A way of double-checking the p-value/CI *without* assuming the results follow a textbook bell curve — it resamples the actual historical results thousands of times and sees how often a real edge would still show up. Shown as "% of resamples ≤ 0": a low number means it was rare, in resampling, for the edge to disappear or flip negative. |
| **Sub-period stability** | Splits the backtest history into chunks (e.g. first half vs. second half) and checks whether the edge shows up in *both*, not just one lucky stretch. A mean IC built from one great period and one terrible one is much weaker evidence than the same average built from two steady, similar periods. |
| **Autocorrelation** | Checks whether consecutive backtest periods are secretly echoing each other rather than being independent measurements. High autocorrelation would mean the "195 periods" behind a result are worth fewer than 195 genuinely independent pieces of evidence, making the p-value overconfident. Low (near zero) is the reassuring answer. |
| **Equity curve** | A chart of ₹1 growing (or shrinking) over time if you'd followed the strategy versus just holding the benchmark — the visual version of CAGR/Sharpe. |

### The Model Transparency screen

| Term | What it means here |
|---|---|
| **Hit rate by score decile** | All past predictions are split into 10 equal-sized groups (deciles) from lowest score (0) to highest score (9), and this shows how often each group's stocks actually beat that same day's median stock (see Score above). A trustworthy model shows a **rising staircase** — decile 9 (highest-scored stocks) winning more often than decile 0 (lowest-scored). If every decile looks about the same, the score isn't actually predictive yet. |
| **Resolved predictions** | Predictions where enough time has passed for the horizon to actually play out, so we know whether they were right or wrong. Predictions made yesterday for a 90-day horizon aren't "resolved" for 90 days — this number grows slowly, which is normal, not a bug. |

---

## 2. Walking through each screen

Open the app (`streamlit run apps/streamlit_app/app.py`, or your deployed
Streamlit Cloud URL) and use the sidebar to set:
- **Investment horizon** — 5d / 30d / 90d
- **Top N** — how many stocks to show/consider
- **Backtest strategy id** — leave as default unless you know you want a different one
- **Risk profile** — Conservative / Balanced / Aggressive

Then the five tabs:

**Top Picks** — the ranked list for your chosen horizon: rank, symbol,
price, score, and disagreement, plus a bar chart. Start here to see what's
currently ranked highest.

**Stock Detail** — pick any symbol to see its rank/score/disagreement, the
SHAP factor breakdown (why the model scored it this way), and recent
news/sentiment. Use this to sanity-check *why* a stock is ranked where it
is, not just trust the number blindly.

**Portfolio Constructor** — turns your Top N into an illustrative
allocation: weights, entry/stop/target per stock, and portfolio-level
expected return/volatility/Sharpe. This is the "if I were to build a
portfolio from this list, here's one disciplined way to size and risk-manage
it" screen — again, illustrative, not an instruction.

**Backtest Lab** — the honesty check: how would this exact strategy have
performed historically, compared to just holding the index? Look at this
*before* trusting anything else on the other screens. Below the metrics
table, a "Is that IC actually distinguishable from noise?" panel runs the
p-value/confidence-interval/bootstrap/sub-period checks described above —
this is the single most important box on the whole dashboard to actually
read, not skim past.

**Model Transparency** — the calibration check: are the scores actually
meaningful yet, based on resolved history? Early on (few resolved
predictions), this will say there isn't enough data yet — that's expected,
not broken.

---

## 3. A sane way to actually use this, as a beginner

1. **Start with Backtest Lab and Model Transparency**, not Top Picks. Get a
   feel for whether this system has *any* real track record yet, and how
   modest/noisy that track record honestly is (small `n_periods` = be extra
   skeptical).
2. **Then look at Top Picks and Stock Detail** for a horizon that matches
   your actual intended holding period — a 5-day score tells you nothing
   useful about a stock you plan to hold for 3 months.
3. **Use Portfolio Constructor as a sizing/risk-management template**, not
   a shopping list — the position caps and stop-losses exist specifically
   to stop any single bad pick from doing serious damage.
4. **Never treat a single day's Top Picks as final.** This is a batch
   system that re-ranks every night; today's rank 1 can be tomorrow's
   rank 50.
5. **Watch the disagreement and diversification-warning fields.** They're
   the app's own built-in "be careful here" signals.
6. **If you're not sure what a number means while using the app, come back
   to this guide (or the "Honesty notes" section of README.md) rather than
   guessing.**

---

## 4. Common beginner mistakes this app is specifically designed to help you avoid

- **Reading "score 0.55" as "55% chance of profit."** It's a probability of
  *beating that day's typical stock*, not of making money at all (the whole
  market can fall, and a stock can "outperform" its peers while still
  losing money in absolute terms).
- **Treating a backtest Sharpe of 1.4 as a promise.** It's a historical
  measurement over a specific (probably still fairly short) window. Small
  sample size = wide uncertainty, even when the number itself looks good.
- **Ignoring high ensemble disagreement because the score looks fine.**
  Disagreement is exactly the kind of thing an experienced trader learns to
  respect — it means the underlying signals are genuinely mixed.
- **Assuming the stop-loss will actually execute at that exact price.**
  Real markets can gap past a stop-loss level, especially around news —
  the suggested stop is a discipline tool, not a guaranteed exit price.
- **Confusing "not yet a trained-model feature" with "not useful."** The
  news/sentiment panel is real, live data — it's just not (yet) proven to
  improve the score, so it's shown separately rather than silently baked
  in without evidence.
- **Trusting a positive mean IC without checking its p-value.** A small
  positive number averaged over a modest number of periods can easily be
  noise. The Backtest Lab's significance panel exists specifically so you
  don't have to take "the average was positive" on faith.
- **Note: the Backtest Lab's numbers don't update every night.** Unlike
  the rankings (which refresh nightly), the backtest is a separate,
  manually-run check — the significance numbers reflect whenever it was
  last run, not today's data.
- **Only looking at the benchmark comparison, not the "universe
  (hold-everything)" one.** As of the most recent backtest run, this
  system's strategy beats the NIFTY 500 benchmark on all three horizons —
  but *loses* to simply holding the whole equal-weight universe on every
  one of them. That's the honest current state: real evidence of skill
  over "beat the index," not yet real evidence of skill over "beat a coin
  flip." Check both columns, not just one.
- **Seeing many stocks with the identical score and assuming the ranking
  is broken.** It isn't — that's honest calibration legitimately grouping
  similar predictions together (see Calibration above). The rank order
  among those tied stocks is still meaningful; check the **relative
  strength** column to see why one outranks another.

---

*This guide describes the app as of the current codebase. If a screen or
number doesn't match what's described here, the code changed since this
was written — check the relevant module's docstring (mentioned throughout
README.md) for the current, authoritative behavior.*
