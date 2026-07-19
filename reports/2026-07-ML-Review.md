# ML Review -- 2026-07

Generated from `published_predictions` and `validation_results` history. 0 predictions resolved this month; 0 resolved all-time.

## 1. Which predictions succeeded?

_No data available yet._

## 2. Which predictions failed?

_No data available yet._

## 3. Which sectors consistently outperform?

(All-time history, sectors with >= 3 resolved predictions.)

_No data available yet._

## 4. Which sectors consistently underperform?

_No data available yet._

## 5. Which features contributed positively?

(Mean SHAP contribution across this month's resolved predictions' Top-N explanations.)

_No data available yet._

## 6. Which features had little or no predictive power?

(Mean |SHAP contribution| below 0.3%.)

_No data available yet._

## 7. Did news sentiment improve predictions?

Sentiment features (`features/sentiment.py`) are computed and stored but are **not yet part of `ALL_FEATURE_COLUMNS`** -- the news connector has no historical backfill, so real day-by-day coverage only started accumulating from nightly ingestion (see the project's standing sentiment-readiness follow-up). This question cannot be honestly answered from live model behavior until sentiment is actually folded into the feature set and evaluated out-of-sample.

## 8. Did technical indicators improve predictions?

(Mean SHAP contribution by factor block -- see explain/factors.py -- for this month's resolved predictions; a positive mean contribution suggests the block helped rather than hurt.)

<!-- ANALYSIS: fill in during monthly review -->

## 9. Were false positives concentrated in specific industries?

(Sector distribution of this month's misses.)

_No data available yet._

## 10. Were false negatives concentrated in specific industries?

This system only publishes a Top-N recommendation list -- it never records an explicit "avoid this stock" prediction, so there is no natural false-negative population to measure here. If this question matters going forward, consider also tracking a Bottom-N list (see Part 6, Risk Metrics to Add).

## 11. Is the model becoming stale?

(Information Coefficient -- prediction probability vs. realized return -- by month, last 0 months with >=3 resolved predictions.)

_No data available yet._


## 12. Is there evidence of overfitting?

Latest committed walk-forward backtest (`top_k_technical_fundamental_v1`, 90d): mean IC = 0.0592, strategy Sharpe = 1.3240. Compare against this month's live figures above -- a live IC persistently far above the backtested mean IC across many months would be the overfitting red flag; a single month is not enough evidence either way.

## 13. Is prediction confidence well calibrated?

(Hit rate by predicted-probability decile, all-time resolved history -- monitoring/accuracy.py.)

_No data available yet._

## 14. Which stocks repeatedly fooled the model?

(All-time, symbols with >= 2 misses.)

_No data available yet._

## 15. Which indicators appear redundant?

Candidates are the same low-importance features from Question 6, cross-checked against each other for near-duplicate signal (e.g. two moving-average variants moving together) -- that cross-check is a judgment call, not computable from SHAP alone.

<!-- ANALYSIS: fill in during monthly review -->

## 16. Which new features could improve prediction quality?

| Candidate | Rationale |
| --- | --- |
| Options Open Interest | Derivatives positioning often leads spot price moves. |
| FII/DII activity | Institutional flow is a well-documented Indian-market factor. |
| Quarterly earnings surprise | Post-earnings drift is a documented anomaly. |
| Mutual fund holdings | Ownership concentration/changes can precede re-ratings. |
| Insider trading | SAST disclosures are a public, point-in-time-safe signal source. |
| Analyst upgrades/downgrades | Sell-side revisions are a fast-moving sentiment proxy. |
| Alternative news providers | Broadens beyond Google News RSS's no-historical-backfill limit. |
| Macroeconomic indicators | Rate/inflation/currency regime context, currently unused. |
