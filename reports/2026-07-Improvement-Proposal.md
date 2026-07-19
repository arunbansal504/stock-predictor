# Improvement Proposal -- 2026-07

## Recommendations

### [LOW] No rule-triggered recommendations this month

- **Expected Impact:** Low
- **Implementation Complexity:** n/a
- **Confidence:** Low
- **Reasoning:** None of the standing thresholds (feature importance, sector alpha, IC trend, calibration spread) were tripped by this month's data.
- **Supporting Evidence:** See ML-Review.md for the underlying tables.


## Model Change Proposal

### Features to Add

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

### Features to Remove

_No data available yet._

### Hyperparameters to Tune

<!-- ANALYSIS: fill in during monthly review --> -- see `config/model.yaml` for the current base-learner/calibration configuration; no automated hyperparameter search exists yet.

### Models Worth Testing

- Gradient-boosted alternatives to LightGBM (XGBoost, CatBoost) as an additional base learner.
<!-- ANALYSIS: fill in during monthly review -->

### Feature Engineering Improvements

<!-- ANALYSIS: fill in during monthly review -->

### Data Quality Improvements

- Sentiment history depth -- see the project's standing sentiment-readiness follow-up.
- Fundamentals remain sparse/laggy by nature of the free data source (see features/fundamental.py).

### Potential Bugs Detected

None newly detected by this report. Check `run_metadata` (orchestration/run_tracking.py) for any failed pipeline stages this month.

### Ranking Improvements

<!-- ANALYSIS: fill in during monthly review -->

### Probability Calibration Improvements

<!-- ANALYSIS: fill in during monthly review --> -- see ML-Review.md Question 13's decile table.

### Risk Metrics to Add

- Sector/industry concentration limits across the published Top-N.
- A Bottom-N "avoid" list, to make false-negative analysis (Question 10) possible.

### Expected Benefit / Confidence Score

<!-- ANALYSIS: fill in during monthly review -->
