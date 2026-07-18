"""Turns per-row SHAP attributions into the factor-block summary and top
positive/negative signal lists shown on a recommendation card (§11, §27
step 11).
"""

from __future__ import annotations

import pandas as pd

from stockpredictor.explain.factors import feature_to_block
from stockpredictor.explain.shap_explainer import compute_shap_values


def summarize_factor_blocks(shap_row: pd.Series) -> pd.Series:
    """Sum SHAP contributions within each factor block for one row (one
    stock's explanation), sorted most-positive first."""
    blocks = shap_row.index.map(feature_to_block)
    return shap_row.groupby(blocks).sum().sort_values(ascending=False)


def top_signals(shap_row: pd.Series, n: int = 5) -> dict[str, list[dict]]:
    """The `n` most positive and `n` most negative individual feature
    contributions for one row, each tagged with its factor block -- the
    literal "positive signals" / "negative signals" lists on a
    recommendation card."""
    sorted_row = shap_row.sort_values(ascending=False)
    positive = sorted_row[sorted_row > 0].head(n)
    negative = sorted_row[sorted_row < 0].tail(n).sort_values()

    def _to_list(series: pd.Series) -> list[dict]:
        return [
            {"feature": name, "block": feature_to_block(name), "contribution": float(value)}
            for name, value in series.items()
        ]

    return {"positive": _to_list(positive), "negative": _to_list(negative)}


def explain_predictions(model, X: pd.DataFrame, symbols: pd.Series, n_signals: int = 5) -> pd.DataFrame:
    """End-to-end: compute SHAP for `model.lgbm` (the StackedRanker's
    LightGBM base learner, see models/ensemble.py) on X, and produce a
    per-symbol explanation record (factor-block summary + top signals)."""
    if X.empty:
        return pd.DataFrame(columns=["symbol", "factor_blocks", "top_positive_signals", "top_negative_signals"])

    shap_df = compute_shap_values(model.lgbm, X)

    records = []
    for idx, symbol in zip(X.index, symbols):
        row = shap_df.loc[idx]
        blocks = summarize_factor_blocks(row)
        signals = top_signals(row, n=n_signals)
        records.append(
            {
                "symbol": symbol,
                "factor_blocks": blocks.to_dict(),
                "top_positive_signals": signals["positive"],
                "top_negative_signals": signals["negative"],
            }
        )
    return pd.DataFrame(records)
