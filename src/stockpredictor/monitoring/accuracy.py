"""Historical calibration/accuracy computation (§15 "Model Transparency"
screen), shared between the API (api/app.py) and the Streamlit UI so the two
surfaces can't silently disagree on how "accuracy" is defined.

Joins accumulated predictions (one row per symbol/date/horizon, appended
every nightly run -- see prediction/registry.py) against labels that have
since resolved, and computes a hit-rate-by-decile calibration check: a top
decile that isn't meaningfully better than the bottom decile means the model
isn't adding value yet, which is the honest signal this exists to surface
(§30), not a bug to hide.
"""

from __future__ import annotations

from stockpredictor.backtest.metrics import hit_rate_by_decile
from stockpredictor.common.types import DataLayer
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN
from stockpredictor.prediction.registry import GOLD_DOMAIN as PREDICTIONS_DOMAIN
from stockpredictor.storage.lake import Lake


def compute_accuracy(lake: Lake, horizon: str) -> dict | None:
    """Returns None if there isn't yet enough resolved history to compute
    anything meaningful -- callers (API/UI) decide how to present that."""
    predictions = lake.read_all(DataLayer.GOLD, PREDICTIONS_DOMAIN)
    labels = lake.read_all(DataLayer.GOLD, LABELS_DOMAIN)
    if predictions.empty or labels.empty:
        return None

    predictions = predictions[predictions["horizon"] == horizon]
    labels = labels[labels["horizon"] == horizon]
    merged = predictions.merge(labels, on=["symbol", "date", "horizon"], how="inner")
    merged = merged.dropna(subset=["outperform"])
    if merged.empty:
        return None

    deciles = hit_rate_by_decile(merged["score"], merged["outperform"].astype(int))
    return {
        "horizon": horizon,
        "n_resolved_predictions": int(len(merged)),
        "hit_rate_by_score_decile": {int(k): float(v) for k, v in deciles.items()},
    }
