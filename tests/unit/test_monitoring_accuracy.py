from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN
from stockpredictor.monitoring.accuracy import compute_accuracy
from stockpredictor.prediction.registry import persist_predictions


def test_compute_accuracy_none_without_predictions_or_labels(tmp_lake):
    assert compute_accuracy(tmp_lake, "5d") is None


def test_compute_accuracy_returns_decile_hit_rates(tmp_lake):
    n = 100
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, n)
    outperform = pd.array(scores > 0.5, dtype="boolean")

    persist_predictions(
        tmp_lake,
        pd.DataFrame(
            {"symbol": ["AAA"] * n, "date": dates, "horizon": ["5d"] * n, "score": scores, "disagreement": 0.1}
        ),
    )
    labels = pd.DataFrame(
        {
            "symbol": ["AAA"] * n,
            "date": dates,
            "horizon": ["5d"] * n,
            "excess_return": rng.normal(0, 0.01, n),
            "outperform": outperform,
            "label_valid_date": dates,
        }
    )
    tmp_lake.write(labels, DataLayer.GOLD, LABELS_DOMAIN, "AAA", key_cols=["symbol", "date", "horizon"])

    result = compute_accuracy(tmp_lake, "5d")
    assert result is not None
    assert result["n_resolved_predictions"] == n
    assert result["hit_rate_by_score_decile"][9] > result["hit_rate_by_score_decile"][0]


def test_compute_accuracy_none_when_horizon_has_no_resolved_labels(tmp_lake):
    n = 10
    dates = pd.bdate_range("2024-01-01", periods=n)
    persist_predictions(
        tmp_lake,
        pd.DataFrame(
            {"symbol": ["AAA"] * n, "date": dates, "horizon": ["30d"] * n, "score": [0.5] * n, "disagreement": 0.1}
        ),
    )
    labels = pd.DataFrame(
        {
            "symbol": ["AAA"] * n,
            "date": dates,
            "horizon": ["5d"] * n,  # different horizon than the predictions
            "excess_return": [0.01] * n,
            "outperform": pd.array([True] * n, dtype="boolean"),
            "label_valid_date": dates,
        }
    )
    tmp_lake.write(labels, DataLayer.GOLD, LABELS_DOMAIN, "AAA", key_cols=["symbol", "date", "horizon"])

    assert compute_accuracy(tmp_lake, "30d") is None
