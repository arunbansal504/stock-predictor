from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.common.types import DataLayer
from stockpredictor.features.registry import ALL_FEATURE_COLUMNS, GOLD_DOMAIN as FEATURES_DOMAIN
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN
from stockpredictor.prediction import predict


def _seed_features(tmp_lake, symbol: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    data = {"symbol": [symbol] * n, "date": dates}
    for col in ALL_FEATURE_COLUMNS:
        raw = rng.normal(0, 1, n)
        data[col] = raw
        data[f"{col}_xrank"] = pd.Series(raw).rank(pct=True)
    tmp_lake.write(pd.DataFrame(data), DataLayer.GOLD, FEATURES_DOMAIN, symbol, key_cols=["symbol", "date"])


def _seed_labels(tmp_lake, symbol: str, n: int, seed: int, horizon: str = "5d") -> None:
    rng = np.random.default_rng(seed + 100)
    dates = pd.bdate_range("2023-01-01", periods=n)
    outperform = pd.array(rng.uniform(0, 1, n) < 0.5, dtype="boolean")
    df = pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "horizon": [horizon] * n,
            "excess_return": rng.normal(0, 0.02, n),
            "outperform": outperform,
            "label_valid_date": dates,
        }
    )
    tmp_lake.write(df, DataLayer.GOLD, LABELS_DOMAIN, symbol, key_cols=["symbol", "date", "horizon"])


def test_get_latest_feature_snapshot_picks_most_recent_row_per_symbol(tmp_lake):
    _seed_features(tmp_lake, "AAA", n=10, seed=1)
    _seed_features(tmp_lake, "BBB", n=15, seed=2)

    snapshot = predict.get_latest_feature_snapshot(tmp_lake)
    assert len(snapshot) == 2
    aaa_row = snapshot[snapshot["symbol"] == "AAA"].iloc[0]
    assert aaa_row["date"] == pd.bdate_range("2023-01-01", periods=10)[-1]


def test_get_latest_feature_snapshot_empty_when_no_features(tmp_lake):
    assert predict.get_latest_feature_snapshot(tmp_lake).empty


def test_train_production_model_raises_without_training_data(tmp_lake):
    with pytest.raises(ValueError, match="No training data"):
        predict.train_production_model(tmp_lake, "5d")


def test_score_universe_end_to_end(tmp_lake):
    for symbol, seed in [("AAA", 1), ("BBB", 2), ("CCC", 3)]:
        _seed_features(tmp_lake, symbol, n=200, seed=seed)
        _seed_labels(tmp_lake, symbol, n=200, seed=seed, horizon="5d")

    scored = predict.score_universe(tmp_lake, "5d", random_state=42)
    assert set(scored["symbol"]) == {"AAA", "BBB", "CCC"}
    assert (scored["score"] >= 0).all() and (scored["score"] <= 1).all()
    assert (scored["disagreement"] >= 0).all()
    assert (scored["horizon"] == "5d").all()
