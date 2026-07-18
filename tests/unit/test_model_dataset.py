from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.models import dataset
from stockpredictor.features.registry import GOLD_DOMAIN as FEATURES_DOMAIN, TECHNICAL_FEATURE_COLUMNS
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN


def _seed_features(tmp_lake, symbol: str, n: int = 10) -> None:
    dates = pd.bdate_range("2024-01-01", periods=n)
    data = {"symbol": [symbol] * n, "date": dates}
    for col in TECHNICAL_FEATURE_COLUMNS:
        data[col] = np.linspace(0, 1, n)
        data[f"{col}_xrank"] = np.linspace(0, 1, n)
    df = pd.DataFrame(data)
    tmp_lake.write(df, DataLayer.GOLD, FEATURES_DOMAIN, symbol, key_cols=["symbol", "date"])


def _seed_labels(tmp_lake, symbol: str, n: int = 10, horizon: str = "5d", n_unresolved: int = 2) -> None:
    dates = pd.bdate_range("2024-01-01", periods=n)
    outperform = [True, False] * (n // 2)
    outperform = pd.array(outperform, dtype="boolean")
    if n_unresolved:
        outperform[-n_unresolved:] = pd.NA
    df = pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "horizon": [horizon] * n,
            "excess_return": np.linspace(-0.05, 0.05, n),
            "outperform": outperform,
            "label_valid_date": dates,
        }
    )
    tmp_lake.write(df, DataLayer.GOLD, LABELS_DOMAIN, symbol, key_cols=["symbol", "date", "horizon"])


def test_build_training_frame_empty_without_features_or_labels(tmp_lake):
    assert dataset.build_training_frame(tmp_lake, "5d").empty
    _seed_features(tmp_lake, "AAA")
    assert dataset.build_training_frame(tmp_lake, "5d").empty  # labels still missing


def test_build_training_frame_joins_and_drops_unresolved_rows(tmp_lake):
    _seed_features(tmp_lake, "AAA", n=10)
    _seed_labels(tmp_lake, "AAA", n=10, horizon="5d", n_unresolved=2)

    out = dataset.build_training_frame(tmp_lake, "5d")
    assert len(out) == 8  # 10 - 2 unresolved
    for col in dataset.get_feature_columns():
        assert col in out.columns
    assert "outperform" in out.columns


def test_build_training_frame_filters_by_requested_horizon(tmp_lake):
    _seed_features(tmp_lake, "AAA", n=10)
    _seed_labels(tmp_lake, "AAA", n=10, horizon="5d", n_unresolved=0)
    _seed_labels(tmp_lake, "AAA", n=10, horizon="30d", n_unresolved=0)

    out5 = dataset.build_training_frame(tmp_lake, "5d")
    out30 = dataset.build_training_frame(tmp_lake, "30d")
    assert (out5["horizon"] == "5d").all()
    assert (out30["horizon"] == "30d").all()


def test_get_feature_columns_cross_sectional_vs_raw():
    xrank_cols = dataset.get_feature_columns(use_cross_sectional=True)
    raw_cols = dataset.get_feature_columns(use_cross_sectional=False)
    assert all(c.endswith("_xrank") for c in xrank_cols)
    assert raw_cols == list(TECHNICAL_FEATURE_COLUMNS)
