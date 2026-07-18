from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.explain.signals import explain_predictions, summarize_factor_blocks, top_signals
from stockpredictor.models.ensemble import StackedRanker


def test_summarize_factor_blocks_sums_within_block():
    shap_row = pd.Series(
        {
            "return_5d": 0.10,  # Momentum/Trend
            "return_20d": 0.05,  # Momentum/Trend
            "rsi_14": -0.02,  # Oscillators
            "obv": 0.01,  # Volume/Liquidity
        }
    )
    out = summarize_factor_blocks(shap_row)
    assert out["Momentum/Trend"] == pytest.approx(0.15)
    assert out["Oscillators"] == pytest.approx(-0.02)
    assert out["Volume/Liquidity"] == pytest.approx(0.01)
    assert out.index[0] == "Momentum/Trend"  # sorted most-positive first


def test_top_signals_separates_positive_and_negative_correctly():
    shap_row = pd.Series({"return_5d": 0.20, "rsi_14": 0.05, "obv": -0.01, "atr_14": -0.15})
    out = top_signals(shap_row, n=5)
    positive_features = {s["feature"] for s in out["positive"]}
    negative_features = {s["feature"] for s in out["negative"]}
    assert positive_features == {"return_5d", "rsi_14"}
    assert negative_features == {"obv", "atr_14"}
    # Most negative first in the negative list.
    assert out["negative"][0]["feature"] == "atr_14"


def test_top_signals_respects_n_limit():
    shap_row = pd.Series({f"f{i}": float(i) for i in range(1, 11)})  # all positive, 10 features
    out = top_signals(shap_row, n=3)
    assert len(out["positive"]) == 3
    assert len(out["negative"]) == 0


def test_top_signals_includes_block_tag():
    shap_row = pd.Series({"return_5d": 0.1})
    out = top_signals(shap_row, n=5)
    assert out["positive"][0]["block"] == "Momentum/Trend"


def test_explain_predictions_end_to_end_produces_one_row_per_symbol():
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.bdate_range("2022-01-01", periods=n)
    X = pd.DataFrame(
        {
            "return_5d": rng.normal(0, 1, n),
            "rsi_14": rng.normal(0, 1, n),
            "obv": rng.normal(0, 1, n),
        }
    )
    y = (X["return_5d"] > 0).astype(int)
    model = StackedRanker(random_state=0)
    model.fit(X, y, pd.Series(dates))

    symbols = pd.Series([f"SYM{i}" for i in range(n)])
    out = explain_predictions(model, X, symbols, n_signals=2)

    assert len(out) == n
    assert set(out.columns) == {"symbol", "factor_blocks", "top_positive_signals", "top_negative_signals"}
    first = out.iloc[0]
    assert isinstance(first["factor_blocks"], dict)
    assert isinstance(first["top_positive_signals"], list)


def test_explain_predictions_empty_input_returns_empty_frame():
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.bdate_range("2022-01-01", periods=n)
    X = pd.DataFrame({"return_5d": rng.normal(0, 1, n)})
    y = (X["return_5d"] > 0).astype(int)
    model = StackedRanker(random_state=0)
    model.fit(X, y, pd.Series(dates))

    out = explain_predictions(model, pd.DataFrame(columns=["return_5d"]), pd.Series(dtype="object"))
    assert out.empty
