from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.models.ensemble import StackedRanker


def _synthetic_dataset(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    signal = rng.normal(0, 1, n)
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.normal(0, 1, n)
    X = pd.DataFrame({"signal": signal, "noise1": noise1, "noise2": noise2})
    # y correlated with `signal` through a logistic link -- a genuinely
    # learnable (not deterministic) relationship, like real market data.
    prob = 1 / (1 + np.exp(-signal))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return X, y, pd.Series(dates)


def test_stacked_ranker_fit_predict_end_to_end():
    X, y, dates = _synthetic_dataset()
    model = StackedRanker(random_state=42)
    model.fit(X, y, dates)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_stacked_ranker_learns_the_real_signal_direction():
    X, y, dates = _synthetic_dataset(n=600, seed=1)
    model = StackedRanker(random_state=42)
    model.fit(X, y, dates)

    proba = model.predict_proba(X)
    # Rows with high `signal` should on average get higher probability than
    # rows with low `signal` -- a directional sanity check, not an exact value.
    high_signal_proba = proba[X["signal"] > 1].mean()
    low_signal_proba = proba[X["signal"] < -1].mean()
    assert high_signal_proba > low_signal_proba


def test_predict_proba_before_fit_raises():
    X, _, _ = _synthetic_dataset(n=50)
    model = StackedRanker()
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict_proba(X)


def test_fit_raises_with_too_few_rows():
    X, y, dates = _synthetic_dataset(n=15)
    model = StackedRanker()
    with pytest.raises(ValueError, match="Not enough rows"):
        model.fit(X, y, dates)


def test_disagreement_is_nonnegative_and_bounded():
    X, y, dates = _synthetic_dataset(n=300, seed=2)
    model = StackedRanker(random_state=42)
    model.fit(X, y, dates)

    disagreement = model.disagreement(X)
    assert (disagreement >= 0).all()
    assert (disagreement <= 1).all()


def test_fit_uses_only_chronologically_earlier_rows_for_base_learners():
    """Correctness-critical: shuffling the row order passed to fit() must not
    change which rows the base learners see -- `dates` alone determines the
    base/meta split, not input row order."""
    X, y, dates = _synthetic_dataset(n=300, seed=3)

    model_a = StackedRanker(random_state=42)
    model_a.fit(X, y, dates)

    shuffle = np.random.default_rng(9).permutation(len(X))
    model_b = StackedRanker(random_state=42)
    model_b.fit(X.iloc[shuffle].reset_index(drop=True), y[shuffle], dates.iloc[shuffle].reset_index(drop=True))

    proba_a = model_a.predict_proba(X)
    proba_b = model_b.predict_proba(X)
    np.testing.assert_allclose(proba_a, proba_b, atol=1e-8)
