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


def test_meta_score_before_fit_raises():
    X, _, _ = _synthetic_dataset(n=50)
    model = StackedRanker()
    with pytest.raises(RuntimeError, match="must be fit"):
        model.meta_score(X)


def test_meta_score_is_bounded_probability():
    X, y, dates = _synthetic_dataset(n=300, seed=4)
    model = StackedRanker(random_state=42)
    model.fit(X, y, dates)

    meta_score = model.meta_score(X)
    assert meta_score.shape == (len(X),)
    assert (meta_score >= 0).all() and (meta_score <= 1).all()


def test_predict_proba_equals_calibrator_applied_to_meta_score():
    """predict_proba must be exactly the calibrated transform of meta_score
    -- these should never silently diverge into two different computations
    (see nightly_flow.py/predict.py, which call calibrator.transform(
    meta_score) directly to avoid redoing base-learner inference)."""
    X, y, dates = _synthetic_dataset(n=300, seed=5)
    model = StackedRanker(random_state=42)
    model.fit(X, y, dates)

    proba = model.predict_proba(X)
    via_meta_score = model.calibrator.transform(model.meta_score(X))
    np.testing.assert_allclose(proba, via_meta_score, atol=1e-12)


def test_separation_info_before_fit_raises():
    X, _, _ = _synthetic_dataset(n=50)
    model = StackedRanker()
    with pytest.raises(RuntimeError, match="must be fit"):
        model.separation_info(X)


def test_separation_info_is_calibrator_separation_info_of_meta_score():
    """StackedRanker.separation_info must be a thin pass-through to
    calibrator.separation_info(meta_score(X)) -- see IsotonicCalibrator's
    own tests (test_calibration.py) for what the flag itself means."""
    X, y, dates = _synthetic_dataset(n=300, seed=9)
    model = StackedRanker(random_state=42)
    model.fit(X, y, dates)

    via_model = model.separation_info(X)
    via_calibrator = model.calibrator.separation_info(model.meta_score(X))
    pd.testing.assert_frame_equal(via_model.reset_index(drop=True), via_calibrator.reset_index(drop=True))
    assert set(via_model.columns) == {"n", "empirical_rate", "p_value", "separation_direction", "base_rate"}
    assert len(via_model) == len(X)


def test_meta_score_stays_differentiated_when_calibrated_score_collapses():
    """The whole reason meta_score exists (§ranking/engine.py): isotonic
    calibration is a genuine step function and can legitimately collapse
    many distinct inputs onto one output value in a sparse/noisy region.
    When that happens, meta_score must still distinguish those rows -- this
    reproduces the collapse with a deliberately tiny, noisy calibration set
    where isotonic regression has almost no evidence to work with."""
    rng = np.random.default_rng(6)
    n = 200
    dates = pd.bdate_range("2022-01-01", periods=n)
    # A very weak, noisy signal -- deliberately hard for isotonic
    # calibration to resolve finely, mirroring the live IC ~0.0165 case.
    signal = rng.normal(0, 0.05, n)
    X = pd.DataFrame({"signal": signal, "noise1": rng.normal(0, 1, n), "noise2": rng.normal(0, 1, n)})
    prob = 1 / (1 + np.exp(-signal))
    y = (rng.uniform(0, 1, n) < prob).astype(int)

    model = StackedRanker(random_state=42)
    model.fit(X, y, pd.Series(dates))

    calibrated = model.predict_proba(X)
    meta = model.meta_score(X)

    # Even if calibration collapsed many rows onto few distinct values,
    # meta_score must retain much finer resolution -- the whole point.
    assert len(np.unique(meta)) >= len(np.unique(calibrated))


def test_meta_score_variance_regression_guard():
    """Regression guard for the live bug this module's meta_score exists to
    fix: every stock in the ranked output showing an identical score to
    many decimal places. Runs a realistic-scale (80 "stocks"), deliberately
    weak/noisy signal (mirroring the live IC ~0.0165) through the full
    fit/score path and asserts meta_score's variance clears a small
    epsilon -- if a future change collapses meta_score the way isotonic
    calibration legitimately collapses `score`, ranking would silently
    degrade back to arbitrary row order again."""
    rng = np.random.default_rng(11)
    n = 800  # 800 decision-rows across ~80 distinct symbols' worth of history
    dates = pd.bdate_range("2022-01-01", periods=n)
    signal = rng.normal(0, 0.05, n)  # weak, like the real edge
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise1": rng.normal(0, 1, n),
            "noise2": rng.normal(0, 1, n),
            "noise3": rng.normal(0, 1, n),
        }
    )
    prob = 1 / (1 + np.exp(-signal))
    y = (rng.uniform(0, 1, n) < prob).astype(int)

    model = StackedRanker(random_state=42)
    model.fit(X, y, pd.Series(dates))

    # Score a diverse batch of 80 distinct "stocks" (rows), as a live
    # ranking run would.
    batch = X.iloc[:80].reset_index(drop=True)
    meta_score = model.meta_score(batch)

    EPSILON = 1e-6
    assert np.var(meta_score) > EPSILON, (
        f"meta_score variance {np.var(meta_score)} did not clear epsilon {EPSILON} -- "
        "ranking would silently degrade to arbitrary row-order tie-breaking again"
    )
    assert len(np.unique(meta_score)) > 1


def _weak_signal_dataset(n: int, seed: int, signal_scale: float = 0.05):
    """Mirrors the live IC~0.0165 case -- weak, noisy, and genuinely
    supports very little fine-grained differentiation once calibrated.
    Used by the monotonicity and full-population variance tests, which
    should hold regardless of how coarse a *legitimately* honest
    calibration turns out to be."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    signal = rng.normal(0, signal_scale, n)
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise1": rng.normal(0, 1, n),
            "noise2": rng.normal(0, 1, n),
            "noise3": rng.normal(0, 1, n),
        }
    )
    prob = 1 / (1 + np.exp(-signal))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return X, y, pd.Series(dates)


def test_score_is_monotonic_nondecreasing_in_meta_score():
    """Structural invariant, not a statistical one -- isotonic regression
    is monotonic by construction, so `score` can never rank two rows in
    the opposite order from `meta_score`/relative_strength. This is what
    rules out a wrong/separate model head or a wiring bug silently
    swapping in a different signal for `score`: if score ever contradicts
    meta_score's ordering, something upstream of calibration is broken,
    regardless of how coarse the calibrated output legitimately is."""
    X, y, dates = _weak_signal_dataset(n=600, seed=20)
    model = StackedRanker(random_state=42)
    model.fit(X, y, dates)

    meta_score = model.meta_score(X)
    score = model.predict_proba(X)

    order = np.argsort(meta_score)  # ascending
    score_in_meta_order = score[order]
    # Non-decreasing: each step is >= the previous, within float tolerance.
    diffs = np.diff(score_in_meta_order)
    assert (diffs >= -1e-9).all(), "score disagreed in direction with meta_score somewhere -- possible wrong/separate head"


def test_score_has_nontrivial_variance_across_the_full_population():
    """`score` collapsing within a tied sub-group (e.g. the top of a
    ranking) is legitimate honest calibration -- but `score` collapsing
    across the ENTIRE population would mean the model produces no usable
    signal at all, which is a real regression, not honest calibration.
    Samples across the full range of meta_score (not an arbitrary top-N
    slice) and asserts the calibrated score still shows real variance."""
    X, y, dates = _weak_signal_dataset(n=1000, seed=21)
    model = StackedRanker(random_state=42)
    model.fit(X, y, dates)

    batch = X.iloc[:200].reset_index(drop=True)
    meta_score = model.meta_score(batch)
    score = model.predict_proba(batch)

    # Sample spread evenly across the meta_score distribution, not just
    # the top -- mirrors how the live investigation deliberately sampled
    # across the full range rather than only the tied top-N.
    order = np.argsort(meta_score)
    spread_idx = order[np.linspace(0, len(order) - 1, 30, dtype=int)]

    EPSILON = 1e-6
    spread_variance = np.var(score[spread_idx])
    assert spread_variance > EPSILON, (
        f"score variance across a population-spanning sample was {spread_variance}, "
        f"below epsilon {EPSILON} -- the model may be producing no usable signal at all"
    )


def test_top_n_score_does_not_collapse_when_the_underlying_signal_supports_separation():
    """Guards against calibration becoming *more* coarse than the data
    actually warrants at the top end -- distinct from the monotonicity and
    full-population tests, which both pass even if the top-N block is
    (legitimately) one flat value. This test uses a deliberately STRONGER,
    well-separated top-end signal (unlike the weak-signal fixture used
    elsewhere) where the top-ranked rows genuinely differ in outcome
    likelihood -- a correctly-behaving isotonic calibration should be able
    to preserve some of that real separation. Run across several
    independent seeds ("a large random sample of runs") rather than one,
    since any single fit can have an unlucky sample; only a majority
    collapsing to near-zero top-N variance indicates a real regression."""
    n_collapsed = 0
    n_seeds = 8
    for seed in range(n_seeds):
        rng = np.random.default_rng(100 + seed)
        n = 1000
        dates = pd.bdate_range("2020-01-01", periods=n)
        # A much stronger, more separable signal than the live weak-IC
        # case -- deliberately so the top end has real information for
        # calibration to preserve, unlike the other tests in this file.
        signal = rng.normal(0, 0.6, n)
        X = pd.DataFrame(
            {
                "signal": signal,
                "noise1": rng.normal(0, 1, n),
                "noise2": rng.normal(0, 1, n),
            }
        )
        prob = 1 / (1 + np.exp(-signal))
        y = (rng.uniform(0, 1, n) < prob).astype(int)

        model = StackedRanker(random_state=42)
        model.fit(X, y, pd.Series(dates))

        batch = X.iloc[:100].reset_index(drop=True)
        meta_score = model.meta_score(batch)
        score = model.predict_proba(batch)

        order = np.argsort(meta_score)[::-1]  # descending -- best first
        for top_n in (10, 20):
            top_slice = score[order[:top_n]]
            if np.var(top_slice) <= 1e-6:
                n_collapsed += 1
                break  # count this seed once, not once per N

    assert n_collapsed <= n_seeds // 2, (
        f"{n_collapsed}/{n_seeds} runs collapsed to near-zero score variance in the "
        "top-10/top-20 slice despite a deliberately strong, separable signal -- "
        "calibration may have become more aggressive/coarse than the data warrants"
    )


def test_fit_is_exactly_reproducible_across_independent_fits():
    """Same inputs, same random_state, fit twice independently -- must
    produce bit-identical output. This is the model-level regression guard
    for the "different ranks every run" bug: LightGBM's default
    multithreaded histogram construction is not bit-reproducible on its own
    even with a fixed random_state (gradient-accumulation order depends on
    thread scheduling), which is exactly what make_lightgbm_classifier's
    `deterministic=True`/`force_row_wise=True` exist to fix."""
    X, y, dates = _synthetic_dataset(n=400, seed=8)

    model_a = StackedRanker(random_state=42)
    model_a.fit(X, y, dates)

    model_b = StackedRanker(random_state=42)
    model_b.fit(X, y, dates)

    assert np.array_equal(model_a.predict_proba(X), model_b.predict_proba(X))
    assert np.array_equal(model_a.meta_score(X), model_b.meta_score(X))


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
