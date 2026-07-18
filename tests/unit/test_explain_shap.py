from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from stockpredictor.explain.shap_explainer import compute_shap_values


def _fitted_model_and_X():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "signal": rng.normal(0, 1, n),
            "noise": rng.normal(0, 1, n),
        }
    )
    y = (X["signal"] > 0).astype(int)
    model = LGBMClassifier(n_estimators=50, max_depth=3, verbosity=-1, random_state=0)
    model.fit(X, y)
    return model, X


def test_compute_shap_values_shape_matches_input():
    model, X = _fitted_model_and_X()
    shap_df = compute_shap_values(model, X)
    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == list(X.columns)
    assert list(shap_df.index) == list(X.index)


def test_informative_feature_has_larger_mean_abs_shap_than_noise():
    model, X = _fitted_model_and_X()
    shap_df = compute_shap_values(model, X)
    assert shap_df["signal"].abs().mean() > shap_df["noise"].abs().mean()


def test_shap_sign_matches_direction_of_influence():
    model, X = _fitted_model_and_X()
    shap_df = compute_shap_values(model, X)
    # Rows with high `signal` (which drives the positive class) should have,
    # on average, positive SHAP contribution from that feature.
    high = shap_df.loc[X["signal"] > 1, "signal"]
    low = shap_df.loc[X["signal"] < -1, "signal"]
    assert high.mean() > low.mean()
