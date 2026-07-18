"""SHAP-based factor attribution (§11 "basic" Phase 1 scope: SHAP -> factor
blocks -> positive/negative signal lists). The LLM narrative layer (§11's
second layer, RAG-grounded prose over these attributions) is Phase 2 --
explicitly out of scope here (§27).

SHAP is computed on the LightGBM base learner only, not the full calibrated
stack (meta-learner + isotonic calibration, see models/ensemble.py). This is
a documented simplification, not a hidden one: LightGBM is the dominant,
most complex base learner, and `shap.TreeExplainer` gives exact, fast
attribution for it. Explaining the full stack end-to-end would require a
model-agnostic explainer (e.g. KernelSHAP) that is both much slower and a
noisier approximation -- not worth it for Phase 1's "why did this stock rank
where it did" use case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier


def compute_shap_values(model: LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-feature SHAP contribution to the LightGBM base learner's
    output for the positive ("outperform") class. Returns a frame the same
    shape as X, indexed the same way."""
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    # shap's return shape has varied across versions for binary classifiers:
    # a list of two (class0, class1) arrays, or a single 3D (rows, features,
    # classes) array. Normalize to the positive-class 2D contribution either way.
    if isinstance(raw, list):
        values = raw[1]
    elif np.ndim(raw) == 3:
        values = raw[:, :, 1]
    else:
        values = raw
    return pd.DataFrame(values, columns=X.columns, index=X.index)
