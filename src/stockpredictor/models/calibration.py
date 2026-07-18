"""Probability calibration (§6, §8): converts a classifier's raw scores into
honest probabilities via isotonic regression.

Must be fit on out-of-sample predictions -- never on the same rows a base
model trained on, or the calibration curve inherits the base model's
in-sample overconfidence and the whole point (§30: "reliability curve shows
predicted probabilities ~= realized frequencies") is lost.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(raw_scores, y_true)
        self._fitted = True
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fit before transform")
        return self._iso.predict(raw_scores)

    def fit_transform(self, raw_scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(raw_scores, y_true).transform(raw_scores)
