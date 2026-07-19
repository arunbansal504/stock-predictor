"""Base learners + stacking ensemble for the ranking model (§8).

Phase 1 keeps exactly two base learners -- LightGBM (the tree-based
workhorse for tabular cross-sectional finance data) and a regularized
logistic regression (the "honesty baseline": if LightGBM can't beat this
out-of-sample, it's overfitting, not learning) -- not the full 20-model zoo
from the original brief. Every other model in §8's table is deferred until
it earns its place with an out-of-sample metric improvement (Truth 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stockpredictor.models.calibration import IsotonicCalibrator


def make_lightgbm_classifier(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        random_state=random_state,
        verbosity=-1,
        # A fixed random_state alone doesn't make LightGBM's multithreaded
        # histogram construction bit-reproducible -- gradient accumulation
        # order depends on thread scheduling, which can nudge split gains
        # and occasionally pick a different split. `deterministic=True`
        # (which requires forcing row- or col-wise histograms) fixes that
        # without pinning num_threads=1, so training stays parallel.
        deterministic=True,
        force_row_wise=True,
    )


def make_linear_baseline(random_state: int = 42) -> Pipeline:
    """LightGBM handles NaN/unscaled features natively; logistic regression
    does not, so median-impute + standardize are baked into this pipeline."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                # sklearn >=1.8 infers the penalty from l1_ratio/C rather than
                # an explicit `penalty=` string (deprecated); l1_ratio=0.5
                # alone is enough to specify elasticnet here.
                LogisticRegression(
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


class StackedRanker:
    """LightGBM + linear-baseline stack with a logistic meta-learner and
    isotonic calibration.

    Correctness note: this is a three-way chronological split, not two --
    base learners train on the earliest slice (`base_frac`); the next slice
    (`meta_frac`) is held out to generate out-of-fold-equivalent predictions
    used to fit the meta-learner; the final slice is held out *again* to fit
    the calibrator on the meta-learner's out-of-sample predictions. Fitting
    the calibrator on the meta-learner's own training rows (an earlier
    version of this class did exactly that) contradicts
    models/calibration.py's own documented requirement -- "never on the
    same rows a base model trained on" -- because the meta-learner IS a
    model, and predictions on its own training rows are in-sample for it
    even though they're out-of-sample for the base learners underneath.
    Nothing is refit on the full training set afterward -- doing so would
    create a train/predict distribution mismatch (the meta-learner would be
    scoring outputs from a differently-fit base model than the one it
    learned to combine, and the calibrator would be calibrating a
    differently-fit meta-learner), which is a leak just as real as a
    lookahead in the raw data. The trade-off is using less data for each
    stage; correctness wins over data efficiency here.
    """

    def __init__(self, random_state: int = 42, base_frac: float = 0.6, meta_frac: float = 0.2) -> None:
        self.random_state = random_state
        self.base_frac = base_frac
        self.meta_frac = meta_frac
        self.lgbm = make_lightgbm_classifier(random_state)
        self.linear = make_linear_baseline(random_state)
        self.meta = LogisticRegression(max_iter=1000, random_state=random_state)
        self.calibrator = IsotonicCalibrator()
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series, dates: pd.Series) -> "StackedRanker":
        order = np.argsort(pd.to_datetime(dates).to_numpy(), kind="stable")
        X_sorted = X.iloc[order].reset_index(drop=True)
        y_sorted = np.asarray(y).astype(int)[order]

        n = len(X_sorted)
        base_end = int(n * self.base_frac)
        meta_end = int(n * (self.base_frac + self.meta_frac))
        base_n, meta_n, calib_n = base_end, meta_end - base_end, n - meta_end
        if base_n < 10 or meta_n < 10 or calib_n < 10:
            raise ValueError(
                f"Not enough rows ({n}) to split into base/meta/calibration training "
                f"sets (got {base_n}/{meta_n}/{calib_n}) -- need at least ~10 rows in each."
            )

        X_base, y_base = X_sorted.iloc[:base_end], y_sorted[:base_end]
        X_meta, y_meta = X_sorted.iloc[base_end:meta_end], y_sorted[base_end:meta_end]
        X_calib, y_calib = X_sorted.iloc[meta_end:], y_sorted[meta_end:]

        self.lgbm.fit(X_base, y_base)
        self.linear.fit(X_base, y_base)

        meta_train_features = self._base_predictions(X_meta)
        self.meta.fit(meta_train_features, y_meta)

        calib_meta_scores = self.meta.predict_proba(self._base_predictions(X_calib))[:, 1]
        self.calibrator.fit(calib_meta_scores, y_calib)

        self._fitted = True
        return self

    def _base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        lgbm_scores = self.lgbm.predict_proba(X)[:, 1]
        linear_scores = self.linear.predict_proba(X)[:, 1]
        return np.column_stack([lgbm_scores, linear_scores])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of outperformance for each row."""
        if not self._fitted:
            raise RuntimeError("StackedRanker must be fit before predict_proba")
        return self.calibrator.transform(self.meta_score(X))

    def meta_score(self, X: pd.DataFrame) -> np.ndarray:
        """The meta-learner's own probability output, *before* isotonic
        calibration -- a continuous, finer-grained signal than
        `predict_proba`'s output.

        Isotonic calibration is a genuine step function by construction
        (Pool Adjacent Violators merges any region where the empirical
        win-rate isn't reliably monotonic into one flat block), and does
        exactly that in the sparse, noisy tail of a modest signal -- e.g.
        observed live, raw scores from ~0.49 to ~0.56 all collapsing onto
        one calibrated value, because there wasn't enough evidence to
        honestly distinguish them. That's calibration doing its job
        correctly, not a bug -- but it means `predict_proba`'s output alone
        is a poor *ranking* key: dozens of genuinely different stocks can
        land on the exact same calibrated score. `meta_score` stays
        continuous through that same region, so ranking/engine.py uses it
        to break ties meaningfully instead of falling back to arbitrary row
        order. The calibrated score remains what's shown to the user as
        the honest probability estimate -- this is only for sort order."""
        if not self._fitted:
            raise RuntimeError("StackedRanker must be fit before meta_score")
        meta_features = self._base_predictions(X)
        return self.meta.predict_proba(meta_features)[:, 1]

    def separation_info(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row calibration evidence backing `predict_proba`'s score --
        see `IsotonicCalibrator.separation_info` for what each column means.
        Lets callers show *why* a score should (or shouldn't) be trusted as
        more than a coin flip, instead of just the score itself."""
        if not self._fitted:
            raise RuntimeError("StackedRanker must be fit before separation_info")
        return self.calibrator.separation_info(self.meta_score(X))

    def disagreement(self, X: pd.DataFrame) -> np.ndarray:
        """Absolute difference between base learners' raw scores -- a cheap
        ensemble-disagreement signal (§6: confidence combines calibrated
        probability with "ensemble disagreement (variance across base
        learners)")."""
        preds = self._base_predictions(X)
        return np.abs(preds[:, 0] - preds[:, 1])
