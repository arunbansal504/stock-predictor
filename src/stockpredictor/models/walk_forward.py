"""Walk-forward cross-validation with an embargo (§6, §22, §25).

Financial time series must never be split with a random/shuffled CV -- the
i.i.d. assumption is false and adjacent trading days are autocorrelated
(architecture doc Truth 2). This implements a simple, honest
expanding-window walk-forward scheme: each fold trains on all data up to a
cutoff and tests on the following window, then the cutoff advances.

An embargo is enforced between train and test (Lopez de Prado): training
rows whose label isn't yet *resolved* (`label_valid_date`, see
labels/returns.py) by the train cutoff are excluded. Without this, a 90-day
horizon label decided a week before the cutoff would leak ~83 days of
post-cutoff price action into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    dates: pd.DatetimeIndex | pd.Series,
    min_train_days: int,
    test_window_days: int,
    step_days: int,
) -> list[WalkForwardFold]:
    """Generate expanding-window fold boundaries over the sorted unique
    trading dates present in the panel. Pure date-boundary computation --
    does not filter rows itself, see `split` for that."""
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(dates))))
    folds: list[WalkForwardFold] = []
    fold_id = 0
    train_end_idx = min_train_days - 1
    while True:
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_window_days - 1
        if test_end_idx >= len(unique_dates):
            break
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=unique_dates[0],
                train_end=unique_dates[train_end_idx],
                test_start=unique_dates[test_start_idx],
                test_end=unique_dates[test_end_idx],
            )
        )
        fold_id += 1
        train_end_idx += step_days
    return folds


def split(
    df: pd.DataFrame,
    folds: list[WalkForwardFold],
    date_col: str = "date",
    label_valid_date_col: str = "label_valid_date",
) -> list[tuple[pd.Index, pd.Index]]:
    """For each fold, return (train_index, test_index) into `df`. Training
    rows are embargoed: excluded unless their label was already resolved
    (`label_valid_date_col <= train_end`) -- see module docstring. Test rows
    are not embargoed -- a test fold evaluates decisions made in that window
    regardless of when their labels later resolved (some may still be NaN if
    the window runs past available history; callers should drop those before
    scoring)."""
    out = []
    dates = pd.to_datetime(df[date_col])
    label_valid = pd.to_datetime(df[label_valid_date_col])
    for fold in folds:
        train_mask = (dates <= fold.train_end) & (label_valid <= fold.train_end)
        test_mask = (dates >= fold.test_start) & (dates <= fold.test_end)
        out.append((df.index[train_mask], df.index[test_mask]))
    return out
