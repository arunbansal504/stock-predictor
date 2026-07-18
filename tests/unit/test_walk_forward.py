from __future__ import annotations

import pandas as pd

from stockpredictor.models.walk_forward import generate_folds, split


def test_generate_folds_produces_expected_boundaries():
    dates = pd.bdate_range("2024-01-01", periods=30)
    folds = generate_folds(dates, min_train_days=10, test_window_days=5, step_days=5)

    assert len(folds) > 0
    first = folds[0]
    assert first.train_end == dates[9]  # 0-indexed: 10th day
    assert first.test_start == dates[10]
    assert first.test_end == dates[14]


def test_generate_folds_stops_before_running_out_of_data():
    dates = pd.bdate_range("2024-01-01", periods=20)
    folds = generate_folds(dates, min_train_days=10, test_window_days=5, step_days=5)
    for fold in folds:
        assert fold.test_end <= dates[-1]


def test_generate_folds_empty_when_not_enough_history():
    dates = pd.bdate_range("2024-01-01", periods=5)
    folds = generate_folds(dates, min_train_days=10, test_window_days=5, step_days=5)
    assert folds == []


def test_split_embargoes_unresolved_labels_out_of_training():
    dates = pd.bdate_range("2024-01-01", periods=20)
    # label_valid_date is 10 trading days after date -- a 10d-horizon label.
    df = pd.DataFrame(
        {
            "date": dates,
            "label_valid_date": dates.shift(10, freq="B"),
        }
    )
    folds = generate_folds(dates, min_train_days=15, test_window_days=1, step_days=1)
    assert len(folds) >= 1
    fold = folds[0]

    train_idx, test_idx = split(df, [fold])[0]
    train_rows = df.loc[train_idx]
    # Every training row's label must have resolved by the train cutoff.
    assert (train_rows["label_valid_date"] <= fold.train_end).all()
    # And some rows with date <= train_end were excluded precisely because
    # their label wasn't resolved yet -- the embargo actually did something.
    naive_would_include = df[df["date"] <= fold.train_end]
    assert len(train_rows) < len(naive_would_include)


def test_split_test_mask_covers_the_test_window_regardless_of_label_resolution():
    dates = pd.bdate_range("2024-01-01", periods=20)
    df = pd.DataFrame({"date": dates, "label_valid_date": dates.shift(10, freq="B")})
    folds = generate_folds(dates, min_train_days=15, test_window_days=2, step_days=2)
    fold = folds[0]

    _, test_idx = split(df, [fold])[0]
    test_rows = df.loc[test_idx]
    assert (test_rows["date"] >= fold.test_start).all()
    assert (test_rows["date"] <= fold.test_end).all()
