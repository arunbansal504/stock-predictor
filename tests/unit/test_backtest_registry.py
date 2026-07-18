from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.backtest.engine import BacktestResult
from stockpredictor.backtest.registry import (
    persist_backtest_result,
    read_backtest_results,
    read_latest_backtest_result,
    read_latest_return_calibration,
)


def _result() -> BacktestResult:
    idx = pd.Index(["d1", "d2"], name="date")
    return BacktestResult(
        per_period_returns=pd.Series([0.02, 0.01], index=idx),
        benchmark_returns=pd.Series([0.01, 0.005], index=idx),
        ic_by_date=pd.Series([0.1, 0.2], index=idx),
        metrics={"cagr": 0.15, "sharpe": 1.2, "n_periods": 2},
        benchmark_metrics={"cagr": 0.08, "sharpe": 0.6, "n_periods": 2},
    )


def test_persist_and_read_backtest_result_roundtrip(tmp_lake):
    rows = persist_backtest_result(tmp_lake, _result(), horizon="5d", strategy_id="top_k_v1", run_date=pd.Timestamp("2024-01-01"))
    assert rows == 1

    out = read_backtest_results(tmp_lake, "top_k_v1")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["strategy_cagr"] == 0.15
    assert row["benchmark_cagr"] == 0.08
    assert row["mean_ic"] == pytest.approx(0.15)
    assert isinstance(row["equity_curve"], list)
    assert row["equity_curve"][0]["strategy_return"] == 0.02


def test_read_latest_backtest_result_picks_most_recent_run(tmp_lake):
    persist_backtest_result(tmp_lake, _result(), horizon="5d", strategy_id="top_k_v1", run_date=pd.Timestamp("2024-01-01"))
    persist_backtest_result(tmp_lake, _result(), horizon="5d", strategy_id="top_k_v1", run_date=pd.Timestamp("2024-02-01"))

    latest = read_latest_backtest_result(tmp_lake, "top_k_v1", "5d")
    assert latest is not None
    assert pd.Timestamp(latest["run_date"]) == pd.Timestamp("2024-02-01")


def test_read_latest_backtest_result_none_when_missing(tmp_lake):
    assert read_latest_backtest_result(tmp_lake, "nope", "5d") is None


def test_read_latest_backtest_result_none_for_wrong_horizon(tmp_lake):
    persist_backtest_result(tmp_lake, _result(), horizon="5d", strategy_id="top_k_v1", run_date=pd.Timestamp("2024-01-01"))
    assert read_latest_backtest_result(tmp_lake, "top_k_v1", "30d") is None


def test_persist_and_read_return_calibration_roundtrip(tmp_lake):
    calibration = pd.DataFrame(
        {"decile": [0, 1], "score_min": [0.0, 0.5], "score_max": [0.49, 1.0], "mean_return": [0.01, 0.05], "median_return": [0.01, 0.05], "n_obs": [10, 10]}
    )
    persist_backtest_result(
        tmp_lake, _result(), horizon="5d", strategy_id="top_k_v1",
        run_date=pd.Timestamp("2024-01-01"), return_calibration=calibration,
    )

    out = read_latest_return_calibration(tmp_lake, "top_k_v1", "5d")
    assert len(out) == 2
    assert out.iloc[1]["mean_return"] == pytest.approx(0.05)


def test_read_latest_return_calibration_empty_when_no_backtest(tmp_lake):
    out = read_latest_return_calibration(tmp_lake, "nope", "5d")
    assert out.empty


def test_read_latest_return_calibration_empty_when_not_provided(tmp_lake):
    persist_backtest_result(tmp_lake, _result(), horizon="5d", strategy_id="top_k_v1", run_date=pd.Timestamp("2024-01-01"))
    out = read_latest_return_calibration(tmp_lake, "top_k_v1", "5d")
    assert out.empty
