"""Persistence for backtest results (§13 `backtests` table, §15 "Backtest
Lab" screen).

A `BacktestResult` (backtest/engine.py) is flattened into one summary row
per (strategy_id, horizon, run_date): scalar metrics as columns, and the
per-period equity/benchmark curves JSON-serialized (same rationale as
explain/registry.py -- variable-length series don't belong as raw Parquet
columns across many small per-write files).
"""

from __future__ import annotations

import json

import pandas as pd

from stockpredictor.backtest.engine import BacktestResult
from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

GOLD_DOMAIN = "backtests"
GOLD_KEY_COLS = ["strategy_id", "horizon", "run_date"]


def persist_backtest_result(
    lake: Lake,
    result: BacktestResult,
    horizon: str,
    strategy_id: str,
    run_date: pd.Timestamp | None = None,
) -> int:
    run_date = run_date or pd.Timestamp.today().normalize()

    curve = pd.DataFrame(
        {
            "date": result.per_period_returns.index.astype(str),
            "strategy_return": result.per_period_returns.values,
            "benchmark_return": result.benchmark_returns.reindex(result.per_period_returns.index).values,
        }
    )

    row = {
        "strategy_id": strategy_id,
        "horizon": horizon,
        "run_date": run_date,
        **{f"strategy_{k}": v for k, v in result.metrics.items()},
        **{f"benchmark_{k}": v for k, v in result.benchmark_metrics.items()},
        "mean_ic": float(result.ic_by_date.mean()) if not result.ic_by_date.empty else float("nan"),
        "equity_curve": json.dumps(curve.to_dict(orient="records")),
    }
    df = pd.DataFrame([row])
    rows = lake.write(df, DataLayer.GOLD, GOLD_DOMAIN, strategy_id, key_cols=GOLD_KEY_COLS)
    logger.info("Persisted backtest result: strategy=%s horizon=%s", strategy_id, horizon)
    return rows


def read_backtest_results(lake: Lake, strategy_id: str) -> pd.DataFrame:
    df = lake.read(DataLayer.GOLD, GOLD_DOMAIN, strategy_id)
    if df.empty:
        return df
    df = df.copy()
    df["equity_curve"] = df["equity_curve"].apply(json.loads)
    return df


def read_latest_backtest_result(lake: Lake, strategy_id: str, horizon: str) -> dict | None:
    df = read_backtest_results(lake, strategy_id)
    if df.empty:
        return None
    df = df[df["horizon"] == horizon]
    if df.empty:
        return None
    latest = df.sort_values("run_date").iloc[-1]
    return latest.to_dict()
