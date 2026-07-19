"""Daily validation: resolves published predictions whose horizon has
completed and stores actual outcomes + risk metrics (ML Review Board spec
Part 2).

Reuses the Gold `labels` domain (labels/registry.py) for the resolution
date and the stock-vs-benchmark forward return, rather than re-deriving
"N trading days after prediction_date" from raw prices -- `label_valid_date`
and `excess_return` there already are exactly this, computed the identical
way the model's own training labels are. Daily price paths are still needed
separately for the intra-holding-period risk metrics (drawdown/run-up/
volatility/Sharpe/information ratio), which a single point-to-point forward
return can't provide.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.backtest.metrics import information_ratio as compute_information_ratio
from stockpredictor.backtest.metrics import max_drawdown, max_gain, sharpe_ratio
from stockpredictor.common.logging import get_logger
from stockpredictor.common.trading_calendar import last_completed_nse_session
from stockpredictor.common.types import DataLayer
from stockpredictor.ingestion.macro import read_macro_series
from stockpredictor.labels.registry import DEFAULT_BENCHMARK_SERIES
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN
from stockpredictor.storage.db import session_scope
from stockpredictor.storage.lake import Lake
from stockpredictor.storage.models import PublishedPrediction, ValidationResult

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def _none_if_nan(value: float) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _unvalidated_predictions(session: Session) -> list[PublishedPrediction]:
    stmt = (
        select(PublishedPrediction)
        .outerjoin(ValidationResult, ValidationResult.prediction_id == PublishedPrediction.prediction_id)
        .where(ValidationResult.id.is_(None))
    )
    return list(session.execute(stmt).scalars())


def _resolved_labels(lake: Lake, as_of: dt.date) -> pd.DataFrame:
    labels = lake.read_all(DataLayer.GOLD, LABELS_DOMAIN)
    if labels.empty:
        return labels
    labels = labels.dropna(subset=["forward_return", "label_valid_date"])
    labels["label_valid_date"] = pd.to_datetime(labels["label_valid_date"]).dt.normalize()
    return labels[labels["label_valid_date"] <= pd.Timestamp(as_of)]


def _daily_return_series(lake: Lake, symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    prices = lake.read(DataLayer.SILVER, "prices", symbol)
    if prices.empty:
        return pd.Series(dtype="float64")
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    window = prices[(prices["date"] >= pd.Timestamp(start)) & (prices["date"] <= pd.Timestamp(end))]
    window = window.sort_values("date")
    return window.set_index("date")["close_adj"].pct_change().dropna()


def _compute_metrics(lake: Lake, symbol: str, prediction_date: dt.date, resolution_date: dt.date) -> dict:
    stock_returns = _daily_return_series(lake, symbol, prediction_date, resolution_date)

    benchmark = read_macro_series(lake, DEFAULT_BENCHMARK_SERIES)
    benchmark = benchmark.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    bench_window = benchmark[
        (benchmark["date"] >= pd.Timestamp(prediction_date)) & (benchmark["date"] <= pd.Timestamp(resolution_date))
    ].sort_values("date")
    benchmark_returns = bench_window.set_index("date")["close"].pct_change().dropna()

    aligned = pd.DataFrame({"stock": stock_returns}).join(
        pd.DataFrame({"benchmark": benchmark_returns}), how="inner"
    )

    volatility = (
        float(stock_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(stock_returns) >= 2 else float("nan")
    )

    return {
        "maximum_drawdown": max_drawdown(stock_returns),
        "maximum_gain": max_gain(stock_returns),
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio(stock_returns, horizon_days=1),
        "information_ratio": (
            compute_information_ratio(aligned["stock"], aligned["benchmark"], horizon_days=1)
            if not aligned.empty
            else float("nan")
        ),
    }


def run_daily_validation(lake: Lake, session_factory: sessionmaker[Session]) -> int:
    """Validate every published prediction whose horizon has resolved by the
    last completed NSE session and that hasn't been validated yet. Returns
    the number of new `ValidationResult` rows written."""
    as_of = last_completed_nse_session()
    labels = _resolved_labels(lake, as_of)
    if labels.empty:
        logger.info("No resolved labels available yet -- nothing to validate")
        return 0

    written = 0
    with session_scope(session_factory) as session:
        pending = _unvalidated_predictions(session)
        if not pending:
            logger.info("No unvalidated published predictions on record")
            return 0

        for prediction in pending:
            match = labels[
                (labels["symbol"] == prediction.stock_symbol)
                & (labels["horizon"] == prediction.prediction_horizon)
                & (pd.to_datetime(labels["date"]).dt.normalize() == pd.Timestamp(prediction.prediction_date))
            ]
            if match.empty:
                continue  # not resolved yet
            label_row = match.iloc[0]
            resolution_date = label_row["label_valid_date"].date()

            metrics = _compute_metrics(lake, prediction.stock_symbol, prediction.prediction_date, resolution_date)
            actual_return = float(label_row["forward_return"])
            benchmark_return = float(label_row["benchmark_forward_return"])
            alpha = float(label_row["excess_return"])

            session.add(
                ValidationResult(
                    prediction_id=prediction.prediction_id,
                    actual_return=actual_return,
                    benchmark_return=benchmark_return,
                    alpha=alpha,
                    hit_or_miss=alpha > 0,
                    maximum_drawdown=_none_if_nan(metrics["maximum_drawdown"]),
                    maximum_gain=_none_if_nan(metrics["maximum_gain"]),
                    volatility=_none_if_nan(metrics["volatility"]),
                    sharpe_ratio=_none_if_nan(metrics["sharpe_ratio"]),
                    information_ratio=_none_if_nan(metrics["information_ratio"]),
                )
            )
            written += 1

    logger.info("Validated %d predictions as of %s", written, as_of)
    return written
