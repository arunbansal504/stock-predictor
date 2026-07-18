"""Per-stock stop-loss/target and expected-return estimates (§12).

Stop-loss/target use a standard ATR-based bracket -- independent of model
prediction quality, well-understood, and always available (ATR is already
a technical feature, see features/technical.py). "Expected return" is
separately derived from backtest/calibration_curve.py's decile-conditional
historical realized returns -- see that module's docstring for why we
don't fabricate a return magnitude directly from the classifier's score.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpredictor.backtest.calibration_curve import lookup_expected_return


@dataclass(frozen=True)
class StockTargets:
    entry_price: float
    stop_loss: float
    target_price: float
    expected_return: float | None  # None if no calibration data was available


def compute_stop_loss_target(
    entry_price: float,
    atr: float,
    stop_multiplier: float,
    reward_risk_ratio: float,
) -> tuple[float, float]:
    """ATR-based bracket for a long position: stop below entry by
    `stop_multiplier` ATRs, target above entry by `reward_risk_ratio` times
    that same stop distance. Standard technical risk management, entirely
    independent of model confidence -- a stock with a NaN ATR (insufficient
    price history) correctly produces a NaN stop/target rather than a
    fabricated bracket."""
    stop_distance = stop_multiplier * atr
    stop_loss = entry_price - stop_distance
    target_price = entry_price + reward_risk_ratio * stop_distance
    return stop_loss, target_price


def compute_stock_targets(
    entry_price: float,
    atr: float,
    score: float,
    stop_multiplier: float,
    reward_risk_ratio: float,
    return_calibration: pd.DataFrame,
) -> StockTargets:
    stop_loss, target_price = compute_stop_loss_target(entry_price, atr, stop_multiplier, reward_risk_ratio)
    expected_return = lookup_expected_return(score, return_calibration)
    return StockTargets(
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        expected_return=expected_return,
    )
