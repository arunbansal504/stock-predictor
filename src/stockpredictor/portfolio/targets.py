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

# How many multiples of the reference horizon a custom `n_days` may be
# stretched (in either direction) before `estimate_return_for_days`'s linear
# extrapolation is flagged as reflecting essentially no direct historical
# evidence -- e.g. a 5-day calibration curve stretched to 5000 days (~20
# years) is 1000x away and says nothing real about that horizon, correct
# scaling math notwithstanding. Chosen generously (an order of magnitude)
# so ordinary "what if I held a bit longer/shorter" questions never trip it.
MAX_REASONABLE_EXTRAPOLATION_MULTIPLE = 10


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


def estimate_return_for_days(
    score: float,
    return_calibration: pd.DataFrame,
    n_days: int,
    reference_horizon_days: int,
) -> float | None:
    """"What if I invest for N days" extrapolation of a calibrated expected
    return: looks up the historically-calibrated return over
    `reference_horizon_days` (the horizon backtest/calibration_curve.py's
    table was actually built for, e.g. 5/30/90) via `lookup_expected_return`,
    then scales it to an arbitrary `n_days` **linearly** --
    `calibrated_return * (n_days / reference_horizon_days)`.

    This is deliberately NOT the sqrt-of-time convention
    portfolio/constructor.py uses for `expected_sharpe` -- that scales a
    *ratio* (return / volatility), where sqrt-of-time is correct precisely
    because, under a random-walk assumption, expected return scales
    linearly with time (E[R_n] = n * E[R_1]) while volatility scales with
    sqrt(time) (vol_n = sqrt(n) * vol_1); dividing the two leaves a net
    sqrt(time) factor on the ratio. Applying sqrt(time) directly to a raw
    return, as an earlier version of this function did, conflates those two
    quantities and silently *understates* long-horizon returns (e.g.
    sqrt(1000) ≈ 32 vs. the correct linear factor of 1000). Naively
    *compounding* instead (`(1 + r) ** (n / ref) - 1`) overcorrects the
    other way, exploding a modest few-percent calibrated return into an
    absurd figure for a large day count -- linear scaling is the honest
    middle ground consistent with the calibration data actually being a
    *simple*, non-compounded realized return over `reference_horizon_days`.

    This is still just a linear extrapolation from the nearest calibrated
    horizon, not a dedicated forecast for `n_days` -- see DISCLAIMER. The
    further `n_days` is from `reference_horizon_days`, the less this
    extrapolation actually means; a 5d calibration curve stretched out to
    a `n_days` of, say, 5000 (~20 years) reflects no evidence about
    behavior at that horizon at all, linear or otherwise. Returns None if
    there's no calibration data for `score` at all (mirrors
    `lookup_expected_return`). Raises ValueError for `n_days <= 0`."""
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")
    calibrated_return = lookup_expected_return(score, return_calibration)
    if calibrated_return is None:
        return None
    return calibrated_return * (n_days / reference_horizon_days)


def extrapolation_warning(n_days: int, reference_horizon_days: int) -> str | None:
    """Flags when `n_days` (as passed to `estimate_return_for_days`) is
    stretched far enough from `reference_horizon_days` that the linear
    extrapolation reflects essentially no direct historical evidence --
    see `estimate_return_for_days`'s docstring and
    MAX_REASONABLE_EXTRAPOLATION_MULTIPLE above. Symmetric: `n_days` far
    *below* the reference horizon (e.g. asking a 90d curve about 1 day) is
    just as much an unevidenced extrapolation as one far above it -- both
    directions are checked via the larger of the two ratios. Returns None
    when `n_days` stays within a reasonable multiple either way."""
    multiple = max(n_days / reference_horizon_days, reference_horizon_days / n_days)
    if multiple <= MAX_REASONABLE_EXTRAPOLATION_MULTIPLE:
        return None
    return (
        f"{n_days} day(s) is {multiple:.0f}x this strategy's nearest published "
        f"calibration horizon ({reference_horizon_days}d) -- this extrapolation is far "
        "outside any horizon the calibration curve has actually been checked against. "
        "Treat it as a rough order-of-magnitude guess, not even an approximation."
    )
