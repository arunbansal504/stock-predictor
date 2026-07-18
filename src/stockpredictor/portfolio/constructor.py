"""Portfolio construction (§12): turns a ranked Top-N list into an
illustrative research portfolio -- HRP allocation tilted by confidence,
risk-profile position/sector caps, per-stock stop-loss/target, and
portfolio-level expected return/volatility/Sharpe.

This is NOT model inference -- it's a deterministic optimization over
already-published rankings/scores/prices (§4's "the UI never triggers
model inference on demand" is about training/prediction, not this). Safe to
compute on demand from an API request, unlike re-scoring a fresh model.

Deliberately does not estimate a portfolio-level max-drawdown figure: doing
so honestly would need a return-path simulation (e.g. historical
bootstrap), not a closed-form number from static weights and a covariance
matrix. A single "expected volatility" scalar is defensible; a fabricated
single drawdown number would carry false precision. Revisit if a
simulation-based estimate gets built.

`construct_portfolio` is a pure function (DataFrames/Series in, a
dataclass out) with no lake/DB access, so it's fully unit-testable without
fixtures beyond plain pandas objects -- the API layer (api/app.py) does the
actual lake/DB reads and calls this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockpredictor.common.types import RiskProfile
from stockpredictor.portfolio.hrp import compute_hrp_weights, compute_returns_matrix
from stockpredictor.portfolio.risk_profiles import get_risk_profile_params
from stockpredictor.portfolio.sizing import apply_confidence_tilt, apply_position_cap, apply_sector_caps
from stockpredictor.portfolio.targets import compute_stock_targets

TRADING_DAYS_PER_YEAR = 252

DISCLAIMER = (
    "Illustrative research portfolio, not investment advice. Shows how a "
    "systematic process would weight and risk-manage these positions -- not a directive."
)


def _parse_horizon_days(horizon: str) -> int | None:
    """Parse the "Nd" trading-day-count convention used throughout this
    codebase (see common/types.py's Horizon enum: "5d", "30d", "90d", ...).
    Returns None (rather than raising) for an unrecognized format -- a
    display-only horizon label shouldn't crash portfolio construction."""
    if horizon.endswith("d") and horizon[:-1].isdigit():
        return int(horizon[:-1])
    return None


@dataclass(frozen=True)
class PortfolioPosition:
    symbol: str
    weight: float
    score: float
    sector: str | None
    entry_price: float
    stop_loss: float
    target_price: float
    expected_return: float | None


@dataclass(frozen=True)
class ConstructedPortfolio:
    risk_profile: str
    horizon: str
    positions: list[PortfolioPosition]
    expected_return: float | None
    expected_volatility: float | None
    expected_sharpe: float | None
    diversification_warning: str | None
    total_allocated_weight: float
    excluded_symbols: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


def construct_portfolio(
    ranked: pd.DataFrame,
    prices: pd.DataFrame,
    atr_by_symbol: pd.Series,
    sector_by_symbol: pd.Series,
    return_calibration: pd.DataFrame,
    risk_profile: RiskProfile,
    horizon: str,
    top_n: int,
    lookback_days: int = 90,
) -> ConstructedPortfolio:
    """
    ranked: must have columns [symbol, rank, score], one row per candidate.
    prices: long silver-prices-shaped frame [symbol, date, close_adj]
        covering at least `lookback_days` of history for the candidates.
    atr_by_symbol / sector_by_symbol: latest ATR-14 and sector, indexed by symbol.
    return_calibration: from backtest/calibration_curve.py, see targets.py.
    """
    params = get_risk_profile_params(risk_profile)
    candidates = ranked.sort_values("rank").head(top_n)
    requested_symbols = candidates["symbol"].tolist()

    diversification_warning = None
    if len(requested_symbols) < params.min_positions:
        diversification_warning = (
            f"Only {len(requested_symbols)} candidate(s) requested; the {risk_profile.value} risk "
            f"profile recommends at least {params.min_positions} positions for adequate diversification."
        )

    returns_matrix = compute_returns_matrix(prices, requested_symbols, lookback_days)
    usable_symbols = list(returns_matrix.columns)
    excluded_symbols = sorted(set(requested_symbols) - set(usable_symbols))
    candidates = candidates[candidates["symbol"].isin(usable_symbols)].set_index("symbol")

    if not usable_symbols:
        return ConstructedPortfolio(
            risk_profile=risk_profile.value,
            horizon=horizon,
            positions=[],
            expected_return=None,
            expected_volatility=None,
            expected_sharpe=None,
            diversification_warning=diversification_warning,
            total_allocated_weight=0.0,
            excluded_symbols=excluded_symbols,
        )

    hrp_weights = compute_hrp_weights(returns_matrix)
    scores = candidates["score"]
    tilted = apply_confidence_tilt(hrp_weights, scores, params.confidence_tilt_strength)

    sectors = sector_by_symbol.reindex(usable_symbols).fillna("Unknown")
    sector_capped = apply_sector_caps(tilted, sectors, params.max_sector_weight)
    final_weights = apply_position_cap(sector_capped, params.max_position_weight)

    # Last *valid* price per symbol, not just the chronologically last row --
    # observed live: a free-data-source gap can leave a row present for the
    # latest date with a null close_adj (not merely a missing row), which
    # `.tail(1)` alone would silently pick up and propagate as a NaN entry
    # price into every downstream stop-loss/target calculation.
    valid_prices = prices[prices["symbol"].isin(usable_symbols)].dropna(subset=["close_adj"])
    latest_close = valid_prices.sort_values("date").groupby("symbol").tail(1).set_index("symbol")["close_adj"]

    positions = []
    for symbol in usable_symbols:
        entry_price = float(latest_close.get(symbol, np.nan))
        atr = float(atr_by_symbol.get(symbol, np.nan))
        score = float(scores.get(symbol, np.nan))
        targets = compute_stock_targets(
            entry_price, atr, score, params.stop_loss_atr_multiplier, params.target_reward_risk_ratio, return_calibration
        )
        positions.append(
            PortfolioPosition(
                symbol=symbol,
                weight=float(final_weights.get(symbol, 0.0)),
                score=score,
                sector=sector_by_symbol.get(symbol),
                entry_price=entry_price,
                stop_loss=targets.stop_loss,
                target_price=targets.target_price,
                expected_return=targets.expected_return,
            )
        )

    weights_vec = final_weights.reindex(usable_symbols).values
    cov = returns_matrix.cov().reindex(index=usable_symbols, columns=usable_symbols).values
    daily_vol = float(np.sqrt(weights_vec @ cov @ weights_vec))
    expected_volatility = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    total_allocated_weight = float(sum(p.weight for p in positions))
    if total_allocated_weight < 0.999:
        # A tight per-position cap applied to too few names creates a hard
        # mathematical ceiling below 100% (e.g. 5 names at a 10% cap can
        # never exceed 50%, regardless of implementation) -- this is the
        # correct, honest consequence of the constraint, but must be
        # surfaced explicitly rather than left for the caller to notice by
        # summing weights themselves.
        shortfall_note = (
            f"Position/sector caps for the {risk_profile.value} profile only allow "
            f"{total_allocated_weight:.0%} of capital to be allocated across these "
            f"{len(usable_symbols)} name(s) -- increase top_n or choose a less "
            f"conservative risk profile to fully allocate."
        )
        diversification_warning = (
            f"{diversification_warning} {shortfall_note}" if diversification_warning else shortfall_note
        )

    return_components = [
        (p.weight, p.expected_return) for p in positions if p.expected_return is not None
    ]
    if return_components:
        total_w = sum(w for w, _ in return_components)
        expected_return = sum(w * r for w, r in return_components) / total_w if total_w > 0 else None
    else:
        expected_return = None

    # expected_return is over the holding period (`horizon`, e.g. 5 trading
    # days); expected_volatility is annualized. Naively dividing them mixes
    # units; naively *compounding* the return to "annualize" it is worse --
    # a modest 2-3% return compounded ~50x/year (252/5) explodes to a
    # triple-digit number that looks like a bug, not an estimate. The
    # standard, honest convention (matching backtest/metrics.py's own
    # sharpe_ratio: per-period mean/std, then scaled by sqrt(periods/year))
    # is sqrt-of-time scaling applied consistently to both the return and
    # the volatility side of the ratio, not compounding on one side only.
    horizon_days = _parse_horizon_days(horizon)
    if expected_return is not None and horizon_days and daily_vol > 0:
        period_vol = daily_vol * np.sqrt(horizon_days)  # volatility over the same period as expected_return
        expected_sharpe = (expected_return / period_vol) * np.sqrt(TRADING_DAYS_PER_YEAR / horizon_days)
    else:
        expected_sharpe = None

    return ConstructedPortfolio(
        risk_profile=risk_profile.value,
        horizon=horizon,
        positions=positions,
        expected_return=expected_return,
        expected_volatility=expected_volatility,
        expected_sharpe=expected_sharpe,
        diversification_warning=diversification_warning,
        total_allocated_weight=total_allocated_weight,
        excluded_symbols=excluded_symbols,
    )
