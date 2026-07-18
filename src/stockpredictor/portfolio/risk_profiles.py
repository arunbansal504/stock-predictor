"""Risk profile presets (§12): "Risk profiles: Conservative / Balanced /
Aggressive change target vol, max single-name weight, and min
diversification."

Reuses the `RiskProfile` enum already defined in common/types.py (used
elsewhere for the same three-tier concept) rather than duplicating it.
Parameter values are standard, defensible retail risk-tiering conventions
(tighter caps and stops for Conservative, looser for Aggressive) -- not
empirically fit to this system's own backtest, since we don't yet have
enough live portfolio-level history to fit them honestly. Revisit once we
do.
"""

from __future__ import annotations

from dataclasses import dataclass

from stockpredictor.common.types import RiskProfile


@dataclass(frozen=True)
class RiskProfileParams:
    max_position_weight: float  # cap on any single stock's portfolio weight
    max_sector_weight: float  # cap on any single sector's combined weight
    min_positions: int  # diversification floor -- fewer names than this is too concentrated
    confidence_tilt_strength: float  # 0 = pure HRP risk-parity, 1 = fully score-weighted
    stop_loss_atr_multiplier: float  # stop = entry - multiplier * ATR
    target_reward_risk_ratio: float  # target distance = reward_risk_ratio * stop distance


RISK_PROFILE_PARAMS: dict[RiskProfile, RiskProfileParams] = {
    RiskProfile.CONSERVATIVE: RiskProfileParams(
        max_position_weight=0.10,
        max_sector_weight=0.25,
        min_positions=10,
        confidence_tilt_strength=0.2,
        stop_loss_atr_multiplier=1.5,
        target_reward_risk_ratio=1.5,
    ),
    RiskProfile.BALANCED: RiskProfileParams(
        max_position_weight=0.15,
        max_sector_weight=0.35,
        min_positions=6,
        confidence_tilt_strength=0.4,
        stop_loss_atr_multiplier=2.0,
        target_reward_risk_ratio=2.0,
    ),
    RiskProfile.AGGRESSIVE: RiskProfileParams(
        max_position_weight=0.25,
        max_sector_weight=0.50,
        min_positions=3,
        confidence_tilt_strength=0.6,
        stop_loss_atr_multiplier=2.5,
        target_reward_risk_ratio=2.5,
    ),
}


def get_risk_profile_params(profile: RiskProfile) -> RiskProfileParams:
    return RISK_PROFILE_PARAMS[profile]
