"""Position sizing (§12): blends HRP's risk-parity base weights with the
model's confidence score, then enforces risk-profile constraints (max
position weight, max sector weight).

"Position size also scaled by confidence and inverse-volatility" (§12) --
HRP already captures inverse-volatility via its risk-parity construction;
this module adds the confidence tilt on top and then enforces the hard caps
a systematic risk-management process needs regardless of what the
unconstrained optimization wants.
"""

from __future__ import annotations

import pandas as pd


def apply_confidence_tilt(hrp_weights: pd.Series, scores: pd.Series, tilt_strength: float) -> pd.Series:
    """Blend HRP weights with score-proportional weights:
    final = (1 - tilt_strength) * HRP + tilt_strength * score-weighted.
    tilt_strength=0 -> pure risk parity; tilt_strength=1 -> pure conviction
    weighting (ignoring risk entirely) -- risk profiles pick a middle ground
    (see risk_profiles.py)."""
    scores = scores.reindex(hrp_weights.index)
    score_weights = scores / scores.sum()
    blended = (1 - tilt_strength) * hrp_weights + tilt_strength * score_weights
    return blended / blended.sum()


def apply_position_cap(weights: pd.Series, max_weight: float) -> pd.Series:
    """Cap any single position at `max_weight`, redistributing the excess
    proportionally across uncapped positions. Iterates because capping one
    position can push another over the cap once the excess lands on it;
    bounded to len(weights) iterations since each iteration caps at least
    one previously-uncapped position."""
    weights = weights.copy()
    for _ in range(len(weights)):
        over = weights[weights > max_weight]
        if over.empty:
            break
        excess = (over - max_weight).sum()
        weights[over.index] = max_weight
        under = weights[weights < max_weight]
        if under.empty:
            # Every position is already at the cap -- cannot redistribute
            # further; weights will sum to < 1, an honest "can't fully
            # allocate under these constraints" signal, not a silently
            # violated cap.
            break
        weights[under.index] += excess * (under / under.sum())
    return weights


def apply_sector_caps(weights: pd.Series, sectors: pd.Series, max_sector_weight: float) -> pd.Series:
    """Cap any single sector's combined weight at `max_sector_weight`,
    redistributing the excess proportionally across positions in
    under-cap sectors. Same bounded-iteration approach as apply_position_cap."""
    weights = weights.copy()
    sectors = sectors.reindex(weights.index)
    n_sectors = sectors.nunique()

    for _ in range(n_sectors + 1):
        sector_totals = weights.groupby(sectors).sum()
        over_sectors = sector_totals[sector_totals > max_sector_weight]
        if over_sectors.empty:
            break

        total_excess = 0.0
        for sector, total in over_sectors.items():
            members = sectors[sectors == sector].index
            scale = max_sector_weight / total
            total_excess += weights[members].sum() - max_sector_weight
            weights[members] *= scale

        under_mask = ~sectors.isin(over_sectors.index)
        under = weights[under_mask]
        if under.empty:
            break
        weights[under.index] += total_excess * (under / under.sum())

    return weights
