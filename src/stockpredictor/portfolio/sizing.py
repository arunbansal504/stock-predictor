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

import numpy as np
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


def _cap_and_redistribute(weights: pd.Series, caps: pd.Series) -> pd.Series:
    """Like `apply_position_cap`, but against a per-position ceiling
    (`caps`) instead of one scalar shared by every position -- lets
    `apply_position_and_sector_caps` shrink individual positions' room as
    sector violations are discovered, reusing the same cascading
    redistribution logic rather than re-deriving it."""
    weights = weights.copy()
    for _ in range(len(weights)):
        over = weights[weights > caps + 1e-12]
        if over.empty:
            break
        excess = (over - caps[over.index]).sum()
        weights[over.index] = caps[over.index]
        under = weights[weights < caps - 1e-12]
        if under.empty:
            break
        weights[under.index] += excess * (under / under.sum())
    return weights


def apply_position_and_sector_caps(
    weights: pd.Series, sectors: pd.Series, max_position_weight: float, max_sector_weight: float
) -> pd.Series:
    """Enforce both caps simultaneously, not by applying each once in a
    fixed order. A single `apply_position_cap` -> `apply_sector_caps` pass
    (or the reverse) can leave one constraint violated: neither function is
    aware of the other's grouping, so redistributing one cap's excess can
    push a position or sector back over the other cap -- e.g. observed
    live, three same-sector positions each independently satisfying a 15%
    position cap but summing to 45% against a 35% sector cap once position-
    cap redistribution ran last and undid the sector cap's work.

    A naive alternate-until-the-composed-pass-is-a-no-op loop is NOT
    sufficient here either -- verified by hand-tracing an adversarial case
    (three same-sector positions plus one other-sector position, tight caps
    on both): position-cap redistribution and sector-cap redistribution can
    settle into a genuine 2-cycle where the *composition* is stable (each
    full pass reproduces the same numbers) even though neither individual
    constraint is actually satisfied by that fixed point -- the position
    cap gets silently violated again on the very state the loop treats as
    "converged."

    Instead: track a per-position `effective_cap` (starts at
    `max_position_weight`), and whenever a sector is found over
    `max_sector_weight`, scale that whole sector's members down to fit and
    permanently tighten their `effective_cap` to the scaled-down value.
    Once tightened, `_cap_and_redistribute` will never again treat that
    position as having spare room (it's exactly at its own cap), so a later
    pass can no longer hand it more weight and re-violate the sector cap --
    this is what breaks the ping-pong instead of just hiding it behind a
    coincidental fixed point. `effective_cap` only ever shrinks, so the
    outer loop provably terminates: each iteration either tightens at least
    one previously-unlocked position (bounded by len(weights)) or leaves
    every sector compliant and weights unchanged, at which point both
    constraints hold and the loop exits."""
    weights = weights.copy()
    sectors = sectors.reindex(weights.index)
    effective_cap = pd.Series(max_position_weight, index=weights.index)

    max_iterations = len(weights) + int(sectors.nunique()) + 4
    for _ in range(max_iterations):
        before = weights.to_numpy(copy=True)
        weights = _cap_and_redistribute(weights, effective_cap)

        sector_totals = weights.groupby(sectors).sum()
        over_sectors = sector_totals[sector_totals > max_sector_weight + 1e-9]
        if over_sectors.empty:
            if np.allclose(weights.to_numpy(), before, atol=1e-12):
                break
            continue

        # Same batched excess-then-redistribute convention as
        # apply_sector_caps: accumulate all over-sectors' excess first,
        # THEN redistribute it once into positions outside any over-cap
        # sector -- freed-up capacity from capping one sector must be able
        # to flow into another under-cap sector, not just vanish (an
        # earlier version of this function dropped it here entirely,
        # which under-allocated capital that a jointly-feasible solution
        # could have used).
        total_excess = 0.0
        for sector, total in over_sectors.items():
            members = sectors[sectors == sector].index
            scale = max_sector_weight / total
            total_excess += weights[members].sum() - max_sector_weight
            weights[members] *= scale
            effective_cap[members] = np.minimum(effective_cap[members], weights[members])

        under_mask = ~sectors.isin(over_sectors.index)
        under = weights[under_mask]
        if not under.empty:
            weights[under.index] += total_excess * (under / under.sum())

    return weights
