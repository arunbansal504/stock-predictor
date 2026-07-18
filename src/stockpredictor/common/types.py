"""Shared enums and small value types used across the pipeline.

Keeping these centralized avoids "magic string" drift between connectors,
features, and storage (e.g. one module writing "NSE" and another "nse").
"""

from __future__ import annotations

from enum import Enum


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"


class Horizon(str, Enum):
    """Prediction horizons. MVP (Phase 1) implements a subset — see
    config/model.yaml — the rest exist so downstream code can already model
    the full F6 requirement without a later breaking rename.
    """

    D1 = "1d"
    D3 = "3d"
    D5 = "5d"
    D10 = "10d"
    D30 = "30d"
    D90 = "90d"
    D180 = "180d"
    D365 = "365d"


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class DataLayer(str, Enum):
    """Medallion architecture layers (§5 of the architecture doc)."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
