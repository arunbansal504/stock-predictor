"""Factor block definitions (§7): maps engineered feature names to the
interpretable groups shown to users (Trend, Oscillators, Volatility,
Volume, Fundamental/Quality). Sentiment, Macro, and Ownership blocks remain
placeholders until those data sources are ingested (§27 Phase 2+); a
feature landing in "Other" is a signal that this mapping needs updating,
not a bug in the caller.
"""

from __future__ import annotations

FACTOR_BLOCKS: dict[str, list[str]] = {
    "Momentum/Trend": [
        "return_5d",
        "return_20d",
        "return_60d",
        "return_120d",
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "price_vs_sma20",
        "price_vs_sma50",
        "macd",
        "macd_signal",
        "macd_hist",
        "dist_from_52w_high",
        "dist_from_52w_low",
    ],
    "Oscillators": ["rsi_14", "bb_pctb", "bb_width"],
    "Volatility/Risk": ["atr_14", "atr_14_pct", "realized_vol_20d", "realized_vol_60d"],
    "Volume/Liquidity": ["obv", "volume_zscore_20d"],
    # revenue_growth_yoy/eps_growth_yoy deliberately excluded here too --
    # see features/fundamental.py's docstring for why they're not in
    # FUNDAMENTAL_FEATURE_COLUMNS (and therefore never appear as real model
    # features to explain).
    "Fundamental/Quality": ["pe_ratio", "pb_ratio", "roe", "roa", "debt_to_equity", "net_margin"],
}

_FEATURE_TO_BLOCK: dict[str, str] = {
    feature: block for block, features in FACTOR_BLOCKS.items() for feature in features
}


def feature_to_block(feature_name: str) -> str:
    """Map a raw feature name (or its `_xrank` cross-sectional variant) to
    its factor block. Unknown features map to "Other" rather than raising --
    explainability should degrade gracefully, not crash a ranking run."""
    base = feature_name[: -len("_xrank")] if feature_name.endswith("_xrank") else feature_name
    return _FEATURE_TO_BLOCK.get(base, "Other")
