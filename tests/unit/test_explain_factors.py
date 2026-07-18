from __future__ import annotations

from stockpredictor.explain.factors import FACTOR_BLOCKS, feature_to_block
from stockpredictor.features.registry import ALL_FEATURE_COLUMNS


def test_feature_to_block_maps_known_raw_feature():
    assert feature_to_block("rsi_14") == "Oscillators"
    assert feature_to_block("return_5d") == "Momentum/Trend"
    assert feature_to_block("atr_14_pct") == "Volatility/Risk"
    assert feature_to_block("volume_zscore_20d") == "Volume/Liquidity"


def test_feature_to_block_maps_xrank_variant_same_as_raw():
    assert feature_to_block("rsi_14_xrank") == feature_to_block("rsi_14")


def test_feature_to_block_unknown_feature_returns_other():
    assert feature_to_block("some_future_fundamental_feature") == "Other"


def test_feature_to_block_maps_known_fundamental_feature():
    assert feature_to_block("roe") == "Fundamental/Quality"
    assert feature_to_block("pe_ratio_xrank") == "Fundamental/Quality"


def test_every_feature_column_is_mapped_to_a_real_block():
    """Lineage guard: if features/registry.py adds a new technical or
    fundamental feature without updating FACTOR_BLOCKS, that feature would
    silently land in "Other" on every explanation card. Catch the drift
    here instead."""
    unmapped = [c for c in ALL_FEATURE_COLUMNS if feature_to_block(c) == "Other"]
    assert unmapped == [], f"Unmapped feature columns: {unmapped}"


def test_no_feature_appears_in_more_than_one_block():
    seen = set()
    for block, features in FACTOR_BLOCKS.items():
        for f in features:
            assert f not in seen, f"'{f}' appears in more than one factor block"
            seen.add(f)
