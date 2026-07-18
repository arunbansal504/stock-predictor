"""Feature registry + build orchestration (§7: "feature registry with
lineage").

Tracks which named, versioned feature set is active and provides the single
function (`build_technical_features_for_universe`) other modules (labels,
model, backtest) call to get a consistent feature matrix -- so a change to
the feature set is a deliberate version bump, not a silent redefinition that
breaks reproducibility (§3 NFR: "reproducible predictions"). Despite the
name (kept for call-site stability), this now also merges in the
Fundamental/Quality block (features/fundamental.py) -- see
`ALL_FEATURE_COLUMNS` for what a model actually trains on.
"""

from __future__ import annotations

import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.features import technical
from stockpredictor.features.cross_sectional import add_cross_sectional_rank
from stockpredictor.features.fundamental import FUNDAMENTAL_FEATURE_COLUMNS, build_fundamental_features_for_symbol
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

FEATURE_SET_VERSION = "v2"  # bumped: v1 was technical-only, v2 adds Fundamental/Quality

# Raw (per-symbol, non-cross-sectional) technical feature columns produced by
# features.technical.compute_technical_features. One source of truth so
# cross-sectional ranking and downstream model code agree on exactly which
# columns are "features" vs. metadata (symbol/date/knowable_date).
TECHNICAL_FEATURE_COLUMNS: list[str] = [
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
    "rsi_14",
    "atr_14",
    "atr_14_pct",
    "bb_pctb",
    "bb_width",
    "realized_vol_20d",
    "realized_vol_60d",
    "obv",
    "dist_from_52w_high",
    "dist_from_52w_low",
    "volume_zscore_20d",
]

ALL_FEATURE_COLUMNS: list[str] = TECHNICAL_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_COLUMNS

# features/sentiment.py's SENTIMENT_FEATURE_COLUMNS is deliberately NOT
# merged in here yet -- see that module's docstring: the news connector has
# no historical backfill, so there isn't enough real history to evaluate
# out-of-sample yet. It's ingested nightly (accumulating real data) and
# used standalone for the UI's live sentiment panel in the meantime.

GOLD_DOMAIN = "features"
GOLD_KEY_COLS = ["symbol", "date"]


def build_technical_features_for_universe(lake: Lake) -> pd.DataFrame:
    """Compute the technical + fundamental blocks for every symbol with
    silver price data, add cross-sectional rank transforms, and return the
    combined feature matrix (not yet written to the lake -- see
    persist_features). Fundamentals are optional: a symbol with no
    fundamentals data yet still gets technical features, with NaN
    fundamental columns (an honest gap, not a reason to drop the symbol)."""
    prices = lake.read_all(DataLayer.SILVER, "prices")
    if prices.empty:
        return pd.DataFrame()

    fundamentals = lake.read_all(DataLayer.SILVER, "fundamentals")

    per_symbol = []
    for symbol, group in prices.groupby("symbol"):
        feats = technical.compute_technical_features(group)

        symbol_fundamentals = (
            fundamentals[fundamentals["symbol"] == symbol] if not fundamentals.empty else pd.DataFrame()
        )
        fund_feats = build_fundamental_features_for_symbol(group, symbol_fundamentals)
        feats = feats.merge(fund_feats.drop(columns=["symbol"]), on="date", how="left")

        per_symbol.append(feats)
    matrix = pd.concat(per_symbol, ignore_index=True)

    matrix = add_cross_sectional_rank(matrix, ALL_FEATURE_COLUMNS)
    matrix["feature_set_version"] = FEATURE_SET_VERSION
    return matrix


def persist_features(lake: Lake, matrix: pd.DataFrame) -> int:
    """Write the feature matrix to the Gold layer, one file per symbol
    (matches the lake's per-symbol partitioning convention, see
    storage/lake.py)."""
    if matrix.empty:
        return 0
    total = 0
    for symbol, group in matrix.groupby("symbol"):
        total += lake.write(group, DataLayer.GOLD, GOLD_DOMAIN, symbol, key_cols=GOLD_KEY_COLS)
    logger.info("Persisted %d feature rows (version=%s)", total, FEATURE_SET_VERSION)
    return total
