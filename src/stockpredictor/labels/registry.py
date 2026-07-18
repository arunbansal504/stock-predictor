"""Label build orchestration, mirroring features/registry.py's role: the
single function other modules (model training, backtest) call to get a
consistent, versioned label set (§27 Phase 1 step 7).
"""

from __future__ import annotations

import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.ingestion.macro import read_macro_series
from stockpredictor.labels.returns import build_labels_for_symbol
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

# Default horizons for Phase 1 (§27, §29) -- kept in sync with
# config/model.yaml's `horizons` list; not read from YAML directly here to
# avoid a config-parsing dependency in a small, easily-unit-tested module.
DEFAULT_HORIZONS: dict[str, int] = {"5d": 5, "30d": 30, "90d": 90}

GOLD_DOMAIN = "labels"
GOLD_KEY_COLS = ["symbol", "date", "horizon"]
DEFAULT_BENCHMARK_SERIES = "NIFTY500"


def build_labels_for_universe(
    lake: Lake,
    benchmark_series: str = DEFAULT_BENCHMARK_SERIES,
    horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build excess-return labels for every symbol with silver price data,
    against the given benchmark series (must already be ingested -- see
    ingestion/macro.py). Returns the combined label matrix, not yet
    persisted (see persist_labels)."""
    horizons = horizons or DEFAULT_HORIZONS
    prices = lake.read_all(DataLayer.SILVER, "prices")
    if prices.empty:
        return pd.DataFrame()

    benchmark = read_macro_series(lake, benchmark_series)
    if benchmark.empty:
        raise ValueError(
            f"Benchmark series '{benchmark_series}' not found in the lake -- "
            "run ingestion.macro.ingest_macro_series first."
        )

    per_symbol = [
        build_labels_for_symbol(group, benchmark, horizons) for _, group in prices.groupby("symbol")
    ]
    return pd.concat(per_symbol, ignore_index=True)


def persist_labels(lake: Lake, labels: pd.DataFrame) -> int:
    """Write the label matrix to the Gold layer, one file per symbol."""
    if labels.empty:
        return 0
    total = 0
    for symbol, group in labels.groupby("symbol"):
        total += lake.write(group, DataLayer.GOLD, GOLD_DOMAIN, symbol, key_cols=GOLD_KEY_COLS)
    logger.info("Persisted %d label rows", total)
    return total
