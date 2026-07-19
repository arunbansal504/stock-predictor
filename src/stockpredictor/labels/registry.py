"""Label build orchestration, mirroring features/registry.py's role: the
single function other modules (model training, backtest) call to get a
consistent, versioned label set (§27 Phase 1 step 7).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.pit import filter_as_of
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
    as_of: dt.date | None = None,
) -> pd.DataFrame:
    """Build forward-return labels for every symbol with silver price data,
    against the given benchmark series (must already be ingested -- see
    ingestion/macro.py). Returns the combined label matrix, not yet
    persisted (see persist_labels).

    `outperform` -- the model's actual training target -- is overwritten
    here, cross-sectionally, as "this stock's forward return beat the
    same-date universe median stock's forward return", not
    labels/returns.py's per-symbol "beat the benchmark index" value.
    Comparing a dividend-adjusted stock return (`close_adj`) against a
    price-only benchmark index (ingestion/macro.py ingests `^CRSLDX` without
    dividends) gives dividend-paying stocks a persistent, artificial ~1-1.5
    pct/yr edge in the label -- a bias that has nothing to do with real
    predictive skill. Comparing every stock against the same date's median
    stock keeps both sides on the identical `close_adj` basis, and it's the
    more natural target for a *cross-sectional ranking* model anyway: what
    the ranking needs to get right is which stocks beat which, not which
    stocks beat one particular index. `excess_return` (still vs. the
    benchmark index) is kept as-is, unchanged, as a reporting/explain
    column -- not as a training target.

    `as_of`, when given, drops price/benchmark rows after that date first --
    same rationale as features/registry.py's `as_of` (common/trading_calendar.py):
    keeps a stale partial bar already on disk from leaking into labels."""
    horizons = horizons or DEFAULT_HORIZONS
    prices = lake.read_all(DataLayer.SILVER, "prices")
    if prices.empty:
        return pd.DataFrame()
    if as_of is not None:
        prices = filter_as_of(prices, pd.Timestamp(as_of))
        if prices.empty:
            return pd.DataFrame()

    benchmark = read_macro_series(lake, benchmark_series)
    if benchmark.empty:
        raise ValueError(
            f"Benchmark series '{benchmark_series}' not found in the lake -- "
            "run ingestion.macro.ingest_macro_series first."
        )
    if as_of is not None:
        benchmark = filter_as_of(benchmark, pd.Timestamp(as_of))

    per_symbol = [
        build_labels_for_symbol(group, benchmark, horizons) for _, group in prices.groupby("symbol")
    ]
    labels = pd.concat(per_symbol, ignore_index=True)

    median_forward_return = labels.groupby(["horizon", "date"])["forward_return"].transform("median")
    outperform_median = labels["forward_return"].gt(median_forward_return)
    labels["outperform"] = outperform_median.astype("boolean").mask(labels["forward_return"].isna())
    return labels


def persist_labels(lake: Lake, labels: pd.DataFrame) -> int:
    """Write the label matrix to the Gold layer, one file per symbol."""
    if labels.empty:
        return 0
    total = 0
    for symbol, group in labels.groupby("symbol"):
        total += lake.write(group, DataLayer.GOLD, GOLD_DOMAIN, symbol, key_cols=GOLD_KEY_COLS)
    logger.info("Persisted %d label rows", total)
    return total
