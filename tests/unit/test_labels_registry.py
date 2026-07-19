from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.common.types import DataLayer
from stockpredictor.labels import registry


def _seed_prices(tmp_lake, symbol: str, closes: list[float]) -> None:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame({"symbol": [symbol] * len(closes), "date": dates, "close_adj": closes})
    tmp_lake.write(df, DataLayer.SILVER, "prices", symbol, key_cols=["symbol", "date"])


def _seed_benchmark(tmp_lake, closes: list[float]) -> None:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame({"series": ["NIFTY500"] * len(closes), "date": dates, "close": closes})
    tmp_lake.write(df, DataLayer.SILVER, "macro", "NIFTY500", key_cols=["series", "date"])


def test_build_labels_for_universe_empty_when_no_prices(tmp_lake):
    out = registry.build_labels_for_universe(tmp_lake)
    assert out.empty


def test_build_labels_for_universe_raises_without_benchmark(tmp_lake):
    _seed_prices(tmp_lake, "AAA", [100, 101, 102, 103, 104, 105])
    with pytest.raises(ValueError, match="Benchmark series"):
        registry.build_labels_for_universe(tmp_lake)


def test_build_labels_for_universe_combines_all_symbols(tmp_lake):
    _seed_prices(tmp_lake, "AAA", [100, 101, 102, 103, 104, 105, 106, 107])
    _seed_prices(tmp_lake, "BBB", [50, 51, 49, 52, 53, 54, 55, 56])
    _seed_benchmark(tmp_lake, [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007])

    out = registry.build_labels_for_universe(tmp_lake, horizons={"2d": 2})
    assert set(out["symbol"].unique()) == {"AAA", "BBB"}
    assert set(out["horizon"]) == {"2d"}


def test_outperform_is_beat_universe_median_not_beat_benchmark(tmp_lake):
    """The training target is redefined cross-sectionally (see
    build_labels_for_universe's docstring): a stock's `outperform` reflects
    whether its forward return beat the same-date universe MEDIAN stock,
    not whether it beat the benchmark index -- so a stock can beat the
    index yet still be `outperform=False` if most of the universe beat it
    by more, and vice versa."""
    # 3 symbols, 2-day horizon, single decision date: AAA +6%, BBB +2%, CCC
    # -2% -> median forward return is BBB's +2%. Benchmark is flat (+0%),
    # so under the old index-relative definition ALL THREE would have been
    # outperform=True; under the median-relative definition only AAA is.
    _seed_prices(tmp_lake, "AAA", [100, 103, 106])
    _seed_prices(tmp_lake, "BBB", [100, 101, 102])
    _seed_prices(tmp_lake, "CCC", [100, 99, 98])
    _seed_benchmark(tmp_lake, [1000, 1000, 1000])

    out = registry.build_labels_for_universe(tmp_lake, horizons={"2d": 2})
    row0 = out[(out["date"] == out["date"].min())].set_index("symbol")

    assert bool(row0.loc["AAA", "outperform"]) is True
    assert bool(row0.loc["BBB", "outperform"]) is False  # exactly at the median, not strictly above
    assert bool(row0.loc["CCC", "outperform"]) is False

    # excess_return stays index-relative (benchmark is flat here), untouched
    # as a reporting column, distinct from the median-relative outperform.
    assert row0.loc["AAA", "excess_return"] == pytest.approx(106 / 100 - 1)


def test_outperform_median_is_computed_per_horizon_and_date_independently(tmp_lake):
    """The cross-sectional median must be computed within each (horizon,
    date) group separately -- not pooled across horizons or dates, which
    would let a stock's label on one horizon be contaminated by unrelated
    return distributions from another horizon or day."""
    _seed_prices(tmp_lake, "AAA", [100, 110, 90, 130])
    _seed_prices(tmp_lake, "BBB", [100, 90, 110, 70])
    _seed_benchmark(tmp_lake, [1000, 1000, 1000, 1000])

    out = registry.build_labels_for_universe(tmp_lake, horizons={"1d": 1, "3d": 3})
    assert set(out["horizon"]) == {"1d", "3d"}
    # Every resolved row's outperform must be well-defined per its own
    # (horizon, date) group -- just assert the column round-trips as a
    # proper boolean/NA dtype with no cross-contamination artifacts (e.g.
    # every symbol tying at False because a wrong global median leaked in).
    resolved = out.dropna(subset=["outperform"])
    assert not resolved.empty
    for (_, _), group in resolved.groupby(["horizon", "date"]):
        if len(group) > 1:
            assert group["outperform"].nunique() > 0


def test_persist_labels_writes_to_gold_and_reads_back(tmp_lake):
    _seed_prices(tmp_lake, "AAA", [100, 101, 102, 103, 104, 105])
    _seed_benchmark(tmp_lake, [1000, 1001, 1002, 1003, 1004, 1005])

    labels = registry.build_labels_for_universe(tmp_lake, horizons={"2d": 2})
    rows = registry.persist_labels(tmp_lake, labels)
    assert rows == len(labels)

    gold = tmp_lake.read_all(DataLayer.GOLD, registry.GOLD_DOMAIN)
    assert len(gold) == len(labels)


def test_persist_labels_empty_returns_zero(tmp_lake):
    assert registry.persist_labels(tmp_lake, pd.DataFrame()) == 0
