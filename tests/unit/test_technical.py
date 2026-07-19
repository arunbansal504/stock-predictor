from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.features import technical


def _price_frame(closes: list[float], symbol: str = "AAA", start: str = "2024-01-01") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    closes_arr = np.array(closes)
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "open": closes_arr,
            "high": closes_arr * 1.01,
            "low": closes_arr * 0.99,
            "close": closes_arr,
            "close_adj": closes_arr,
            "volume": np.full(n, 1000, dtype="int64"),
        }
    )


def test_adjust_ohlc_scales_open_high_low_by_split_ratio():
    # Simulate a 1:2 split on the 3rd bar: raw close jumps but close_adj is
    # continuous (the standard backward-adjustment convention).
    df = pd.DataFrame(
        {
            "open": [100.0, 102.0, 200.0],
            "high": [101.0, 103.0, 202.0],
            "low": [99.0, 101.0, 198.0],
            "close": [100.0, 102.0, 200.0],
            "close_adj": [50.0, 51.0, 100.0],  # pre-split rows halved, post-split unchanged
            "volume": [1000, 1100, 2000],
        }
    )
    out = technical._adjust_ohlc(df)
    # factor = close_adj / close: 0.5, 0.5, 0.5
    assert out["open_adj"].tolist() == pytest.approx([50.0, 51.0, 100.0])
    assert out["high_adj"].tolist() == pytest.approx([50.5, 51.5, 101.0])
    assert out["low_adj"].tolist() == pytest.approx([49.5, 50.5, 99.0])


def test_compute_returns_matches_manual_pct_change():
    df = _price_frame([100, 105, 110, 108, 115])
    out = technical.compute_returns(df, windows=(2,))
    expected = df["close_adj"].pct_change(2)
    assert out["return_2d"].tolist() == pytest.approx(expected.tolist(), nan_ok=True)


def test_rsi_is_bounded_between_0_and_100():
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0, 1, 200))
    df = _price_frame(list(closes))
    out = technical.compute_rsi(df)
    valid = out["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_near_100_for_strictly_increasing_prices():
    df = _price_frame(list(range(100, 140)))  # strictly increasing, no losses
    out = technical.compute_rsi(df)
    assert out["rsi_14"].iloc[-1] > 95


def test_atr_is_non_negative():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 2, 100))
    df = _price_frame(list(closes))
    out = technical.compute_atr(df)
    valid = out["atr_14"].dropna()
    assert (valid >= 0).all()


def test_realized_vol_is_zero_for_constant_price():
    df = _price_frame([100.0] * 80)
    out = technical.compute_realized_vol(df)
    valid = out["realized_vol_20d"].dropna()
    assert (valid.abs() < 1e-9).all()


def test_obv_increases_on_up_days_and_decreases_on_down_days():
    df = _price_frame([100, 105, 103, 108])  # up, down, up
    out = technical.compute_obv(df)
    obv = out["obv"]
    assert obv.iloc[1] > obv.iloc[0]  # up day: +volume
    assert obv.iloc[2] < obv.iloc[1]  # down day: -volume
    assert obv.iloc[3] > obv.iloc[2]  # up day: +volume


def test_volume_zscore_nan_before_window_then_populated():
    df = _price_frame(list(range(100, 100 + 30)))
    df["volume"] = np.concatenate([np.full(29, 1000), [5000]])
    out = technical.compute_volume_zscore(df, window=20)
    assert out["volume_zscore_20d"].iloc[:19].isna().all()
    assert out["volume_zscore_20d"].iloc[-1] > 0  # spike above rolling mean


def test_adjust_ohlc_scales_volume_inversely_to_the_price_factor():
    df = pd.DataFrame(
        {
            "open": [200.0, 100.0],
            "high": [200.0, 100.0],
            "low": [200.0, 100.0],
            "close": [200.0, 100.0],  # pre-split, post-split
            "close_adj": [100.0, 100.0],  # continuous
            "volume": [1000, 2000],  # raw share count roughly doubles at the split
        }
    )
    out = technical._adjust_ohlc(df)
    # factor = close_adj/close = [0.5, 1.0]; volume_adj = volume/factor.
    assert out["volume_adj"].tolist() == pytest.approx([2000.0, 2000.0])


def test_obv_accumulates_split_adjusted_volume_not_raw_volume():
    """Hand-computed regression check: with a split factor that changes
    from 0.5 (pre-split) to 1.0 (post-split) partway through, and a chosen
    volume_adj series, obv must match cumsum(direction * volume_adj) -- not
    direction * raw volume, which gives a different, wrong answer (a 1:2
    split roughly doubles post-split raw volume for the same real
    activity, see the module docstring)."""
    dates = pd.bdate_range("2024-01-01", periods=4)
    close_adj = [100.0, 102.0, 105.0, 103.0]
    factor = [0.5, 0.5, 1.0, 1.0]
    close = [c / f for c, f in zip(close_adj, factor)]
    volume_adj = [500.0, 600.0, 700.0, 800.0]
    volume = [round(v * f) for v, f in zip(volume_adj, factor)]
    df = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "close_adj": close_adj,
            "volume": volume,
        }
    )
    out = technical.compute_obv(df)
    # direction = [0 (first row), +1, +1, -1]; obv = cumsum(direction * volume_adj).
    assert out["obv"].tolist() == pytest.approx([0.0, 600.0, 1300.0, 500.0])


def test_volume_zscore_has_no_spurious_spike_at_a_split_boundary():
    """Regression guard for the split-adjustment fix: real trading activity
    (volume_adj) is flat (small noise) straddling a 1:2 split, so a
    correctly adjusted z-score at the first post-split day is an ordinary
    value -- not the outlier spike raw (unadjusted) volume shows purely
    from the share-count doubling, with nothing to do with real activity."""
    rng = np.random.default_rng(3)
    n = 44
    split_at = 22
    noise = rng.normal(0, 20, n)
    volume_adj = 1000.0 + noise
    factor = np.where(np.arange(n) < split_at, 0.5, 1.0)
    volume = np.round(volume_adj * factor).astype("int64")
    close_adj = np.full(n, 100.0)  # flat adjusted price -- isolate volume's effect
    close = close_adj / factor
    df = pd.DataFrame(
        {
            "symbol": ["AAA"] * n,
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "close_adj": close_adj,
            "volume": volume,
        }
    )
    out = technical.compute_volume_zscore(df, window=20)
    z_at_split = out["volume_zscore_20d"].iloc[split_at]
    assert abs(z_at_split) < 2, f"expected an ordinary z-score at the split boundary, got {z_at_split}"

    # Sanity check the fixture actually exercises the bug this guards
    # against: the same rolling formula on raw (unadjusted) volume must
    # show a real outlier here, or this scenario isn't testing anything.
    raw = pd.Series(volume, dtype="float64")
    raw_z_at_split = (raw.iloc[split_at] - raw.rolling(20).mean().iloc[split_at]) / raw.rolling(20).std().iloc[
        split_at
    ]
    assert abs(raw_z_at_split) > 3, "fixture didn't actually create a raw-volume outlier at the split"


def test_compute_technical_features_end_to_end_shape_and_pit_stamp():
    df = _price_frame(list(100 + np.cumsum(np.random.default_rng(7).normal(0, 1, 300))))
    out = technical.compute_technical_features(df)
    assert len(out) == len(df)
    assert (out["knowable_date"] == out["date"]).all()  # same-day knowability
    assert "rsi_14" in out.columns
    assert "macd" in out.columns
    assert (out["symbol"] == "AAA").all()
