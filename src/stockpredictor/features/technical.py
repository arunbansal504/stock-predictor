"""Technical feature block (§7): a curated ~20-feature set covering momentum,
trend, oscillators, volatility, and volume — not the exhaustive 40-indicator
wishlist from the original brief. Each additional indicator must prove its
worth out-of-sample before joining this set (architecture doc Truth 3).

All price-based indicators are computed on **split/dividend-adjusted** OHLC,
not raw OHLC. yfinance only gives us an adjusted close (`close_adj`); this
module derives an adjustment factor (`close_adj / close`) and applies it to
open/high/low too, so ATR/Bollinger/RSI don't see a fake gap at every split —
the same correctness concern ingestion/prices.py raises for raw returns
applies equally to every indicator built on OHLC.

Known limitation, documented rather than silently ignored: raw `volume` is
NOT split-adjusted here (a 1:2 split roughly doubles post-split share volume
for the same traded value). Rolling volume features computed on a window
that straddles a split date will show a level shift. Acceptable for MVP;
revisit if volume features prove valuable enough to be worth the extra
adjustment logic.

Every function takes a single symbol's silver price frame (columns: date,
open, high, low, close, close_adj, volume — see ingestion/prices.py) sorted
ascending by date, and returns engineered columns of the same length,
NaN-padded until a rolling window has enough history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

RETURN_WINDOWS = (5, 20, 60, 120)
VOL_WINDOWS = (20, 60)


def _adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    factor = out["close_adj"] / out["close"].replace(0, np.nan)
    out["open_adj"] = out["open"] * factor
    out["high_adj"] = out["high"] * factor
    out["low_adj"] = out["low"] * factor
    return out


def compute_returns(df: pd.DataFrame, windows: tuple[int, ...] = RETURN_WINDOWS) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"return_{w}d"] = df["close_adj"].pct_change(w)
    return out


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close_adj"]
    out = pd.DataFrame(index=df.index)
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["price_vs_sma20"] = close / out["sma_20"] - 1.0
    out["price_vs_sma50"] = close / out["sma_50"] - 1.0
    return out


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close_adj"]
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    out = pd.DataFrame(index=df.index)
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal
    return out


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    close = df["close_adj"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Edge cases the RS ratio can't express directly: an all-gains window
    # (avg_loss == 0) is maximally overbought -> RSI 100, not NaN. Symmetric
    # for an all-losses window. Insufficient history (avg_gain/avg_loss still
    # NaN from the ewm min_periods warmup) is left as NaN, not defaulted to a
    # neutral 50 -- a silent fabricated value would be worse than a gap.
    rsi = rsi.mask((avg_loss == 0) & avg_gain.notna(), 100.0)
    rsi = rsi.mask((avg_gain == 0) & avg_loss.notna(), 0.0)
    out = pd.DataFrame(index=df.index)
    out[f"rsi_{period}"] = rsi
    return out


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    adj = _adjust_ohlc(df)
    prev_close = adj["close_adj"].shift(1)
    tr = pd.concat(
        [
            adj["high_adj"] - adj["low_adj"],
            (adj["high_adj"] - prev_close).abs(),
            (adj["low_adj"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    out = pd.DataFrame(index=df.index)
    out[f"atr_{period}"] = atr
    out[f"atr_{period}_pct"] = atr / adj["close_adj"]
    return out


def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    close = df["close_adj"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    out = pd.DataFrame(index=df.index)
    out["bb_pctb"] = (close - lower) / (upper - lower).replace(0, np.nan)
    out["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    return out


def compute_realized_vol(df: pd.DataFrame, windows: tuple[int, ...] = VOL_WINDOWS) -> pd.DataFrame:
    log_ret = np.log(df["close_adj"] / df["close_adj"].shift(1))
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"realized_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return out


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["close_adj"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()
    out = pd.DataFrame(index=df.index)
    out["obv"] = obv
    return out


def compute_52w_range(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close_adj"]
    window = TRADING_DAYS_PER_YEAR
    high_52w = close.rolling(window, min_periods=20).max()
    low_52w = close.rolling(window, min_periods=20).min()
    out = pd.DataFrame(index=df.index)
    out["dist_from_52w_high"] = close / high_52w - 1.0
    out["dist_from_52w_low"] = close / low_52w - 1.0
    return out


def compute_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    vol = df["volume"].astype("float64")
    mean = vol.rolling(window).mean()
    std = vol.rolling(window).std()
    out = pd.DataFrame(index=df.index)
    out[f"volume_zscore_{window}d"] = (vol - mean) / std.replace(0, np.nan)
    return out


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full technical block for one symbol's silver price frame.
    Returns a frame keyed by (symbol, date) with `knowable_date == date`
    (same-day knowability, same as the underlying price data — see
    common/pit.py) and every engineered column, NaN where history is
    insufficient for a rolling window."""
    df = df.sort_values("date").reset_index(drop=True)
    blocks = [
        compute_returns(df),
        compute_moving_averages(df),
        compute_macd(df),
        compute_rsi(df),
        compute_atr(df),
        compute_bollinger(df),
        compute_realized_vol(df),
        compute_obv(df),
        compute_52w_range(df),
        compute_volume_zscore(df),
    ]
    features = pd.concat(blocks, axis=1)
    features.insert(0, "symbol", df["symbol"])
    features.insert(1, "date", df["date"])
    features["knowable_date"] = df["date"]
    return features
