"""Hierarchical Risk Parity allocation (§12): "I recommend HRP because it's
robust to noisy covariance estimates (Markowitz notoriously over-
concentrates on estimation error)."

Implements Lopez de Prado's HRP (2016): (1) hierarchical clustering of
assets by correlation distance, (2) quasi-diagonalization of the
correlation matrix so similar assets sit next to each other in the
resulting order, (3) recursive bisection that allocates weight inversely
proportional to cluster variance at each split. Unlike Markowitz
mean-variance optimization, HRP never inverts the full covariance matrix --
that inversion is exactly what makes Markowitz blow up when the covariance
estimate is noisy, which is the normal situation here (a handful of months
of daily returns for a freshly ranked stock list, not years of stable
history).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def compute_returns_matrix(
    prices: pd.DataFrame, symbols: list[str], lookback_days: int = 90, max_gap_fill: int = 2
) -> pd.DataFrame:
    """Wide daily-return matrix (columns=symbols, rows=date) for the
    trailing `lookback_days`, from a long silver-prices-shaped frame
    (symbol, date, close_adj).

    Small, isolated gaps (up to `max_gap_fill` consecutive missing days --
    e.g. a single symbol's most-recent-day price not yet landing from a
    free/unofficial source, observed live: several otherwise 5-year-complete
    NSE symbols missing exactly one recent day) are forward-filled: "no new
    information, price unchanged" is a standard, defensible convention for
    an isolated illiquid/missing day, distinct from fabricating a new price
    level. Symbols with gaps *larger* than that (genuinely thin history,
    e.g. a stock listed within the lookback window, or an extended halt)
    are still dropped rather than imputed -- a fabricated multi-day trend
    would corrupt the covariance estimate every other weight depends on."""
    subset = prices[prices["symbol"].isin(symbols)]
    wide = subset.pivot(index="date", columns="symbol", values="close_adj").sort_index()
    wide = wide.tail(lookback_days + 1)  # +1 row since pct_change() loses the first one
    wide = wide.ffill(limit=max_gap_fill)
    returns = wide.pct_change().iloc[1:]
    returns = returns.dropna(axis=1, how="any")
    return returns


def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """d(i,j) = sqrt(0.5 * (1 - corr(i,j))) -- a proper metric distance
    (satisfies the triangle inequality, unlike 1-corr alone), per Lopez de
    Prado. Ranges 0 (perfectly correlated) to 1 (perfectly anti-correlated)."""
    return ((1 - corr) / 2.0) ** 0.5


def _quasi_diagonal_order(link: np.ndarray, n_leaves: int) -> list[int]:
    """Recover leaf order from a scipy linkage matrix such that similar
    assets end up adjacent -- recursively expand the root cluster into its
    leaf members in merge order."""
    link = link.astype(int)

    def _expand(cluster_id: int) -> list[int]:
        if cluster_id < n_leaves:
            return [cluster_id]
        left, right = link[cluster_id - n_leaves, 0], link[cluster_id - n_leaves, 1]
        return _expand(left) + _expand(right)

    root = 2 * n_leaves - 2
    return _expand(root)


def _cluster_variance(cov: pd.DataFrame, cluster_items: list) -> float:
    """Variance of a cluster's inverse-variance-weighted sub-portfolio --
    the standard HRP building block for comparing two sibling clusters at
    each bisection."""
    sub_cov = cov.loc[cluster_items, cluster_items]
    inv_var = 1.0 / np.diag(sub_cov)
    weights = inv_var / inv_var.sum()
    return float(weights @ sub_cov.values @ weights)


def _recursive_bisection(cov: pd.DataFrame, sorted_items: list) -> pd.Series:
    """Allocate weight top-down: split the quasi-diagonal-ordered item list
    in half repeatedly; at each split, the less-volatile half gets more
    weight, in proportion to the sibling clusters' relative variance.
    Self-normalizing to sum 1.0 by construction (each split preserves the
    combined weight of its two children), with a defensive final
    normalization against floating-point drift."""
    weights = pd.Series(1.0, index=sorted_items)
    cluster_stack = [sorted_items]

    while any(len(c) > 1 for c in cluster_stack):
        next_stack = []
        for cluster in cluster_stack:
            if len(cluster) <= 1:
                next_stack.append(cluster)
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)
            alpha = 1.0 - left_var / (left_var + right_var)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_stack.append(left)
            next_stack.append(right)
        cluster_stack = next_stack

    return weights / weights.sum()


def compute_hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Full HRP pipeline: correlation-distance clustering -> quasi-diagonal
    ordering -> recursive bisection. `returns` must be a wide frame
    (columns=symbols, rows=dates) with no missing values -- see
    compute_returns_matrix. Returns weights indexed by symbol, summing to 1."""
    if returns.shape[1] == 1:
        return pd.Series([1.0], index=returns.columns)

    corr = returns.corr()
    cov = returns.cov()

    dist = _correlation_distance(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")

    order = _quasi_diagonal_order(link, n_leaves=len(corr))
    sorted_items = list(corr.columns[order])

    weights = _recursive_bisection(cov, sorted_items)
    return weights.reindex(returns.columns)
