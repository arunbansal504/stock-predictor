"""India-specific transaction cost model (§25).

Costs are approximated in basis points per config/backtest.yaml's schedule --
review against current NSE/SEBI rates before relying on backtest results for
real decisions; a hardcoded cost schedule can go stale exactly like a free
data feed can (§26).

A round trip (entering then exiting a position) does not pay every cost
symmetrically: brokerage, exchange charges, and slippage are incurred on
both legs; STT (securities transaction tax) is sell-side only for delivery
equity; stamp duty is buy-side only. Treating everything as "apply once,
double it" would overstate stamp duty/STT and silently make the backtest
look worse than reality -- equally dishonest as understating costs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    brokerage_bps: float = 3.0  # per leg
    stt_bps: float = 10.0  # sell-side only (delivery equity)
    exchange_txn_bps: float = 0.3  # per leg
    gst_bps: float = 1.8  # per leg (18% GST on brokerage+txn charges, approximated in bps)
    stamp_duty_bps: float = 1.5  # buy-side only
    slippage_bps: float = 10.0  # per leg

    def round_trip_cost_bps(self) -> float:
        """Total cost in bps for a full round trip (one entry + one exit)."""
        per_leg = self.brokerage_bps + self.exchange_txn_bps + self.gst_bps + self.slippage_bps
        one_sided = self.stt_bps + self.stamp_duty_bps
        return per_leg * 2 + one_sided

    @classmethod
    def from_config(cls, config: dict) -> "CostModel":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in valid_keys})


def net_of_costs(gross_return: pd.Series | float, cost_model: CostModel) -> pd.Series | float:
    """Subtract the round-trip cost fraction from a gross return (or series
    of gross returns). Costs are a flat bps drag per round trip, not scaled
    by position size beyond the assumption of a single full round trip per
    holding period -- consistent with the Top-K rebalance-at-horizon
    strategy in backtest/engine.py."""
    cost_fraction = cost_model.round_trip_cost_bps() / 10_000.0
    return gross_return - cost_fraction
