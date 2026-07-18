from __future__ import annotations

import pytest

from stockpredictor.backtest.costs import CostModel, net_of_costs


def test_round_trip_cost_matches_manual_per_leg_vs_one_sided_calc():
    cm = CostModel(
        brokerage_bps=3.0,
        stt_bps=10.0,
        exchange_txn_bps=0.3,
        gst_bps=1.8,
        stamp_duty_bps=1.5,
        slippage_bps=10.0,
    )
    per_leg = 3.0 + 0.3 + 1.8 + 10.0  # brokerage + exchange + gst + slippage, x2 legs
    one_sided = 10.0 + 1.5  # stt (sell-only) + stamp duty (buy-only), x1
    expected = per_leg * 2 + one_sided
    assert cm.round_trip_cost_bps() == pytest.approx(expected)


def test_from_config_ignores_unknown_keys():
    cm = CostModel.from_config({"brokerage_bps": 5.0, "unrelated_key": "ignored"})
    assert cm.brokerage_bps == 5.0
    assert cm.stt_bps == CostModel().stt_bps  # untouched default


def test_net_of_costs_subtracts_round_trip_fraction():
    cm = CostModel(
        brokerage_bps=0, stt_bps=0, exchange_txn_bps=0, gst_bps=0, stamp_duty_bps=0, slippage_bps=50.0
    )
    # round trip = 50*2 = 100bps = 1%
    net = net_of_costs(0.10, cm)
    assert net == pytest.approx(0.09)


def test_net_of_costs_works_on_a_series():
    import pandas as pd

    cm = CostModel(brokerage_bps=0, stt_bps=0, exchange_txn_bps=0, gst_bps=0, stamp_duty_bps=0, slippage_bps=50.0)
    gross = pd.Series([0.10, 0.20, -0.05])
    net = net_of_costs(gross, cm)
    assert net.tolist() == pytest.approx([0.09, 0.19, -0.06])
