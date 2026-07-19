from __future__ import annotations

from stockpredictor.reporting.dashboard import render_dashboard_html


def test_render_dashboard_html_empty_state(tmp_path):
    out_path = tmp_path / "dashboard" / "index.html"
    render_dashboard_html({"n_published": 0, "n_resolved": 0}, out_path)

    html = out_path.read_text(encoding="utf-8")
    assert "<html>" in html
    assert "none resolved yet" in html


def test_render_dashboard_html_with_data(tmp_path):
    analytics = {
        "n_published": 10,
        "n_resolved": 6,
        "overall_accuracy": 0.6,
        "top_5_accuracy": 0.8,
        "top_10_accuracy": 0.6,
        "avg_alpha": 0.02,
        "win_rate": 0.6,
        "avg_holding_return": 0.03,
        "monthly_stats": {
            "2024-01": {"n": 3, "hit_rate": 0.66, "avg_alpha": 0.01, "avg_return": 0.02},
            "2024-02": {"n": 3, "hit_rate": 0.33, "avg_alpha": -0.01, "avg_return": -0.01},
        },
        "quarterly_stats": {},
        "rolling_6m": {"n": 6, "hit_rate": 0.5, "avg_alpha": 0.0, "avg_return": 0.01},
        "rolling_12m": None,
        "by_horizon_ratios": {
            "90d": {"n": 6, "cagr": 0.15, "sharpe_ratio": 1.2, "sortino_ratio": 1.5, "max_drawdown": -0.1}
        },
        "probability_distribution": {"0.500-0.600": 4, "0.600-0.700": 2},
        "feature_drift": None,
    }
    out_path = tmp_path / "dashboard" / "index.html"
    render_dashboard_html(analytics, out_path)

    html = out_path.read_text(encoding="utf-8")
    assert "60.0%" in html  # overall accuracy
    assert "2024-01" in html
    assert "svg" in html
