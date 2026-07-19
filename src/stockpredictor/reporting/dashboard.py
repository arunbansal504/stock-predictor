"""Static HTML performance dashboard (ML Review Board spec Part 3).

Plain-Python string templating -- no new dependency (jinja2/plotly aren't
needed for a handful of stat tiles, two bar charts, and data tables), which
matches this repo's "earn its place" minimal-dependency posture (see
storage/models.py's docstring). Regenerated in place on every daily
validation run (unlike predictions/reports, which are append-only) -- see
scripts/run_daily_validation.py.

Colors/marks follow the dataviz skill's method: fixed categorical slot 1
(blue) for the single-series probability histogram, status tokens
(good/critical) for the monthly-alpha bars since that series literally means
"beat the benchmark or not" (the skill's collision rule: a series that means
good/bad wears status tokens, not categorical), stat-tile deltas colored by
direction x whether-up-is-good, dark mode via both `prefers-color-scheme`
and a `data-theme` override, thin 2px-gapped bars with a native `<title>`
hover tooltip, and a full data table under every chart (the table-view
accessibility twin).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# Palette roles from the dataviz skill's references/palette.md -- categorical
# slot 1 (blue) and the fixed status scale, light + dark steps.
_SERIES_1_LIGHT, _SERIES_1_DARK = "#2a78d6", "#3987e5"
_GOOD_LIGHT, _GOOD_DARK = "#0ca30c", "#0ca30c"
_CRITICAL_LIGHT, _CRITICAL_DARK = "#d03b3b", "#e66767"

_STYLE = f"""
:root {{
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
  --text-secondary: #52514e; --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
  --series-1: {_SERIES_1_LIGHT}; --good: {_GOOD_LIGHT}; --critical: {_CRITICAL_LIGHT};
  --border: rgba(11,11,11,0.10);
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
    --series-1: {_SERIES_1_DARK}; --good: {_GOOD_DARK}; --critical: {_CRITICAL_DARK};
    --border: rgba(255,255,255,0.10);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
  --series-1: {_SERIES_1_DARK}; --good: {_GOOD_DARK}; --critical: {_CRITICAL_DARK};
  --border: rgba(255,255,255,0.10);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 32px; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
.subtitle {{ color: var(--text-secondary); margin: 0 0 28px; font-size: 14px; }}
.card {{
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px; margin-bottom: 20px;
}}
.card h2 {{ font-size: 15px; margin: 0 0 14px; color: var(--text-secondary); font-weight: 600; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; }}
.tile-value {{ font-size: 26px; font-weight: 600; }}
.tile-label {{ font-size: 12px; color: var(--text-secondary); margin-top: 2px; }}
.tile-good {{ color: var(--good); }}
.tile-critical {{ color: var(--critical); }}
svg text {{ fill: var(--text-secondary); font-size: 11px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--text-secondary); font-weight: 600; }}
td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
.empty {{ color: var(--muted); font-size: 13px; }}
"""


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_num(value: float | None, decimals: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def _stat_tile(label: str, value: str, css_class: str = "") -> str:
    return f'<div><div class="tile-value {css_class}">{value}</div><div class="tile-label">{label}</div></div>'


def _bar_chart(labels: list[str], values: list[float], color_for: Callable[[float], str]) -> str:
    """A thin, gapped horizontal bar chart with a native-tooltip hover and a
    zero baseline -- values can be negative (monthly alpha), so bars grow
    from a shared zero line rather than the plot's edge."""
    if not labels:
        return '<p class="empty">No resolved data yet.</p>'

    width, row_h, gap, label_w = 720, 22, 4, 90
    plot_w = width - label_w - 60
    max_abs = max(abs(v) for v in values) or 1.0
    zero_x = label_w + plot_w / 2
    scale = (plot_w / 2) / max_abs
    height = len(labels) * (row_h + gap) + gap

    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = gap + i * (row_h + gap)
        bar_w = abs(value) * scale
        x = zero_x if value >= 0 else zero_x - bar_w
        color = color_for(value)
        bars.append(
            f'<text x="{label_w - 8}" y="{y + row_h / 2 + 4}" text-anchor="end">{label}</text>'
            f'<rect x="{x:.1f}" y="{y}" width="{max(bar_w, 1):.1f}" height="{row_h}" '
            f'rx="4" fill="{color}"><title>{label}: {value:+.4f}</title></rect>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">'
        f'<line x1="{zero_x}" y1="0" x2="{zero_x}" y2="{height}" stroke="var(--baseline)" stroke-width="1"/>'
        + "".join(bars)
        + "</svg>"
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="empty">No data yet.</p>'
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f'<td class="num">{c}</td>' for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_dashboard_html(analytics: dict, out_path: Path) -> None:
    """Render `analytics` (reporting.analytics.compute_performance_analytics'
    output) to a self-contained static HTML file at `out_path`."""
    if analytics.get("n_resolved", 0) == 0:
        body = (
            "<h1>Stock Predictor -- Performance Dashboard</h1>"
            f'<p class="subtitle">{analytics.get("n_published", 0)} predictions published, '
            "none resolved yet -- check back after the first horizon completes.</p>"
        )
        _write(out_path, body)
        return

    tiles = "".join(
        [
            _stat_tile("Overall accuracy", _fmt_pct(analytics["overall_accuracy"])),
            _stat_tile("Top-5 accuracy", _fmt_pct(analytics["top_5_accuracy"])),
            _stat_tile("Top-10 accuracy", _fmt_pct(analytics["top_10_accuracy"])),
            _stat_tile(
                "Avg alpha",
                _fmt_pct(analytics["avg_alpha"]),
                "tile-good" if analytics["avg_alpha"] > 0 else "tile-critical",
            ),
            _stat_tile("Win rate", _fmt_pct(analytics["win_rate"])),
            _stat_tile("Avg holding return", _fmt_pct(analytics["avg_holding_return"])),
        ]
    )

    monthly = analytics.get("monthly_stats", {})
    months = list(monthly.keys())
    alpha_chart = _bar_chart(
        months,
        [monthly[m]["avg_alpha"] for m in months],
        color_for=lambda v: "var(--good)" if v >= 0 else "var(--critical)",
    )
    monthly_table = _table(
        ["Month", "N", "Hit rate", "Avg alpha", "Avg return"],
        [
            [m, str(monthly[m]["n"]), _fmt_pct(monthly[m]["hit_rate"]), _fmt_pct(monthly[m]["avg_alpha"]), _fmt_pct(monthly[m]["avg_return"])]
            for m in months
        ],
    )

    prob_dist = analytics.get("probability_distribution", {})
    prob_labels = list(prob_dist.keys())
    prob_chart = _bar_chart(
        prob_labels, [float(v) for v in prob_dist.values()], color_for=lambda _v: "var(--series-1)"
    )

    ratios_rows = []
    for horizon, r in analytics.get("by_horizon_ratios", {}).items():
        ratios_rows.append(
            [horizon, str(r["n"]), _fmt_pct(r["cagr"]), _fmt_num(r["sharpe_ratio"]), _fmt_num(r["sortino_ratio"]), _fmt_pct(r["max_drawdown"])]
        )
    ratios_table = _table(["Horizon", "N", "CAGR", "Sharpe", "Sortino", "Max drawdown"], ratios_rows)

    rolling = "".join(
        [
            _rolling_tile("Rolling 6-month", analytics.get("rolling_6m")),
            _rolling_tile("Rolling 12-month", analytics.get("rolling_12m")),
        ]
    )

    body = f"""
<h1>Stock Predictor -- Performance Dashboard</h1>
<p class="subtitle">{analytics['n_published']} predictions published, {analytics['n_resolved']} resolved.</p>

<div class="card"><h2>Headline accuracy</h2><div class="tiles">{tiles}</div></div>
<div class="card"><h2>Rolling performance</h2><div class="tiles">{rolling}</div></div>
<div class="card"><h2>Monthly avg alpha (green = beat benchmark, red = missed)</h2>{alpha_chart}{monthly_table}</div>
<div class="card"><h2>Risk-adjusted return by horizon</h2>{ratios_table}</div>
<div class="card"><h2>Prediction probability distribution</h2>{prob_chart}</div>
"""
    _write(out_path, body)


def _rolling_tile(label: str, window: dict | None) -> str:
    if window is None:
        return _stat_tile(label, "n/a")
    return _stat_tile(f"{label} hit rate (n={window['n']})", _fmt_pct(window["hit_rate"]))


def _write(out_path: Path, body: str) -> None:
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>Stock Predictor Dashboard</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
