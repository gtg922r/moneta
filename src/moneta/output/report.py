"""Interactive Plotly HTML report generator for Moneta.

Generates a self-contained HTML file with:
- Fan charts (p10/p25/p50/p75/p90 bands) for each asset
- Distribution histograms for distribution queries
- Sample simulation paths overlaid on fan charts
- Probability timelines for probability queries
- Sweep mode comparison with overlaid charts
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from moneta.engine.state import ResultStore
from moneta.parser.models import ScenarioConfig
from moneta.query.engine import QueryResult
from moneta.query.expressions import evaluate, parse_expression

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Colors for fan chart bands (lighter fill for wider bands)
_FAN_COLORS = {
    "p10_p90": "rgba(31, 119, 180, 0.1)",
    "p25_p75": "rgba(31, 119, 180, 0.25)",
    "p50_line": "rgba(31, 119, 180, 1.0)",
}

# Colors for sweep mode (one per scenario)
_SWEEP_COLORS = [
    "rgba(31, 119, 180, {alpha})",   # blue
    "rgba(255, 127, 14, {alpha})",   # orange
    "rgba(44, 160, 44, {alpha})",    # green
    "rgba(214, 39, 40, {alpha})",    # red
    "rgba(148, 103, 189, {alpha})",  # purple
    "rgba(140, 86, 75, {alpha})",    # brown
    "rgba(227, 119, 194, {alpha})",  # pink
    "rgba(127, 127, 127, {alpha})",  # gray
]

_SAMPLE_PATH_COLOR = "rgba(150, 150, 150, 0.4)"


# ---------------------------------------------------------------------------
# Data computation helpers
# ---------------------------------------------------------------------------


def _compute_fan_chart_data(results: ResultStore, asset_idx: int) -> dict:
    """Compute percentile bands for a single asset over time.

    For each time step, computes p10, p25, p50, p75, p90 of the asset
    balance across all simulation runs.

    Args:
        results: Completed simulation ResultStore.
        asset_idx: Column index of the asset in the balances array.

    Returns:
        Dict with keys 'years', 'p10', 'p25', 'p50', 'p75', 'p90',
        each a 1D numpy array of length n_steps.
    """
    # balances shape: (n_runs, n_steps, n_assets)
    asset_data = results.balances[:, :, asset_idx]  # (n_runs, n_steps)

    # Compute percentiles across runs at each time step
    percentiles = np.percentile(
        asset_data, [10, 25, 50, 75, 90], axis=0
    )  # shape: (5, n_steps)

    # Convert month indices (0-based) to years
    years = (np.arange(results.n_steps) + 1) / 12.0

    return {
        "years": years,
        "p10": percentiles[0],
        "p25": percentiles[1],
        "p50": percentiles[2],
        "p75": percentiles[3],
        "p90": percentiles[4],
    }


def _select_sample_paths(
    results: ResultStore,
    asset_idx: int,
    n_samples: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Select random sample paths for a given asset.

    Args:
        results: Completed simulation ResultStore.
        asset_idx: Column index of the asset.
        n_samples: Number of sample paths to select.
        seed: Seed for reproducible sample selection.

    Returns:
        Array of shape (n_samples, n_steps) with the selected paths.
    """
    rng = np.random.default_rng(seed)
    n_runs = results.n_runs
    n_samples = min(n_samples, n_runs)
    indices = rng.choice(n_runs, size=n_samples, replace=False)
    return results.balances[indices, :, asset_idx]


# ---------------------------------------------------------------------------
# Chart creation helpers
# ---------------------------------------------------------------------------


def _create_fan_chart(
    fan_data: dict,
    asset_name: str,
    sample_paths: np.ndarray | None = None,
    color_idx: int = 0,
    show_legend_prefix: str | None = None,
) -> go.Figure:
    """Create a fan chart figure with optional sample paths.

    Args:
        fan_data: Dict from _compute_fan_chart_data with percentile bands.
        asset_name: Name of the asset for the chart title.
        sample_paths: Optional (n_samples, n_steps) array of sample paths.
        color_idx: Color index for sweep mode (default 0 = blue).
        show_legend_prefix: If set, prefix legend entries (for sweep overlay).

    Returns:
        A plotly Figure with the fan chart.
    """
    fig = go.Figure()
    years = fan_data["years"]

    # Determine colors
    if show_legend_prefix is not None and color_idx < len(_SWEEP_COLORS):
        color_template = _SWEEP_COLORS[color_idx]
        band_outer = color_template.format(alpha="0.1")
        band_inner = color_template.format(alpha="0.25")
        line_color = color_template.format(alpha="1.0")
    else:
        band_outer = _FAN_COLORS["p10_p90"]
        band_inner = _FAN_COLORS["p25_p75"]
        line_color = _FAN_COLORS["p50_line"]

    legend_prefix = f"{show_legend_prefix} " if show_legend_prefix else ""

    # p10-p90 band (outermost)
    fig.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([fan_data["p90"], fan_data["p10"][::-1]]),
        fill="toself",
        fillcolor=band_outer,
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{legend_prefix}p10-p90",
        showlegend=True,
        hoverinfo="skip",
    ))

    # p25-p75 band (inner)
    fig.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([fan_data["p75"], fan_data["p25"][::-1]]),
        fill="toself",
        fillcolor=band_inner,
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{legend_prefix}p25-p75",
        showlegend=True,
        hoverinfo="skip",
    ))

    # p50 median line
    fig.add_trace(go.Scatter(
        x=years,
        y=fan_data["p50"],
        mode="lines",
        line=dict(color=line_color, width=2),
        name=f"{legend_prefix}Median (p50)",
    ))

    # Sample paths
    if sample_paths is not None:
        for i in range(sample_paths.shape[0]):
            fig.add_trace(go.Scatter(
                x=years,
                y=sample_paths[i],
                mode="lines",
                line=dict(color=_SAMPLE_PATH_COLOR, width=1),
                name=f"Sample {i + 1}",
                showlegend=(i == 0),
                legendgroup="samples",
            ))

    fig.update_layout(
        title=f"{asset_name} \u2014 Value Distribution",
        xaxis_title="Years",
        yaxis_title="Value ($)",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def _create_histogram(
    values: np.ndarray,
    title: str,
    bins: int = 50,
) -> go.Figure:
    """Create a distribution histogram.

    Args:
        values: 1D array of outcome values.
        title: Chart title.
        bins: Number of histogram bins.

    Returns:
        A plotly Figure with the histogram.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=bins,
        marker_color="rgba(31, 119, 180, 0.7)",
        marker_line_color="rgba(31, 119, 180, 1.0)",
        marker_line_width=1,
        name="Outcomes",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Value ($)",
        yaxis_title="Count",
        template="plotly_white",
        bargap=0.05,
    )

    return fig


def _create_probability_timeline(
    results: ResultStore,
    expression_str: str,
) -> go.Figure:
    """Show how a probability changes over time.

    Evaluates the probability expression at every time step, not just
    the queried point, to show the full trajectory.

    Args:
        results: Completed simulation ResultStore.
        expression_str: The probability expression (e.g. "portfolio > 2000000").

    Returns:
        A plotly Figure with the probability timeline.
    """
    node = parse_expression(expression_str)
    years = (np.arange(results.n_steps) + 1) / 12.0
    probabilities = np.empty(results.n_steps, dtype=np.float64)

    for step in range(results.n_steps):
        # Build values dict for this step
        values: dict[str, np.ndarray] = {}
        for name, idx in results.asset_index.items():
            values[name] = results.balances[:, step, idx]

        result_arr = evaluate(node, values)
        probabilities[step] = float(np.mean(result_arr.astype(np.float64))) * 100.0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years,
        y=probabilities,
        mode="lines",
        line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
        name="Probability",
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.1)",
    ))

    fig.update_layout(
        title=f"Probability Timeline: {expression_str}",
        xaxis_title="Years",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# Sweep comparison helpers
# ---------------------------------------------------------------------------


def _create_sweep_fan_chart(
    sweep_results: list[tuple[str, ResultStore, list[QueryResult]]],
    asset_name: str,
    asset_idx: int,
) -> go.Figure:
    """Create overlaid fan charts from multiple sweep scenarios.

    Args:
        sweep_results: List of (label, ResultStore, QueryResults) per scenario.
        asset_name: Name of the asset.
        asset_idx: Column index of the asset.

    Returns:
        A plotly Figure with overlaid fan chart bands.
    """
    fig = go.Figure()

    for i, (label, store, _qr) in enumerate(sweep_results):
        fan_data = _compute_fan_chart_data(store, asset_idx)
        years = fan_data["years"]
        color_idx = i % len(_SWEEP_COLORS)
        color_template = _SWEEP_COLORS[color_idx]

        band_outer = color_template.format(alpha="0.1")
        band_inner = color_template.format(alpha="0.2")
        line_color = color_template.format(alpha="1.0")

        # p10-p90 band
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([fan_data["p90"], fan_data["p10"][::-1]]),
            fill="toself",
            fillcolor=band_outer,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{label} p10-p90",
            legendgroup=label,
            showlegend=True,
            hoverinfo="skip",
        ))

        # p25-p75 band
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([fan_data["p75"], fan_data["p25"][::-1]]),
            fill="toself",
            fillcolor=band_inner,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{label} p25-p75",
            legendgroup=label,
            showlegend=True,
            hoverinfo="skip",
        ))

        # p50 line
        fig.add_trace(go.Scatter(
            x=years,
            y=fan_data["p50"],
            mode="lines",
            line=dict(color=line_color, width=2),
            name=f"{label} Median",
            legendgroup=label,
        ))

    fig.update_layout(
        title=f"{asset_name} \u2014 Scenario Comparison",
        xaxis_title="Years",
        yaxis_title="Value ($)",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def _create_sweep_summary_html(
    sweep_results: list[tuple[str, ResultStore, list[QueryResult]]],
) -> str:
    """Create an HTML comparison summary table for sweep scenarios.

    Args:
        sweep_results: List of (label, ResultStore, QueryResults) per scenario.

    Returns:
        HTML string with a comparison table.
    """
    if not sweep_results:
        return ""

    # Collect all query labels from the first scenario
    _, _, first_qr = sweep_results[0]
    if not first_qr:
        return ""

    rows = []
    header_labels = [label for label, _, _ in sweep_results]

    for qi, qr_first in enumerate(first_qr):
        row_values = []
        for _label, _store, qr_list in sweep_results:
            if qi < len(qr_list):
                qr = qr_list[qi]
                if qr.query_type == "probability" and qr.probability is not None:
                    row_values.append(f"{qr.probability:.1f}%")
                elif qr.query_type == "expected" and qr.median is not None:
                    row_values.append(f"${qr.median:,.0f}")
                elif qr.query_type == "percentiles" and qr.percentiles is not None:
                    # Show p50 for the last time point
                    last_time = max(qr.percentiles.keys())
                    p50 = qr.percentiles[last_time].get(50, 0)
                    row_values.append(f"${p50:,.0f}")
                else:
                    row_values.append("--")
            else:
                row_values.append("--")
        rows.append((qr_first.label, row_values))

    # Build HTML table
    html = '<div style="margin: 20px; font-family: sans-serif;">\n'
    html += "<h2>Scenario Comparison</h2>\n"
    html += '<table style="border-collapse: collapse; width: 100%;">\n'
    html += "<thead><tr>"
    html += '<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Query</th>'
    for label in header_labels:
        html += f'<th style="border: 1px solid #ddd; padding: 8px; text-align: right;">{label}</th>'
    html += "</tr></thead>\n<tbody>\n"

    for query_label, values in rows:
        html += "<tr>"
        html += f'<td style="border: 1px solid #ddd; padding: 8px;">{query_label}</td>'
        for val in values:
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{val}</td>'
        html += "</tr>\n"

    html += "</tbody></table>\n</div>\n"
    return html


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------


def generate_report(
    results: ResultStore,
    query_results: list[QueryResult],
    scenario_config: ScenarioConfig,
    output_path: Path,
    sweep_results: list[tuple[str, ResultStore, list[QueryResult]]] | None = None,
) -> Path:
    """Generate an interactive HTML report with Plotly charts.

    Creates a self-contained HTML file with fan charts, histograms,
    probability timelines, sample paths, and (for sweep mode) overlaid
    comparison charts.

    Args:
        results: Completed simulation ResultStore (base scenario).
        query_results: Evaluated query results for the base scenario.
        scenario_config: Scenario configuration (name, horizon, etc.).
        output_path: Path to write the HTML file.
        sweep_results: Optional list of (label, ResultStore, QueryResults)
            for sweep mode. When provided, generates comparison charts.

    Returns:
        The path to the generated report file.
    """
    import plotly.io as pio

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures_html: list[str] = []

    # Title section
    title_html = (
        '<div style="font-family: sans-serif; padding: 20px;">'
        f"<h1>Moneta Report: {scenario_config.name}</h1>"
        f"<p>{results.n_runs:,} simulations &middot; "
        f"{scenario_config.time_horizon / 12:.0f} year horizon</p>"
        "</div>"
    )
    figures_html.append(title_html)

    if sweep_results is not None and len(sweep_results) > 0:
        # --- Sweep mode ---
        # Overlaid fan charts per asset
        for asset_name, asset_idx in results.asset_index.items():
            fig = _create_sweep_fan_chart(sweep_results, asset_name, asset_idx)
            figures_html.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

        # Comparison summary table
        summary = _create_sweep_summary_html(sweep_results)
        if summary:
            figures_html.append(summary)

    else:
        # --- Single scenario mode ---
        # Fan charts for each asset
        for asset_name, asset_idx in results.asset_index.items():
            fan_data = _compute_fan_chart_data(results, asset_idx)
            sample_paths = _select_sample_paths(results, asset_idx, n_samples=5, seed=0)
            fig = _create_fan_chart(fan_data, asset_name, sample_paths=sample_paths)
            figures_html.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Distribution histograms from query results
    for qr in query_results:
        if qr.query_type == "distribution" and qr.histogram_bins is not None:
            # Reconstruct approximate values from bins for a plotly histogram
            bin_centers = (qr.histogram_bins[:-1] + qr.histogram_bins[1:]) / 2.0
            fig = _create_histogram(bin_centers, title=qr.label, bins=len(qr.histogram_counts))
            # Use a bar chart based on precomputed histogram instead
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=bin_centers,
                y=qr.histogram_counts,
                marker_color="rgba(31, 119, 180, 0.7)",
                marker_line_color="rgba(31, 119, 180, 1.0)",
                marker_line_width=1,
                name="Outcomes",
                width=(qr.histogram_bins[1] - qr.histogram_bins[0]) * 0.9,
            ))
            fig.update_layout(
                title=qr.label,
                xaxis_title="Value ($)",
                yaxis_title="Count",
                template="plotly_white",
                bargap=0.05,
            )
            figures_html.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Probability timelines
    for qr in query_results:
        if qr.query_type == "probability":
            # Find the expression from the label or query
            # The expression is stored in the label if no label was set,
            # otherwise we need to look at the original query. Since we
            # only have the QueryResult, use label as a best-effort indicator.
            # For the timeline, we iterate the expression over all time steps.
            # We store the expression in the label when no label is provided.
            # For now, try to parse the label as an expression; if it fails,
            # skip the timeline.
            try:
                fig = _create_probability_timeline(results, qr.label)
                figures_html.append(
                    pio.to_html(fig, full_html=False, include_plotlyjs=False)
                )
            except Exception:
                # Label may not be a valid expression (e.g., "$2M at year 10").
                # In that case, skip the timeline for this query.
                pass

    # Combine into a single self-contained HTML file
    plotly_js_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    # Use include_plotlyjs for the first figure, or include CDN
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Moneta Report: {scenario_config.name}</title>
    {plotly_js_cdn}
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 0; background: #fafafa; }}
        .chart-container {{ margin: 20px auto; max-width: 1200px; }}
    </style>
</head>
<body>
<div class="chart-container">
{"".join(figures_html)}
</div>
</body>
</html>
"""

    output_path.write_text(full_html, encoding="utf-8")

    # Try to open in browser (best effort)
    try:
        webbrowser.open(output_path.as_uri())
    except Exception:
        pass

    return output_path
