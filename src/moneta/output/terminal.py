"""Rich terminal output for Moneta simulation results.

Renders query results as formatted terminal output including:
- Header with simulation count, horizon, and timing
- Probability results with dot leaders
- Percentile tables
- Expected value summaries
- Sweep scenario comparison tables
"""

from __future__ import annotations

import io

from rich.console import Console
from rich.table import Table

from moneta.parser.models import ScenarioConfig
from moneta.query.engine import QueryResult

# ---------------------------------------------------------------------------
# Currency formatting
# ---------------------------------------------------------------------------


def format_currency(value: float) -> str:
    """Format a dollar value for display.

    Formatting rules:
    - Negative values: prefixed with "-"
    - $0 exactly: "$0"
    - $0.01 - $999: "$NNN" (whole dollar, no decimals)
    - $1,000 - $9,999: "$N,NNN" (with comma separator)
    - $10,000 - $999,999: "$NNNK" (thousands, no decimal unless needed)
      - If the thousands value has a meaningful fractional part, show one decimal
    - $1,000,000+: "$N.NNM" (millions with up to 2 decimal places)

    Examples:
        612000 -> "$612K"
        1230000 -> "$1.23M"
        23500000 -> "$23.5M"
        1234 -> "$1,234"
        0 -> "$0"
        500 -> "$500"
        10000 -> "$10K"
    """
    if value < 0:
        return f"-{format_currency(-value)}"

    if value == 0:
        return "$0"

    abs_val = abs(value)

    # Millions: >= 1,000,000
    if abs_val >= 1_000_000:
        millions = abs_val / 1_000_000
        # Format with up to 2 decimal places, strip trailing zeros
        formatted = f"{millions:.2f}".rstrip("0").rstrip(".")
        return f"${formatted}M"

    # Thousands: >= 10,000
    if abs_val >= 10_000:
        thousands = abs_val / 1_000
        # If it's a whole number of thousands, show no decimals
        if abs(thousands - round(thousands)) < 0.05:
            return f"${round(thousands)}K"
        # Otherwise show up to 2 decimal places
        formatted = f"{thousands:.2f}".rstrip("0").rstrip(".")
        return f"${formatted}K"

    # Comma-formatted for $1,000-$9,999
    if abs_val >= 1_000:
        return f"${round(abs_val):,}"

    # Small values: $1-$999
    return f"${round(abs_val)}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_header(
    scenario_config: ScenarioConfig,
    elapsed_ms: float,
) -> str:
    """Format the header line with simulation info."""
    n_sims = f"{scenario_config.simulations:,}"
    horizon_years = scenario_config.time_horizon // 12
    timing = f"{elapsed_ms:.0f}ms"
    return (
        f"─── Moneta ── {n_sims} simulations"
        f" ── {horizon_years} year horizon ── {timing} ───"
    )


def _format_probability_result(result: QueryResult) -> str:
    """Format a probability query result with dot leaders."""
    label = result.label
    prob = f"{result.probability:.1f}% probability"

    # Use dot leaders to align. Target ~60 chars total width.
    # Minimum 3 dots.
    total_width = 60
    content_len = len(label) + len(prob) + 2  # 2 for spaces around dots
    n_dots = max(3, total_width - content_len)
    dots = "." * n_dots
    return f"  {label} {dots} {prob}"


def _format_percentile_table(result: QueryResult) -> str:
    """Format a percentile query result as a Rich table."""
    if not result.percentiles:
        return ""

    # Determine percentile keys from the first time point
    first_time = next(iter(result.percentiles))
    pct_keys = sorted(result.percentiles[first_time].keys())

    # Build Rich table
    table = Table(show_header=True, show_edge=True, pad_edge=True)
    table.add_column("", justify="right")
    for p in pct_keys:
        table.add_column(f"p{p}", justify="right")

    for time_months in sorted(result.percentiles.keys()):
        year = time_months // 12
        row_label = f"{year} yr"
        pct_data = result.percentiles[time_months]
        row = [row_label] + [format_currency(pct_data[p]) for p in pct_keys]
        table.add_row(*row)

    # Render table to string
    buf = io.StringIO()
    console = Console(file=buf, width=120, no_color=True)
    console.print(table)
    return buf.getvalue().rstrip()


def _format_expected_result(result: QueryResult) -> str:
    """Format an expected value query result."""
    mean_str = format_currency(result.mean) if result.mean is not None else "N/A"
    median_str = format_currency(result.median) if result.median is not None else "N/A"
    std_str = format_currency(result.std) if result.std is not None else "N/A"
    return f"  {result.label}: mean={mean_str}, median={median_str}, std={std_str}"


def _format_sweep_comparison(
    sweep_results: list[tuple[str, list[QueryResult]]],
) -> str:
    """Format a sweep comparison table.

    Each row is a scenario, each column is a query result.
    For probability queries: show "XX.X%"
    For percentile queries: show the p50 at the first time point
    For expected queries: show the median
    """
    if not sweep_results:
        return ""

    # Get column labels from the first scenario's results
    _, first_results = sweep_results[0]
    if not first_results:
        return ""

    col_labels: list[str] = []
    for qr in first_results:
        if qr.query_type == "probability":
            col_labels.append(qr.label)
        elif qr.query_type == "percentiles":
            # Use "p50 at yr N" for the first time point
            if qr.percentiles:
                first_time = min(qr.percentiles.keys())
                year = first_time // 12
                col_labels.append(f"p50 at yr {year}")
            else:
                col_labels.append(qr.label)
        elif qr.query_type == "expected":
            col_labels.append(qr.label)
        else:
            col_labels.append(qr.label)

    # Build Rich table
    table = Table(show_header=True, show_edge=True, pad_edge=True)
    table.add_column("Scenario", justify="left")
    for label in col_labels:
        table.add_column(label, justify="right")

    for scenario_label, results in sweep_results:
        row = [scenario_label]
        for qr in results:
            if qr.query_type == "probability":
                row.append(f"{qr.probability:.1f}%")
            elif qr.query_type == "percentiles":
                if qr.percentiles:
                    first_time = min(qr.percentiles.keys())
                    pct_data = qr.percentiles[first_time]
                    # Get p50 if available, else first percentile
                    if 50 in pct_data:
                        row.append(format_currency(pct_data[50]))
                    else:
                        first_pct = next(iter(sorted(pct_data.keys())))
                        row.append(format_currency(pct_data[first_pct]))
                else:
                    row.append("N/A")
            elif qr.query_type == "expected":
                if qr.median is not None:
                    row.append(format_currency(qr.median))
                else:
                    row.append("N/A")
            else:
                row.append("--")
        table.add_row(*row)

    # Render to string
    buf = io.StringIO()
    console = Console(file=buf, width=120, no_color=True)
    console.print(table)
    return buf.getvalue().rstrip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_results(
    query_results: list[QueryResult],
    scenario_config: ScenarioConfig,
    elapsed_ms: float,
    sweep_results: list[tuple[str, list[QueryResult]]] | None = None,
) -> str:
    """Render simulation results as formatted terminal output.

    Args:
        query_results: Results from the query engine for the base scenario.
        scenario_config: Scenario configuration (for header info).
        elapsed_ms: Elapsed wall-clock time in milliseconds.
        sweep_results: Optional list of (label, query_results) tuples for
            sweep mode comparison.

    Returns:
        A string containing the formatted terminal output. Does not print
        directly, for testability.
    """
    lines: list[str] = []

    # Header
    lines.append("")
    lines.append(_format_header(scenario_config, elapsed_ms))
    lines.append("")

    # Query results (non-sweep)
    if query_results:
        # Separate probability/expected from percentile results
        prob_expected: list[QueryResult] = []
        percentile_results: list[QueryResult] = []

        for qr in query_results:
            if qr.query_type in ("probability", "expected"):
                prob_expected.append(qr)
            elif qr.query_type == "percentiles":
                percentile_results.append(qr)
            # distribution results are not rendered in terminal

        # Render probability / expected results
        if prob_expected:
            lines.append("Query Results:")
            for qr in prob_expected:
                if qr.query_type == "probability":
                    lines.append(_format_probability_result(qr))
                elif qr.query_type == "expected":
                    lines.append(_format_expected_result(qr))
            lines.append("")

        # Render percentile tables
        for qr in percentile_results:
            lines.append(f"{qr.label}:")
            table_str = _format_percentile_table(qr)
            if table_str:
                # Indent each line of the table
                for table_line in table_str.split("\n"):
                    lines.append(f"  {table_line}")
            lines.append("")

    # Sweep comparison
    if sweep_results:
        lines.append("Scenario Comparison:")
        comparison_str = _format_sweep_comparison(sweep_results)
        if comparison_str:
            for comp_line in comparison_str.split("\n"):
                lines.append(f"  {comp_line}")
        lines.append("")

    return "\n".join(lines)
