"""CLI entry point for Moneta — probabilistic financial modeling engine.

Provides ``moneta run`` and ``moneta validate`` commands via Click.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from moneta import MonetaError
from moneta.engine.orchestrator import run_simulation
from moneta.output.report import generate_report
from moneta.output.terminal import render_results
from moneta.parser.loader import load_model
from moneta.query.engine import evaluate_queries


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main():
    """Moneta — Probabilistic financial modeling engine."""


# ---------------------------------------------------------------------------
# ``moneta run``
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.option(
    "--simulations",
    "-n",
    type=int,
    default=None,
    help="Override simulation count",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./output",
    help="Output directory",
)
@click.option(
    "--no-report",
    is_flag=True,
    help="Skip HTML report generation",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show timing details and debug info",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def run(
    model_file: str,
    simulations: int | None,
    seed: int | None,
    output: str,
    no_report: bool,
    verbose: bool,
    output_format: str,
) -> None:
    """Run Monte Carlo simulation on a model file."""
    try:
        # 1. Load model
        model = load_model(Path(model_file))

        # 2. Apply CLI overrides
        if simulations is not None:
            model = model.model_copy(
                update={
                    "scenario": model.scenario.model_copy(
                        update={"simulations": simulations}
                    )
                }
            )
        if seed is not None:
            model = model.model_copy(
                update={
                    "scenario": model.scenario.model_copy(
                        update={"seed": seed}
                    )
                }
            )

        # 3. Run simulation with timing
        start = time.perf_counter()
        results = run_simulation(model, seed=model.scenario.seed)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 4. Evaluate queries
        query_results = evaluate_queries(model.queries, results)

        # 5. Output based on format
        if output_format == "table":
            output_text = render_results(
                query_results, model.scenario, elapsed_ms
            )
            click.echo(output_text)
        elif output_format == "json":
            json_output = _results_to_json(
                query_results, model.scenario, elapsed_ms
            )
            click.echo(json.dumps(json_output, indent=2))
        elif output_format == "csv":
            csv_output = _results_to_csv(query_results, model.scenario, elapsed_ms)
            click.echo(csv_output)

        # 6. Generate HTML report (unless --no-report)
        if not no_report:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            report_name = Path(model_file).stem + "_report.html"
            report_path = output_dir / report_name
            generate_report(
                results, query_results, model.scenario, report_path
            )
            click.echo(f"\n{_REPORT_ICON} Full report: {report_path}")

    except MonetaError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        if verbose:
            import traceback

            traceback.print_exc()
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# ``moneta validate``
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model_file", type=click.Path(exists=True))
def validate(model_file: str) -> None:
    """Validate a model file without running simulation."""
    try:
        model = load_model(Path(model_file))
        n_assets = len(model.assets)
        n_queries = len(model.queries)
        click.echo(f"{_CHECK_ICON} {model_file} is valid")
        click.echo(
            f"  {n_assets} assets, {n_queries} queries, "
            f"{model.scenario.simulations} simulations"
        )
    except MonetaError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

# Unicode icons used in output
_REPORT_ICON = "\U0001f4c8"  # chart with upwards trend
_CHECK_ICON = "\u2713"  # check mark


def _results_to_json(
    query_results: list,
    scenario_config,
    elapsed_ms: float,
) -> dict:
    """Convert query results to a JSON-serializable dict."""
    return {
        "scenario": {
            "name": scenario_config.name,
            "simulations": scenario_config.simulations,
            "time_horizon_months": scenario_config.time_horizon,
        },
        "elapsed_ms": round(elapsed_ms, 1),
        "queries": [_query_result_to_dict(qr) for qr in query_results],
    }


def _query_result_to_dict(qr) -> dict:
    """Convert a single QueryResult to a JSON-serializable dict."""
    d: dict = {
        "label": qr.label,
        "type": qr.query_type,
    }
    if qr.query_type == "probability":
        d["probability"] = qr.probability
    elif qr.query_type == "percentiles":
        if qr.percentiles is not None:
            # Convert int keys to strings for JSON compatibility
            d["percentiles"] = {
                str(time_months): {
                    str(p): v for p, v in pct_data.items()
                }
                for time_months, pct_data in qr.percentiles.items()
            }
    elif qr.query_type == "expected":
        d["mean"] = qr.mean
        d["median"] = qr.median
        d["std"] = qr.std
    elif qr.query_type == "distribution":
        if qr.histogram_bins is not None:
            d["histogram_bins"] = qr.histogram_bins.tolist()
        if qr.histogram_counts is not None:
            d["histogram_counts"] = qr.histogram_counts.tolist()
    return d


def _results_to_csv(
    query_results: list,
    scenario_config,
    elapsed_ms: float,
) -> str:
    """Convert query results to CSV format."""
    lines: list[str] = []
    for qr in query_results:
        if qr.query_type == "probability":
            lines.append(f"{qr.label},probability,{qr.probability:.1f}%")
        elif qr.query_type == "percentiles" and qr.percentiles:
            for time_months, pct_data in sorted(qr.percentiles.items()):
                for p, v in sorted(pct_data.items()):
                    lines.append(
                        f"{qr.label},percentile,{time_months}m,p{p},{v:.2f}"
                    )
        elif qr.query_type == "expected":
            lines.append(
                f"{qr.label},expected,mean={qr.mean:.2f},"
                f"median={qr.median:.2f},std={qr.std:.2f}"
            )
    return "\n".join(lines)
