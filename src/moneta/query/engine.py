"""Query evaluation engine for Moneta.

Evaluates parsed queries (probability, percentiles, expected, distribution)
against a ResultStore containing completed simulation data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moneta.engine.state import ResultStore
from moneta.parser.models import (
    DistributionQuery,
    ExpectedQuery,
    PercentilesQuery,
    ProbabilityQuery,
)
from moneta.query.expressions import (
    ExpressionError,
    evaluate,
    parse_expression,
)

# ---------------------------------------------------------------------------
# Query result dataclass
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Result of evaluating a single query against simulation results."""

    label: str
    query_type: str

    # For probability queries
    probability: float | None = None

    # For percentile queries: {time_point_months: {percentile: value}}
    percentiles: dict[int, dict[int, float]] | None = None

    # For expected queries
    mean: float | None = None
    median: float | None = None
    std: float | None = None

    # For distribution queries
    histogram_bins: np.ndarray | None = None
    histogram_counts: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _time_to_step(at_months: int, n_steps: int) -> int:
    """Convert a Duration (months) to a 0-based step index.

    The `at` field is already in months (Duration type).  Step 0 is
    month 1, so at=1 -> step 0, at=N -> step N-1.  For queries
    asking about month N (which is step N-1), we clamp to valid range.
    """
    step = at_months - 1  # 1-based month to 0-based step
    if step < 0:
        step = 0
    if step >= n_steps:
        step = n_steps - 1
    return step


def _build_values_dict(
    results: ResultStore,
    step: int,
) -> dict[str, np.ndarray]:
    """Build a {name: float64[n_runs]} dict for a given time step."""
    values: dict[str, np.ndarray] = {}
    for name, idx in results.asset_index.items():
        values[name] = results.balances[:, step, idx]
    # Add virtual field for cash flow shortfall
    if (
        hasattr(results, "cash_flow_shortfall")
        and results.cash_flow_shortfall is not None
    ):
        values["shortfall"] = results.cash_flow_shortfall[:, step]
    return values


def _maybe_adjust_inflation(
    arr: np.ndarray,
    results: ResultStore,
    step: int,
    adjust_for: str | None,
) -> np.ndarray:
    """If adjust_for == 'inflation', divide values by cum_inflation at step."""
    if adjust_for == "inflation":
        cum_inf = results.cum_inflation[:, step]
        # Avoid division by zero (should not happen in practice)
        cum_inf = np.where(cum_inf == 0.0, 1.0, cum_inf)
        result: np.ndarray = arr / cum_inf
        return result
    return arr


def _evaluate_expression_at_step(
    expression_str: str,
    results: ResultStore,
    step: int,
) -> np.ndarray:
    """Parse and evaluate an expression at a given time step.

    Returns float64[n_runs] for arithmetic expressions or
    bool[n_runs] for comparison expressions.
    """
    node = parse_expression(expression_str)
    values = _build_values_dict(results, step)
    result: np.ndarray = evaluate(node, values)
    return result


def _evaluate_of_at_step(
    of_str: str,
    results: ResultStore,
    step: int,
    adjust_for: str | None,
) -> np.ndarray:
    """Parse and evaluate an 'of' field at a given time step.

    The 'of' field should be an arithmetic expression (no comparison).
    Returns float64[n_runs], optionally inflation-adjusted.
    """
    node = parse_expression(of_str)
    values = _build_values_dict(results, step)
    arr = evaluate(node, values)
    return _maybe_adjust_inflation(arr, results, step, adjust_for)


# ---------------------------------------------------------------------------
# Per-query-type evaluators
# ---------------------------------------------------------------------------


def _eval_probability(
    query: ProbabilityQuery,
    results: ResultStore,
) -> QueryResult:
    """Evaluate a probability query."""
    step = _time_to_step(query.at, results.n_steps)

    # For probability queries with inflation adjustment, we need to
    # adjust the values before comparison. Parse the expression to get
    # the AST, then adjust if needed.
    if query.adjust_for == "inflation":
        # We need to evaluate both sides of the comparison with inflation
        # adjustment applied.  The simplest correct approach: build adjusted
        # values dict and evaluate the full expression against it.
        values = _build_values_dict(results, step)
        cum_inf = results.cum_inflation[:, step]
        cum_inf = np.where(cum_inf == 0.0, 1.0, cum_inf)
        adjusted_values = {k: v / cum_inf for k, v in values.items()}
        node = parse_expression(query.expression)
        result_arr = evaluate(node, adjusted_values)
    else:
        result_arr = _evaluate_expression_at_step(query.expression, results, step)

    # result_arr should be bool[n_runs]
    probability = float(np.mean(result_arr.astype(np.float64))) * 100.0

    label = query.label or query.expression
    return QueryResult(
        label=label,
        query_type="probability",
        probability=probability,
    )


def _eval_percentiles(
    query: PercentilesQuery,
    results: ResultStore,
) -> QueryResult:
    """Evaluate a percentiles query."""
    # Normalize at to list of months
    at_values: list[int]
    at_values = query.at if isinstance(query.at, list) else [query.at]

    percentiles_result: dict[int, dict[int, float]] = {}

    for at_months in at_values:
        step = _time_to_step(at_months, results.n_steps)
        arr = _evaluate_of_at_step(query.of, results, step, query.adjust_for)

        pct_values = np.percentile(arr, query.values)
        percentiles_result[at_months] = {
            p: float(v) for p, v in zip(query.values, pct_values, strict=True)
        }

    label = query.label or f"Percentiles of {query.of}"
    return QueryResult(
        label=label,
        query_type="percentiles",
        percentiles=percentiles_result,
    )


def _eval_expected(
    query: ExpectedQuery,
    results: ResultStore,
) -> QueryResult:
    """Evaluate an expected value query."""
    step = _time_to_step(query.at, results.n_steps)
    arr = _evaluate_of_at_step(query.of, results, step, query.adjust_for)

    label = query.label or f"Expected value of {query.of}"
    return QueryResult(
        label=label,
        query_type="expected",
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
    )


def _eval_distribution(
    query: DistributionQuery,
    results: ResultStore,
) -> QueryResult:
    """Evaluate a distribution query."""
    step = _time_to_step(query.at, results.n_steps)
    arr = _evaluate_of_at_step(query.of, results, step, query.adjust_for)

    counts, bins = np.histogram(arr, bins=query.bins)

    label = query.label or f"Distribution of {query.of}"
    return QueryResult(
        label=label,
        query_type="distribution",
        histogram_bins=bins,
        histogram_counts=counts,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_queries(
    queries: list[
        ProbabilityQuery | PercentilesQuery | ExpectedQuery | DistributionQuery
    ],
    results: ResultStore,
) -> list[QueryResult]:
    """Evaluate all queries against the result store.

    Args:
        queries: List of parsed query objects from the model.
        results: Completed simulation ResultStore.

    Returns:
        List of QueryResult objects, one per input query.

    Raises:
        ExpressionError: If a query expression references an unknown asset
            or is otherwise malformed.
    """
    query_results: list[QueryResult] = []

    for query in queries:
        if isinstance(query, ProbabilityQuery):
            query_results.append(_eval_probability(query, results))
        elif isinstance(query, PercentilesQuery):
            query_results.append(_eval_percentiles(query, results))
        elif isinstance(query, ExpectedQuery):
            query_results.append(_eval_expected(query, results))
        elif isinstance(query, DistributionQuery):
            query_results.append(_eval_distribution(query, results))
        else:
            raise ExpressionError(f"Unknown query type: {type(query)}")

    return query_results
