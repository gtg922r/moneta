"""Tests for the query evaluation engine."""

import numpy as np
import pytest

from moneta.engine.state import ResultStore
from moneta.parser.models import (
    DistributionQuery,
    ExpectedQuery,
    PercentilesQuery,
    ProbabilityQuery,
)
from moneta.query.engine import QueryResult, evaluate_queries
from moneta.query.expressions import ExpressionError


# ===================================================================
# Helpers
# ===================================================================


def _make_result_store(
    balances: np.ndarray,
    asset_names: list[str],
    cum_inflation: np.ndarray | None = None,
) -> ResultStore:
    """Create a ResultStore from hand-crafted arrays.

    Args:
        balances: float64[n_runs, n_steps, n_assets]
        asset_names: list of asset name strings
        cum_inflation: float64[n_runs, n_steps] (defaults to all 1.0)
    """
    n_runs, n_steps, n_assets = balances.shape
    asset_index = {name: i for i, name in enumerate(asset_names)}

    if cum_inflation is None:
        cum_inflation = np.ones((n_runs, n_steps), dtype=np.float64)

    return ResultStore(
        balances=balances,
        cum_inflation=cum_inflation,
        cash_flow_shortfall=np.zeros((n_runs, n_steps), dtype=np.float64),
        event_fired_at=np.full((n_runs, 0), -1, dtype=np.int32),
        asset_names=asset_names,
        asset_index=asset_index,
        n_runs=n_runs,
        n_steps=n_steps,
        n_assets=n_assets,
    )


# ===================================================================
# Probability query tests
# ===================================================================


class TestProbabilityQuery:
    def test_60_percent(self):
        """60 of 100 runs above threshold -> 60.0%."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        # At step 11 (month 12): 60 runs have value 300k, 40 have value 100k
        balances[:60, 11, 0] = 300000.0
        balances[60:, 11, 0] = 100000.0

        store = _make_result_store(balances, ["portfolio"])

        query = ProbabilityQuery(
            type="probability",
            expression="portfolio > 200000",
            at=12,  # month 12 -> step 11
            label="$200K at year 1",
        )

        results = evaluate_queries([query], store)
        assert len(results) == 1
        assert results[0].query_type == "probability"
        assert results[0].probability == pytest.approx(60.0)
        assert results[0].label == "$200K at year 1"

    def test_all_above(self):
        """All runs above -> 100%."""
        n_runs, n_steps = 100, 12
        balances = np.full((n_runs, n_steps, 1), 500000.0, dtype=np.float64)
        store = _make_result_store(balances, ["portfolio"])

        query = ProbabilityQuery(
            type="probability",
            expression="portfolio > 200000",
            at=12,
        )

        results = evaluate_queries([query], store)
        assert results[0].probability == pytest.approx(100.0)

    def test_none_above(self):
        """No runs above -> 0%."""
        n_runs, n_steps = 100, 12
        balances = np.full((n_runs, n_steps, 1), 100000.0, dtype=np.float64)
        store = _make_result_store(balances, ["portfolio"])

        query = ProbabilityQuery(
            type="probability",
            expression="portfolio > 200000",
            at=12,
        )

        results = evaluate_queries([query], store)
        assert results[0].probability == pytest.approx(0.0)

    def test_with_asset_arithmetic(self):
        """Probability query with asset arithmetic: a + b > threshold."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 2), dtype=np.float64)
        # portfolio = 150k for all, equity varies
        balances[:, 11, 0] = 150000.0
        balances[:50, 11, 1] = 100000.0   # total: 250k > 200k
        balances[50:, 11, 1] = 30000.0    # total: 180k < 200k

        store = _make_result_store(balances, ["portfolio", "equity"])

        query = ProbabilityQuery(
            type="probability",
            expression="portfolio + equity > 200000",
            at=12,
        )

        results = evaluate_queries([query], store)
        assert results[0].probability == pytest.approx(50.0)

    def test_default_label(self):
        """When no label specified, uses expression as label."""
        n_runs, n_steps = 10, 6
        balances = np.full((n_runs, n_steps, 1), 500000.0, dtype=np.float64)
        store = _make_result_store(balances, ["portfolio"])

        query = ProbabilityQuery(
            type="probability",
            expression="portfolio > 200000",
            at=6,
        )

        results = evaluate_queries([query], store)
        assert results[0].label == "portfolio > 200000"


# ===================================================================
# Percentile query tests
# ===================================================================


class TestPercentilesQuery:
    def test_known_data(self):
        """Known data -> exact percentiles (use range(100))."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        # At step 11: values 0..99 (sorted)
        balances[:, 11, 0] = np.arange(100, dtype=np.float64)

        store = _make_result_store(balances, ["portfolio"])

        query = PercentilesQuery(
            type="percentiles",
            values=[10, 50, 90],
            of="portfolio",
            at=12,
            label="Test percentiles",
        )

        results = evaluate_queries([query], store)
        assert len(results) == 1
        assert results[0].query_type == "percentiles"
        assert 12 in results[0].percentiles

        pcts = results[0].percentiles[12]
        # np.percentile with default linear interpolation
        assert pcts[10] == pytest.approx(np.percentile(np.arange(100), 10))
        assert pcts[50] == pytest.approx(np.percentile(np.arange(100), 50))
        assert pcts[90] == pytest.approx(np.percentile(np.arange(100), 90))

    def test_multiple_time_points(self):
        """Percentile query with multiple time points."""
        n_runs, n_steps = 100, 24
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        # Month 6 (step 5): values 0..99
        balances[:, 5, 0] = np.arange(100, dtype=np.float64)
        # Month 12 (step 11): values 100..199
        balances[:, 11, 0] = np.arange(100, 200, dtype=np.float64)
        # Month 24 (step 23): values 200..299
        balances[:, 23, 0] = np.arange(200, 300, dtype=np.float64)

        store = _make_result_store(balances, ["portfolio"])

        query = PercentilesQuery(
            type="percentiles",
            values=[25, 50, 75],
            of="portfolio",
            at=[6, 12, 24],
        )

        results = evaluate_queries([query], store)
        pcts = results[0].percentiles

        # Check each time point
        assert 6 in pcts
        assert 12 in pcts
        assert 24 in pcts

        # Month 12: values 100..199
        assert pcts[12][50] == pytest.approx(
            np.percentile(np.arange(100, 200), 50)
        )

        # Month 24: values 200..299
        assert pcts[24][50] == pytest.approx(
            np.percentile(np.arange(200, 300), 50)
        )

    def test_single_at_value(self):
        """Percentile query with a single 'at' value (not a list)."""
        n_runs, n_steps = 50, 6
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        balances[:, 5, 0] = np.arange(50, dtype=np.float64)

        store = _make_result_store(balances, ["portfolio"])

        query = PercentilesQuery(
            type="percentiles",
            values=[50],
            of="portfolio",
            at=6,
        )

        results = evaluate_queries([query], store)
        assert 6 in results[0].percentiles


# ===================================================================
# Expected query tests
# ===================================================================


class TestExpectedQuery:
    def test_known_mean_median_std(self):
        """Known data -> exact mean, median, std."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        # At step 11: values 0..99
        data = np.arange(100, dtype=np.float64)
        balances[:, 11, 0] = data

        store = _make_result_store(balances, ["portfolio"])

        query = ExpectedQuery(
            type="expected",
            of="portfolio",
            at=12,
            label="Expected portfolio",
        )

        results = evaluate_queries([query], store)
        assert len(results) == 1
        assert results[0].query_type == "expected"
        assert results[0].mean == pytest.approx(np.mean(data))
        assert results[0].median == pytest.approx(np.median(data))
        assert results[0].std == pytest.approx(np.std(data))

    def test_with_expression(self):
        """Expected query with arithmetic expression (a + b)."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 2), dtype=np.float64)
        balances[:, 11, 0] = 1000.0  # portfolio
        balances[:, 11, 1] = np.arange(100, dtype=np.float64) * 10  # equity

        store = _make_result_store(balances, ["portfolio", "equity"])

        query = ExpectedQuery(
            type="expected",
            of="portfolio + equity",
            at=12,
        )

        results = evaluate_queries([query], store)
        expected_values = 1000.0 + np.arange(100) * 10
        assert results[0].mean == pytest.approx(np.mean(expected_values))
        assert results[0].median == pytest.approx(np.median(expected_values))


# ===================================================================
# Distribution query tests
# ===================================================================


class TestDistributionQuery:
    def test_histogram(self):
        """Distribution query: correct histogram."""
        n_runs, n_steps = 1000, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        # Uniform data at step 11
        rng = np.random.default_rng(42)
        balances[:, 11, 0] = rng.uniform(0, 100, size=n_runs)

        store = _make_result_store(balances, ["portfolio"])

        query = DistributionQuery(
            type="distribution",
            of="portfolio",
            at=12,
            bins=10,
            label="Portfolio distribution",
        )

        results = evaluate_queries([query], store)
        assert len(results) == 1
        assert results[0].query_type == "distribution"
        assert results[0].histogram_bins is not None
        assert results[0].histogram_counts is not None
        # 10 bins -> 11 bin edges
        assert len(results[0].histogram_bins) == 11
        assert len(results[0].histogram_counts) == 10
        # Total counts should equal n_runs
        assert np.sum(results[0].histogram_counts) == n_runs


# ===================================================================
# Inflation adjustment tests
# ===================================================================


class TestInflationAdjustment:
    def test_probability_inflation_adjustment(self):
        """Values correctly divided by cum_inflation for probability query."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        # All runs have 220000 nominal at step 11
        balances[:, 11, 0] = 220000.0

        # cum_inflation = 1.1 for all runs at step 11
        # Real value = 220000 / 1.1 = 200000
        cum_inflation = np.ones((n_runs, n_steps), dtype=np.float64)
        cum_inflation[:, 11] = 1.1

        store = _make_result_store(balances, ["portfolio"], cum_inflation)

        # Without inflation adjustment: 220000 > 200000 -> 100%
        query_nominal = ProbabilityQuery(
            type="probability",
            expression="portfolio > 200000",
            at=12,
        )
        results_nominal = evaluate_queries([query_nominal], store)
        assert results_nominal[0].probability == pytest.approx(100.0)

        # With inflation adjustment: 200000 > 200000 -> False -> 0%
        query_real = ProbabilityQuery(
            type="probability",
            expression="portfolio > 200000",
            at=12,
            adjust_for="inflation",
        )
        results_real = evaluate_queries([query_real], store)
        assert results_real[0].probability == pytest.approx(0.0)

    def test_percentile_inflation_adjustment(self):
        """Percentile values divided by cum_inflation."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        balances[:, 11, 0] = 200.0  # nominal

        cum_inflation = np.ones((n_runs, n_steps), dtype=np.float64)
        cum_inflation[:, 11] = 2.0  # 100% cumulative inflation

        store = _make_result_store(balances, ["portfolio"], cum_inflation)

        query = PercentilesQuery(
            type="percentiles",
            values=[50],
            of="portfolio",
            at=12,
            adjust_for="inflation",
        )

        results = evaluate_queries([query], store)
        # 200 / 2.0 = 100 real dollars
        assert results[0].percentiles[12][50] == pytest.approx(100.0)

    def test_expected_inflation_adjustment(self):
        """Expected values divided by cum_inflation."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        balances[:, 11, 0] = 300.0

        cum_inflation = np.ones((n_runs, n_steps), dtype=np.float64)
        cum_inflation[:, 11] = 1.5

        store = _make_result_store(balances, ["portfolio"], cum_inflation)

        query = ExpectedQuery(
            type="expected",
            of="portfolio",
            at=12,
            adjust_for="inflation",
        )

        results = evaluate_queries([query], store)
        assert results[0].mean == pytest.approx(200.0)
        assert results[0].median == pytest.approx(200.0)

    def test_distribution_inflation_adjustment(self):
        """Distribution values divided by cum_inflation."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        balances[:, 11, 0] = np.arange(100, dtype=np.float64) * 100

        cum_inflation = np.ones((n_runs, n_steps), dtype=np.float64)
        cum_inflation[:, 11] = 2.0

        store = _make_result_store(balances, ["portfolio"], cum_inflation)

        query_nominal = DistributionQuery(
            type="distribution",
            of="portfolio",
            at=12,
            bins=10,
        )
        query_real = DistributionQuery(
            type="distribution",
            of="portfolio",
            at=12,
            bins=10,
            adjust_for="inflation",
        )

        results_nominal = evaluate_queries([query_nominal], store)
        results_real = evaluate_queries([query_real], store)

        # Real histogram should have max bin edge = half of nominal max
        nominal_max = results_nominal[0].histogram_bins[-1]
        real_max = results_real[0].histogram_bins[-1]
        assert real_max == pytest.approx(nominal_max / 2.0)


# ===================================================================
# Query with asset arithmetic tests
# ===================================================================


class TestAssetArithmetic:
    def test_percentile_with_expression(self):
        """Percentile of (a + b) computes correctly."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 2), dtype=np.float64)
        balances[:, 11, 0] = np.arange(100, dtype=np.float64)  # a
        balances[:, 11, 1] = np.arange(100, dtype=np.float64) * 2  # b

        store = _make_result_store(balances, ["a", "b"])

        query = PercentilesQuery(
            type="percentiles",
            values=[50],
            of="a + b",
            at=12,
        )

        results = evaluate_queries([query], store)
        # a + b = 0+0, 1+2, 2+4, ... = 0, 3, 6, ... = 3*i for i in range(100)
        expected_p50 = np.percentile(np.arange(100) * 3.0, 50)
        assert results[0].percentiles[12][50] == pytest.approx(expected_p50)


# ===================================================================
# Error handling tests
# ===================================================================


class TestErrors:
    def test_unknown_asset_in_expression(self):
        """Unknown asset -> ExpressionError."""
        n_runs, n_steps = 10, 6
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        store = _make_result_store(balances, ["portfolio"])

        query = ProbabilityQuery(
            type="probability",
            expression="savings > 100",
            at=6,
        )

        with pytest.raises(ExpressionError, match="Unknown asset name 'savings'"):
            evaluate_queries([query], store)

    def test_malformed_expression(self):
        """Malformed expression -> ExpressionError."""
        n_runs, n_steps = 10, 6
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        store = _make_result_store(balances, ["portfolio"])

        query = ProbabilityQuery(
            type="probability",
            expression="portfolio >",
            at=6,
        )

        with pytest.raises(ExpressionError):
            evaluate_queries([query], store)


# ===================================================================
# Multiple queries test
# ===================================================================


class TestMultipleQueries:
    def test_evaluate_mixed_queries(self):
        """Evaluate multiple different query types in one call."""
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        balances[:, 11, 0] = np.arange(100, dtype=np.float64)

        store = _make_result_store(balances, ["portfolio"])

        queries = [
            ProbabilityQuery(
                type="probability",
                expression="portfolio > 50",
                at=12,
                label="Prob query",
            ),
            PercentilesQuery(
                type="percentiles",
                values=[50],
                of="portfolio",
                at=12,
                label="Pct query",
            ),
            ExpectedQuery(
                type="expected",
                of="portfolio",
                at=12,
                label="Exp query",
            ),
            DistributionQuery(
                type="distribution",
                of="portfolio",
                at=12,
                bins=10,
                label="Dist query",
            ),
        ]

        results = evaluate_queries(queries, store)
        assert len(results) == 4
        assert results[0].query_type == "probability"
        assert results[1].query_type == "percentiles"
        assert results[2].query_type == "expected"
        assert results[3].query_type == "distribution"
