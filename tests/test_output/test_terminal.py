"""Tests for Rich terminal output rendering."""

from __future__ import annotations

import pytest

from moneta.output.terminal import format_currency, render_results
from moneta.parser.models import ScenarioConfig
from moneta.query.engine import QueryResult


# ---------------------------------------------------------------------------
# format_currency tests
# ---------------------------------------------------------------------------


class TestFormatCurrency:
    """Test currency formatting across value ranges."""

    def test_zero(self):
        assert format_currency(0) == "$0"

    def test_small_value(self):
        assert format_currency(500) == "$500"

    def test_hundreds(self):
        assert format_currency(999) == "$999"

    def test_one_dollar(self):
        assert format_currency(1) == "$1"

    def test_comma_range_lower(self):
        result = format_currency(1234)
        assert result == "$1,234"

    def test_comma_range_upper(self):
        # 9999 < 10000, so it falls into the $1,000-$9,999 bracket
        assert format_currency(9999) == "$9,999"

    def test_thousands_exact(self):
        assert format_currency(10000) == "$10K"

    def test_thousands_with_fraction(self):
        result = format_currency(612000)
        assert result == "$612K"

    def test_thousands_non_round(self):
        result = format_currency(15500)
        assert result == "$15.5K"

    def test_millions_lower(self):
        result = format_currency(1230000)
        assert result == "$1.23M"

    def test_millions_large(self):
        result = format_currency(23500000)
        assert result == "$23.5M"

    def test_millions_exact(self):
        result = format_currency(1000000)
        assert result == "$1M"

    def test_millions_one_decimal(self):
        result = format_currency(1100000)
        assert result == "$1.1M"

    def test_millions_two_decimals(self):
        result = format_currency(1120000)
        assert result == "$1.12M"

    def test_negative_value(self):
        result = format_currency(-500)
        assert result == "-$500"

    def test_negative_millions(self):
        result = format_currency(-1230000)
        assert result == "-$1.23M"

    def test_very_large(self):
        result = format_currency(100_000_000)
        assert result == "$100M"

    def test_borderline_thousands(self):
        # Exactly $10,000 should be "$10K"
        assert format_currency(10000) == "$10K"

    def test_borderline_millions(self):
        # Exactly $1,000,000 should be "$1M"
        assert format_currency(1_000_000) == "$1M"


# ---------------------------------------------------------------------------
# ScenarioConfig fixture helper
# ---------------------------------------------------------------------------


def _make_scenario_config(
    name: str = "Test scenario",
    time_horizon_months: int = 360,
    simulations: int = 10000,
    seed: int | None = 42,
) -> ScenarioConfig:
    """Build a ScenarioConfig for testing."""
    return ScenarioConfig(
        name=name,
        time_horizon=time_horizon_months,
        simulations=simulations,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# render_results — probability queries
# ---------------------------------------------------------------------------


class TestRenderProbabilityResults:
    """Test rendering of probability query results."""

    def test_single_probability(self):
        results = [
            QueryResult(
                label="$2M net worth at year 10",
                query_type="probability",
                probability=43.2,
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=142.0)

        assert "43.2% probability" in output
        assert "$2M net worth at year 10" in output

    def test_multiple_probabilities(self):
        results = [
            QueryResult(
                label="$2M net worth at year 10",
                query_type="probability",
                probability=43.2,
            ),
            QueryResult(
                label="$1.5M portfolio (real $) at year 15",
                query_type="probability",
                probability=61.8,
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=142.0)

        assert "43.2% probability" in output
        assert "61.8% probability" in output
        assert "Query Results:" in output

    def test_dot_leaders_present(self):
        results = [
            QueryResult(
                label="$2M net worth at year 10",
                query_type="probability",
                probability=43.2,
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=142.0)

        # Dot leaders should appear between label and probability
        assert "..." in output


# ---------------------------------------------------------------------------
# render_results — percentile queries
# ---------------------------------------------------------------------------


class TestRenderPercentileResults:
    """Test rendering of percentile query results."""

    def test_percentile_table(self):
        results = [
            QueryResult(
                label="Portfolio value distribution (real $)",
                query_type="percentiles",
                percentiles={
                    60: {10: 612000, 25: 742000, 50: 921000, 75: 1140000, 90: 1410000},
                    120: {10: 891000, 25: 1120000, 50: 1480000, 75: 1950000, 90: 2560000},
                },
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=142.0)

        # Should contain table header
        assert "Portfolio value distribution (real $):" in output
        # Should contain percentile column headers
        assert "p10" in output
        assert "p50" in output
        assert "p90" in output
        # Should contain year row labels
        assert "5 yr" in output
        assert "10 yr" in output
        # Should contain formatted currency values
        assert "$612K" in output
        assert "$1.48M" in output

    def test_single_time_point(self):
        results = [
            QueryResult(
                label="Portfolio at year 10",
                query_type="percentiles",
                percentiles={
                    120: {10: 500000, 50: 1000000, 90: 2000000},
                },
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=50.0)

        assert "10 yr" in output
        assert "$500K" in output
        assert "$1M" in output
        assert "$2M" in output


# ---------------------------------------------------------------------------
# render_results — expected value queries
# ---------------------------------------------------------------------------


class TestRenderExpectedResults:
    """Test rendering of expected value query results."""

    def test_expected_result(self):
        results = [
            QueryResult(
                label="Expected portfolio at year 10",
                query_type="expected",
                mean=1500000.0,
                median=1200000.0,
                std=800000.0,
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=100.0)

        assert "Expected portfolio at year 10" in output
        assert "mean=$1.5M" in output
        assert "median=$1.2M" in output
        assert "std=$800K" in output


# ---------------------------------------------------------------------------
# render_results — header
# ---------------------------------------------------------------------------


class TestRenderHeader:
    """Test header formatting with simulation info."""

    def test_header_contains_simulation_count(self):
        results = []
        config = _make_scenario_config(simulations=10000)
        output = render_results(results, config, elapsed_ms=142.0)

        assert "10,000 simulations" in output

    def test_header_contains_horizon(self):
        results = []
        config = _make_scenario_config(time_horizon_months=360)
        output = render_results(results, config, elapsed_ms=142.0)

        assert "30 year horizon" in output

    def test_header_contains_timing(self):
        results = []
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=142.0)

        assert "142ms" in output

    def test_header_moneta_label(self):
        results = []
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=100.0)

        assert "Moneta" in output

    def test_header_different_values(self):
        results = []
        config = _make_scenario_config(simulations=5000, time_horizon_months=120)
        output = render_results(results, config, elapsed_ms=55.0)

        assert "5,000 simulations" in output
        assert "10 year horizon" in output
        assert "55ms" in output


# ---------------------------------------------------------------------------
# render_results — sweep comparison
# ---------------------------------------------------------------------------


class TestRenderSweepComparison:
    """Test sweep scenario comparison table rendering."""

    def test_sweep_with_probability_queries(self):
        sweep_results = [
            (
                "conservative",
                [
                    QueryResult(
                        label="$2M at yr 10",
                        query_type="probability",
                        probability=28.1,
                    ),
                ],
            ),
            (
                "base",
                [
                    QueryResult(
                        label="$2M at yr 10",
                        query_type="probability",
                        probability=43.2,
                    ),
                ],
            ),
        ]

        config = _make_scenario_config()
        output = render_results(
            query_results=[],
            scenario_config=config,
            elapsed_ms=200.0,
            sweep_results=sweep_results,
        )

        assert "Scenario Comparison:" in output
        assert "conservative" in output
        assert "base" in output
        assert "28.1%" in output
        assert "43.2%" in output

    def test_sweep_with_percentile_queries(self):
        sweep_results = [
            (
                "conservative",
                [
                    QueryResult(
                        label="Portfolio distribution",
                        query_type="percentiles",
                        percentiles={
                            120: {10: 800000, 50: 1120000, 90: 1600000},
                        },
                    ),
                ],
            ),
            (
                "base",
                [
                    QueryResult(
                        label="Portfolio distribution",
                        query_type="percentiles",
                        percentiles={
                            120: {10: 900000, 50: 1480000, 90: 2100000},
                        },
                    ),
                ],
            ),
        ]

        config = _make_scenario_config()
        output = render_results(
            query_results=[],
            scenario_config=config,
            elapsed_ms=200.0,
            sweep_results=sweep_results,
        )

        assert "Scenario Comparison:" in output
        assert "conservative" in output
        assert "base" in output
        # Should show p50 values
        assert "$1.12M" in output
        assert "$1.48M" in output

    def test_sweep_with_mixed_queries(self):
        sweep_results = [
            (
                "conservative",
                [
                    QueryResult(
                        label="$2M at yr 10",
                        query_type="probability",
                        probability=28.1,
                    ),
                    QueryResult(
                        label="Portfolio distribution",
                        query_type="percentiles",
                        percentiles={
                            120: {10: 800000, 50: 1120000, 90: 1600000},
                        },
                    ),
                ],
            ),
            (
                "base",
                [
                    QueryResult(
                        label="$2M at yr 10",
                        query_type="probability",
                        probability=43.2,
                    ),
                    QueryResult(
                        label="Portfolio distribution",
                        query_type="percentiles",
                        percentiles={
                            120: {10: 900000, 50: 1480000, 90: 2100000},
                        },
                    ),
                ],
            ),
        ]

        config = _make_scenario_config()
        output = render_results(
            query_results=[],
            scenario_config=config,
            elapsed_ms=200.0,
            sweep_results=sweep_results,
        )

        assert "Scenario Comparison:" in output
        assert "28.1%" in output
        assert "43.2%" in output
        assert "$1.12M" in output
        assert "$1.48M" in output

    def test_sweep_scenario_column_header(self):
        sweep_results = [
            (
                "conservative",
                [
                    QueryResult(
                        label="$2M at yr 10",
                        query_type="probability",
                        probability=28.1,
                    ),
                ],
            ),
        ]

        config = _make_scenario_config()
        output = render_results(
            query_results=[],
            scenario_config=config,
            elapsed_ms=200.0,
            sweep_results=sweep_results,
        )

        assert "Scenario" in output


# ---------------------------------------------------------------------------
# render_results — empty / edge cases
# ---------------------------------------------------------------------------


class TestRenderEdgeCases:
    """Test edge cases and graceful handling."""

    def test_empty_query_results(self):
        config = _make_scenario_config()
        output = render_results([], config, elapsed_ms=100.0)

        # Should still have header
        assert "Moneta" in output
        assert "10,000 simulations" in output
        # Should not crash
        assert isinstance(output, str)

    def test_empty_sweep_results(self):
        config = _make_scenario_config()
        output = render_results(
            query_results=[],
            scenario_config=config,
            elapsed_ms=100.0,
            sweep_results=[],
        )

        assert "Moneta" in output
        assert isinstance(output, str)

    def test_probability_zero(self):
        results = [
            QueryResult(
                label="Impossible event",
                query_type="probability",
                probability=0.0,
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=50.0)

        assert "0.0% probability" in output

    def test_probability_hundred(self):
        results = [
            QueryResult(
                label="Certain event",
                query_type="probability",
                probability=100.0,
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=50.0)

        assert "100.0% probability" in output

    def test_mixed_query_types(self):
        """Mixing probability and percentile results should render both."""
        results = [
            QueryResult(
                label="$2M net worth at year 10",
                query_type="probability",
                probability=43.2,
            ),
            QueryResult(
                label="Portfolio distribution",
                query_type="percentiles",
                percentiles={
                    60: {10: 612000, 50: 921000, 90: 1410000},
                },
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=142.0)

        assert "43.2% probability" in output
        assert "Portfolio distribution:" in output
        assert "$612K" in output

    def test_distribution_query_not_rendered(self):
        """Distribution queries are for HTML reports; terminal should not crash."""
        import numpy as np

        results = [
            QueryResult(
                label="Distribution of portfolio",
                query_type="distribution",
                histogram_bins=np.array([0, 100000, 200000]),
                histogram_counts=np.array([50, 50]),
            ),
        ]
        config = _make_scenario_config()
        output = render_results(results, config, elapsed_ms=50.0)

        # Should not crash, header still present
        assert "Moneta" in output

    def test_sweep_with_expected_query(self):
        """Sweep comparison should handle expected queries."""
        sweep_results = [
            (
                "scenario_a",
                [
                    QueryResult(
                        label="Expected portfolio",
                        query_type="expected",
                        mean=1500000.0,
                        median=1200000.0,
                        std=800000.0,
                    ),
                ],
            ),
            (
                "scenario_b",
                [
                    QueryResult(
                        label="Expected portfolio",
                        query_type="expected",
                        mean=2000000.0,
                        median=1800000.0,
                        std=1000000.0,
                    ),
                ],
            ),
        ]

        config = _make_scenario_config()
        output = render_results(
            query_results=[],
            scenario_config=config,
            elapsed_ms=200.0,
            sweep_results=sweep_results,
        )

        assert "scenario_a" in output
        assert "scenario_b" in output
        assert "$1.2M" in output
        assert "$1.8M" in output

    def test_render_returns_string(self):
        config = _make_scenario_config()
        output = render_results([], config, elapsed_ms=100.0)
        assert isinstance(output, str)
