"""Tests for the Plotly HTML report generator."""

import numpy as np
import plotly.graph_objects as go
import pytest

from moneta.engine.state import ResultStore
from moneta.output.report import (
    _compute_fan_chart_data,
    _create_fan_chart,
    _create_histogram,
    _create_probability_timeline,
    _select_sample_paths,
    generate_report,
)
from moneta.parser.models import ScenarioConfig
from moneta.query.engine import QueryResult

# ===================================================================
# Helpers
# ===================================================================


def _make_result_store(
    n_runs: int = 100,
    n_steps: int = 12,
    n_assets: int = 1,
    asset_names: list[str] | None = None,
    seed: int = 42,
) -> ResultStore:
    """Create a ResultStore with lognormal data for testing."""
    rng = np.random.default_rng(seed)

    if asset_names is None:
        asset_names = [f"asset_{i}" for i in range(n_assets)]

    balances = rng.lognormal(mean=12, sigma=0.5, size=(n_runs, n_steps, n_assets))
    cum_inflation = np.ones((n_runs, n_steps)) * np.linspace(1.0, 1.03, n_steps)
    event_fired_at = np.full((n_runs, 0), -1, dtype=np.int32)
    asset_index = {name: i for i, name in enumerate(asset_names)}

    return ResultStore(
        balances=balances,
        cum_inflation=cum_inflation,
        cash_flow_shortfall=np.zeros((n_runs, n_steps), dtype=np.float64),
        event_fired_at=event_fired_at,
        asset_names=asset_names,
        asset_index=asset_index,
        n_runs=n_runs,
        n_steps=n_steps,
        n_assets=n_assets,
    )


def _make_scenario_config() -> ScenarioConfig:
    """Create a ScenarioConfig for testing."""
    return ScenarioConfig(
        name="Test Scenario",
        time_horizon=12,  # 12 months = 1 year
        simulations=100,
    )


# ===================================================================
# Data-layer tests: _compute_fan_chart_data
# ===================================================================


class TestComputeFanChartData:
    def test_returns_correct_keys(self):
        """Fan chart data has all expected keys."""
        store = _make_result_store()
        data = _compute_fan_chart_data(store, asset_idx=0)

        assert "years" in data
        assert "p10" in data
        assert "p25" in data
        assert "p50" in data
        assert "p75" in data
        assert "p90" in data

    def test_years_array_correct_length(self):
        """Years array has length equal to n_steps."""
        n_steps = 24
        store = _make_result_store(n_steps=n_steps)
        data = _compute_fan_chart_data(store, asset_idx=0)

        assert len(data["years"]) == n_steps

    def test_years_converts_months_to_years(self):
        """Years are correctly computed from monthly steps."""
        n_steps = 12
        store = _make_result_store(n_steps=n_steps)
        data = _compute_fan_chart_data(store, asset_idx=0)

        expected_years = (np.arange(n_steps) + 1) / 12.0
        np.testing.assert_allclose(data["years"], expected_years)

    def test_percentiles_correct_length(self):
        """All percentile arrays have length n_steps."""
        n_steps = 36
        store = _make_result_store(n_steps=n_steps)
        data = _compute_fan_chart_data(store, asset_idx=0)

        for key in ["p10", "p25", "p50", "p75", "p90"]:
            assert len(data[key]) == n_steps

    def test_percentiles_monotonically_ordered(self):
        """p10 <= p25 <= p50 <= p75 <= p90 at every time step."""
        store = _make_result_store(n_runs=500, n_steps=24)
        data = _compute_fan_chart_data(store, asset_idx=0)

        for t in range(24):
            assert data["p10"][t] <= data["p25"][t], f"p10 > p25 at step {t}"
            assert data["p25"][t] <= data["p50"][t], f"p25 > p50 at step {t}"
            assert data["p50"][t] <= data["p75"][t], f"p50 > p75 at step {t}"
            assert data["p75"][t] <= data["p90"][t], f"p75 > p90 at step {t}"

    def test_known_data_percentiles(self):
        """Verify percentiles against known data."""
        n_runs, n_steps, n_assets = 100, 1, 1
        balances = np.zeros((n_runs, n_steps, n_assets), dtype=np.float64)
        # Known data: 0, 1, 2, ..., 99 at the single time step
        balances[:, 0, 0] = np.arange(100, dtype=np.float64)

        store = ResultStore(
            balances=balances,
            cum_inflation=np.ones((n_runs, n_steps)),
            cash_flow_shortfall=np.zeros((n_runs, n_steps), dtype=np.float64),
            event_fired_at=np.full((n_runs, 0), -1, dtype=np.int32),
            asset_names=["test"],
            asset_index={"test": 0},
            n_runs=n_runs,
            n_steps=n_steps,
            n_assets=n_assets,
        )

        data = _compute_fan_chart_data(store, asset_idx=0)

        expected_data = np.arange(100, dtype=np.float64)
        assert data["p10"][0] == pytest.approx(np.percentile(expected_data, 10))
        assert data["p25"][0] == pytest.approx(np.percentile(expected_data, 25))
        assert data["p50"][0] == pytest.approx(np.percentile(expected_data, 50))
        assert data["p75"][0] == pytest.approx(np.percentile(expected_data, 75))
        assert data["p90"][0] == pytest.approx(np.percentile(expected_data, 90))

    def test_multiple_assets(self):
        """Fan chart data can be computed for different asset indices."""
        store = _make_result_store(n_assets=3, asset_names=["a", "b", "c"])

        data_0 = _compute_fan_chart_data(store, asset_idx=0)
        data_1 = _compute_fan_chart_data(store, asset_idx=1)
        data_2 = _compute_fan_chart_data(store, asset_idx=2)

        # All should have the same years
        np.testing.assert_array_equal(data_0["years"], data_1["years"])
        np.testing.assert_array_equal(data_0["years"], data_2["years"])

        # But different percentile values (different random data per asset)
        # At least some percentiles should differ
        assert not np.allclose(data_0["p50"], data_1["p50"])


# ===================================================================
# Data-layer tests: _select_sample_paths
# ===================================================================


class TestSelectSamplePaths:
    def test_returns_correct_shape(self):
        """Sample paths have shape (n_samples, n_steps)."""
        store = _make_result_store(n_runs=100, n_steps=12)
        paths = _select_sample_paths(store, asset_idx=0, n_samples=5)

        assert paths.shape == (5, 12)

    def test_fewer_runs_than_samples(self):
        """If n_runs < n_samples, returns n_runs paths."""
        store = _make_result_store(n_runs=3, n_steps=12)
        paths = _select_sample_paths(store, asset_idx=0, n_samples=5)

        assert paths.shape[0] == 3

    def test_deterministic_with_same_seed(self):
        """Same seed produces same sample paths."""
        store = _make_result_store(n_runs=100, n_steps=12)
        paths_a = _select_sample_paths(store, asset_idx=0, n_samples=5, seed=42)
        paths_b = _select_sample_paths(store, asset_idx=0, n_samples=5, seed=42)

        np.testing.assert_array_equal(paths_a, paths_b)


# ===================================================================
# Chart creation tests
# ===================================================================


class TestCreateFanChart:
    def test_creates_figure(self):
        """Fan chart returns a plotly Figure."""
        store = _make_result_store()
        fan_data = _compute_fan_chart_data(store, asset_idx=0)
        fig = _create_fan_chart(fan_data, "test_asset")

        assert isinstance(fig, go.Figure)
        # Should have at least 3 traces (p10-p90 band, p25-p75 band, p50 line)
        assert len(fig.data) >= 3

    def test_with_sample_paths(self):
        """Fan chart with sample paths has additional traces."""
        store = _make_result_store()
        fan_data = _compute_fan_chart_data(store, asset_idx=0)
        sample_paths = _select_sample_paths(store, asset_idx=0, n_samples=3)
        fig = _create_fan_chart(fan_data, "test_asset", sample_paths=sample_paths)

        # 3 base traces + 3 sample paths = 6
        assert len(fig.data) == 6

    def test_title_contains_asset_name(self):
        """Chart title includes the asset name."""
        store = _make_result_store()
        fan_data = _compute_fan_chart_data(store, asset_idx=0)
        fig = _create_fan_chart(fan_data, "my_portfolio")

        assert "my_portfolio" in fig.layout.title.text


class TestCreateHistogram:
    def test_creates_figure(self):
        """Histogram returns a plotly Figure."""
        rng = np.random.default_rng(42)
        values = rng.normal(100000, 20000, size=1000)
        fig = _create_histogram(values, "Test Distribution")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_title(self):
        """Histogram title matches input."""
        values = np.arange(100, dtype=np.float64)
        fig = _create_histogram(values, "Portfolio at Year 10")

        assert "Portfolio at Year 10" in fig.layout.title.text


class TestCreateProbabilityTimeline:
    def test_creates_figure(self):
        """Probability timeline returns a plotly Figure."""
        # Create a store where half the runs always exceed 50k
        n_runs, n_steps = 100, 12
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        balances[:50, :, 0] = 100000.0  # above threshold
        balances[50:, :, 0] = 10000.0  # below threshold

        store = ResultStore(
            balances=balances,
            cum_inflation=np.ones((n_runs, n_steps)),
            cash_flow_shortfall=np.zeros((n_runs, n_steps), dtype=np.float64),
            event_fired_at=np.full((n_runs, 0), -1, dtype=np.int32),
            asset_names=["portfolio"],
            asset_index={"portfolio": 0},
            n_runs=n_runs,
            n_steps=n_steps,
            n_assets=1,
        )

        fig = _create_probability_timeline(store, "portfolio > 50000")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_known_probability(self):
        """Timeline shows correct probability at all steps."""
        n_runs, n_steps = 100, 6
        balances = np.zeros((n_runs, n_steps, 1), dtype=np.float64)
        # 60 runs above 50k at all time steps
        balances[:60, :, 0] = 100000.0
        balances[60:, :, 0] = 10000.0

        store = ResultStore(
            balances=balances,
            cum_inflation=np.ones((n_runs, n_steps)),
            cash_flow_shortfall=np.zeros((n_runs, n_steps), dtype=np.float64),
            event_fired_at=np.full((n_runs, 0), -1, dtype=np.int32),
            asset_names=["portfolio"],
            asset_index={"portfolio": 0},
            n_runs=n_runs,
            n_steps=n_steps,
            n_assets=1,
        )

        fig = _create_probability_timeline(store, "portfolio > 50000")

        # The y values (probabilities) should all be 60%
        y_data = fig.data[0].y
        for val in y_data:
            assert val == pytest.approx(60.0)


# ===================================================================
# Integration tests: generate_report
# ===================================================================


class TestGenerateReport:
    def test_creates_html_file(self, tmp_path):
        """generate_report creates an HTML file at the specified path."""
        store = _make_result_store()
        config = _make_scenario_config()
        output = tmp_path / "report.html"

        result_path = generate_report(
            results=store,
            query_results=[],
            scenario_config=config,
            output_path=output,
        )

        assert result_path == output
        assert output.exists()

    def test_html_contains_plotly(self, tmp_path):
        """Generated HTML contains 'plotly' reference."""
        store = _make_result_store()
        config = _make_scenario_config()
        output = tmp_path / "report.html"

        generate_report(
            results=store,
            query_results=[],
            scenario_config=config,
            output_path=output,
        )

        html_content = output.read_text(encoding="utf-8")
        assert "plotly" in html_content.lower()

    def test_html_file_larger_than_1kb(self, tmp_path):
        """Generated HTML file is non-empty and >1KB."""
        store = _make_result_store()
        config = _make_scenario_config()
        output = tmp_path / "report.html"

        generate_report(
            results=store,
            query_results=[],
            scenario_config=config,
            output_path=output,
        )

        file_size = output.stat().st_size
        assert file_size > 1024, f"Report is only {file_size} bytes"

    def test_creates_output_directory(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        store = _make_result_store()
        config = _make_scenario_config()
        output = tmp_path / "nested" / "dir" / "report.html"

        generate_report(
            results=store,
            query_results=[],
            scenario_config=config,
            output_path=output,
        )

        assert output.exists()

    def test_with_distribution_query(self, tmp_path):
        """Report with distribution query generates a histogram."""
        store = _make_result_store()
        config = _make_scenario_config()
        output = tmp_path / "report.html"

        qr = QueryResult(
            label="Portfolio at Year 1",
            query_type="distribution",
            histogram_bins=np.linspace(0, 100, 11),
            histogram_counts=np.array([10, 20, 30, 15, 10, 5, 4, 3, 2, 1]),
        )

        generate_report(
            results=store,
            query_results=[qr],
            scenario_config=config,
            output_path=output,
        )

        html_content = output.read_text(encoding="utf-8")
        assert "Portfolio at Year 1" in html_content

    def test_with_probability_query(self, tmp_path):
        """Report with probability query generates timeline chart."""
        store = _make_result_store(asset_names=["portfolio"])
        config = _make_scenario_config()
        output = tmp_path / "report.html"

        # Use an expression that is a valid parseable expression as the label
        qr = QueryResult(
            label="portfolio > 100000",
            query_type="probability",
            probability=43.2,
        )

        generate_report(
            results=store,
            query_results=[qr],
            scenario_config=config,
            output_path=output,
        )

        html_content = output.read_text(encoding="utf-8")
        assert len(html_content) > 1024

    def test_html_contains_scenario_name(self, tmp_path):
        """Report title includes the scenario name."""
        store = _make_result_store()
        config = ScenarioConfig(
            name="My Custom Scenario",
            time_horizon=12,
            simulations=100,
        )
        output = tmp_path / "report.html"

        generate_report(
            results=store,
            query_results=[],
            scenario_config=config,
            output_path=output,
        )

        html_content = output.read_text(encoding="utf-8")
        assert "My Custom Scenario" in html_content


# ===================================================================
# Sweep mode tests
# ===================================================================


class TestSweepReport:
    def test_sweep_generates_report(self, tmp_path):
        """Report with sweep data generates successfully."""
        store_base = _make_result_store(seed=42, asset_names=["portfolio"])
        store_alt = _make_result_store(seed=99, asset_names=["portfolio"])
        config = _make_scenario_config()
        output = tmp_path / "sweep_report.html"

        qr_base = [
            QueryResult(
                label="Test probability",
                query_type="probability",
                probability=50.0,
            )
        ]
        qr_alt = [
            QueryResult(
                label="Test probability",
                query_type="probability",
                probability=65.0,
            )
        ]

        sweep = [
            ("base", store_base, qr_base),
            ("optimistic", store_alt, qr_alt),
        ]

        generate_report(
            results=store_base,
            query_results=qr_base,
            scenario_config=config,
            output_path=output,
            sweep_results=sweep,
        )

        assert output.exists()
        html_content = output.read_text(encoding="utf-8")
        assert "Scenario Comparison" in html_content
        assert "base" in html_content
        assert "optimistic" in html_content

    def test_sweep_html_larger_than_1kb(self, tmp_path):
        """Sweep report is non-trivial in size."""
        store_base = _make_result_store(seed=42, asset_names=["portfolio"])
        store_alt = _make_result_store(seed=99, asset_names=["portfolio"])
        config = _make_scenario_config()
        output = tmp_path / "sweep_report.html"

        sweep = [
            ("base", store_base, []),
            ("alt", store_alt, []),
        ]

        generate_report(
            results=store_base,
            query_results=[],
            scenario_config=config,
            output_path=output,
            sweep_results=sweep,
        )

        file_size = output.stat().st_size
        assert file_size > 1024


# ===================================================================
# Snapshot / Determinism test
# ===================================================================


class TestDeterminism:
    @staticmethod
    def _strip_plotly_uuids(html: str) -> str:
        """Strip Plotly-generated UUIDs and unique IDs from HTML for comparison.

        Plotly assigns unique div IDs each time it generates HTML.
        We replace them with a placeholder to enable content comparison.
        """
        import re

        # Strip standard UUIDs
        html = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "UUID-PLACEHOLDER",
            html,
        )
        # Strip any other generated IDs in div tags and JS references
        html = re.sub(
            r'id="[^"]*"',
            'id="ID-PLACEHOLDER"',
            html,
        )
        html = re.sub(
            r'getElementById\("[^"]*"\)',
            'getElementById("ID-PLACEHOLDER")',
            html,
        )
        return html

    def test_same_inputs_same_output(self, tmp_path):
        """Same inputs produce identical HTML output (deterministic)."""
        config = _make_scenario_config()

        # Generate report twice with identical inputs
        store_1 = _make_result_store(seed=42)
        store_2 = _make_result_store(seed=42)

        output_1 = tmp_path / "report_1.html"
        output_2 = tmp_path / "report_2.html"

        generate_report(
            results=store_1,
            query_results=[],
            scenario_config=config,
            output_path=output_1,
        )

        generate_report(
            results=store_2,
            query_results=[],
            scenario_config=config,
            output_path=output_2,
        )

        html_1 = self._strip_plotly_uuids(output_1.read_text(encoding="utf-8"))
        html_2 = self._strip_plotly_uuids(output_2.read_text(encoding="utf-8"))

        assert html_1 == html_2, "Reports with same inputs should be identical"

    def test_different_seed_different_output(self, tmp_path):
        """Different seeds produce different outputs."""
        config = _make_scenario_config()

        store_1 = _make_result_store(seed=42)
        store_2 = _make_result_store(seed=99)

        output_1 = tmp_path / "report_a.html"
        output_2 = tmp_path / "report_b.html"

        generate_report(
            results=store_1,
            query_results=[],
            scenario_config=config,
            output_path=output_1,
        )

        generate_report(
            results=store_2,
            query_results=[],
            scenario_config=config,
            output_path=output_2,
        )

        html_1 = self._strip_plotly_uuids(output_1.read_text(encoding="utf-8"))
        html_2 = self._strip_plotly_uuids(output_2.read_text(encoding="utf-8"))

        assert html_1 != html_2, "Reports with different data should differ"
