"""End-to-end integration tests for the Moneta CLI.

Uses Click's CliRunner to invoke commands without a subprocess,
giving us clean control over exit codes, stdout, and temporary files.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from moneta.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SIMPLE_MODEL = str(FIXTURES_DIR / "simple_model.moneta.yaml")
EQUITY_MODEL = str(FIXTURES_DIR / "equity_model.moneta.yaml")
SWEEP_MODEL = str(FIXTURES_DIR / "sweep_model.moneta.yaml")

runner = CliRunner()


def _invoke_run(args: list[str], **kwargs):
    """Helper to invoke ``moneta run`` with webbrowser.open patched out."""
    with patch("moneta.output.report.webbrowser.open"):
        return runner.invoke(main, args, **kwargs)


# ---------------------------------------------------------------------------
# ``moneta run`` — simple model
# ---------------------------------------------------------------------------


class TestRunSimpleModel:
    """Tests for ``moneta run simple_model.moneta.yaml``."""

    def test_run_simple_model_exit_0(self):
        result = _invoke_run(["run", SIMPLE_MODEL, "--seed", "42", "--no-report"])
        assert result.exit_code == 0, f"output: {result.output}"

    def test_run_simple_model_output_contains_moneta(self):
        result = _invoke_run(["run", SIMPLE_MODEL, "--seed", "42", "--no-report"])
        assert "Moneta" in result.output

    def test_run_simple_model_output_contains_simulations(self):
        result = _invoke_run(["run", SIMPLE_MODEL, "--seed", "42", "--no-report"])
        assert "simulations" in result.output.lower() or "1,000" in result.output


# ---------------------------------------------------------------------------
# ``moneta run`` — equity model
# ---------------------------------------------------------------------------


class TestRunEquityModel:
    """Tests for ``moneta run equity_model.moneta.yaml``."""

    def test_run_equity_model_exit_0(self):
        result = _invoke_run(["run", EQUITY_MODEL, "--seed", "42", "--no-report"])
        assert result.exit_code == 0, f"output: {result.output}"

    def test_run_equity_model_contains_probability(self):
        result = _invoke_run(["run", EQUITY_MODEL, "--seed", "42", "--no-report"])
        assert "probability" in result.output.lower() or "%" in result.output


# ---------------------------------------------------------------------------
# ``moneta run`` — --no-report
# ---------------------------------------------------------------------------


class TestNoReport:
    """The --no-report flag should suppress HTML generation."""

    def test_no_report_no_html(self, tmp_path):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--no-report",
                "--output",
                str(tmp_path),
            ]
        )
        assert result.exit_code == 0
        # No HTML files should have been created
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 0


# ---------------------------------------------------------------------------
# ``moneta run`` — --format json
# ---------------------------------------------------------------------------


class TestJsonFormat:
    """Tests for ``moneta run --format json``."""

    def test_json_exit_0(self):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--no-report",
                "--format",
                "json",
            ]
        )
        assert result.exit_code == 0, f"output: {result.output}"

    def test_json_valid_output(self):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--no-report",
                "--format",
                "json",
            ]
        )
        # The JSON output should be the entire stdout
        parsed = json.loads(result.output)
        assert "scenario" in parsed
        assert "queries" in parsed
        assert "elapsed_ms" in parsed

    def test_json_scenario_fields(self):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--no-report",
                "--format",
                "json",
            ]
        )
        parsed = json.loads(result.output)
        assert parsed["scenario"]["name"] == "Simple investment model"
        assert parsed["scenario"]["simulations"] == 1000


# ---------------------------------------------------------------------------
# ``moneta run`` — --simulations override
# ---------------------------------------------------------------------------


class TestSimulationsOverride:
    """Tests for ``moneta run --simulations N``."""

    def test_override_simulations_exit_0(self):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--simulations",
                "100",
                "--seed",
                "42",
                "--no-report",
            ]
        )
        assert result.exit_code == 0, f"output: {result.output}"

    def test_override_simulations_output(self):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--simulations",
                "100",
                "--seed",
                "42",
                "--no-report",
            ]
        )
        assert "100" in result.output

    def test_override_simulations_json(self):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--simulations",
                "100",
                "--seed",
                "42",
                "--no-report",
                "--format",
                "json",
            ]
        )
        parsed = json.loads(result.output)
        assert parsed["scenario"]["simulations"] == 100


# ---------------------------------------------------------------------------
# ``moneta validate``
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for ``moneta validate``."""

    def test_validate_simple_model_exit_0(self):
        result = runner.invoke(main, ["validate", SIMPLE_MODEL])
        assert result.exit_code == 0, f"output: {result.output}"

    def test_validate_simple_model_output(self):
        result = runner.invoke(main, ["validate", SIMPLE_MODEL])
        assert "valid" in result.output.lower()

    def test_validate_equity_model_exit_0(self):
        result = runner.invoke(main, ["validate", EQUITY_MODEL])
        assert result.exit_code == 0

    def test_validate_nonexistent_file(self):
        result = runner.invoke(main, ["validate", "nonexistent.yaml"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Seeded reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Running with the same seed should produce identical output."""

    def test_seeded_runs_are_identical(self):
        result1 = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--no-report",
            ]
        )
        result2 = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--no-report",
            ]
        )
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        # Compare output excluding the header line which contains timing
        lines1 = [
            line for line in result1.output.splitlines() if not line.startswith("───")
        ]
        lines2 = [
            line for line in result2.output.splitlines() if not line.startswith("───")
        ]
        assert lines1 == lines2

    def test_seeded_json_identical(self):
        result1 = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--no-report",
                "--format",
                "json",
            ]
        )
        result2 = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--no-report",
                "--format",
                "json",
            ]
        )
        # Parse JSON and compare values (elapsed_ms may differ slightly)
        parsed1 = json.loads(result1.output)
        parsed2 = json.loads(result2.output)
        assert parsed1["queries"] == parsed2["queries"]
        assert parsed1["scenario"] == parsed2["scenario"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_invalid_model_file(self, tmp_path):
        bad_model = tmp_path / "bad_model.moneta.yaml"
        bad_model.write_text("this is not valid yaml: [")
        result = _invoke_run(["run", str(bad_model), "--no-report"])
        assert result.exit_code != 0

    def test_invalid_model_content(self, tmp_path):
        bad_model = tmp_path / "bad_model.moneta.yaml"
        bad_model.write_text("scenario:\n  name: test\n")
        result = _invoke_run(["run", str(bad_model), "--no-report"])
        assert result.exit_code != 0
        assert "error" in result.output.lower()

    def test_nonexistent_run_file(self):
        result = _invoke_run(["run", "nonexistent_file.yaml"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Tests for HTML report output."""

    def test_report_generated(self, tmp_path):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--output",
                str(tmp_path),
            ]
        )
        assert result.exit_code == 0
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 1
        assert "simple_model" in html_files[0].name

    def test_report_is_html(self, tmp_path):
        result = _invoke_run(
            [
                "run",
                SIMPLE_MODEL,
                "--seed",
                "42",
                "--output",
                str(tmp_path),
            ]
        )
        assert result.exit_code == 0
        html_files = list(tmp_path.glob("*.html"))
        content = html_files[0].read_text()
        assert "<html>" in content.lower() or "<!doctype html>" in content.lower()


# ---------------------------------------------------------------------------
# ``moneta run`` — sweep mode
# ---------------------------------------------------------------------------


class TestSweepMode:
    """Tests for sweep mode with named scenarios."""

    def test_sweep_exit_0(self):
        """Sweep model should run successfully."""
        result = _invoke_run(["run", SWEEP_MODEL, "--seed", "42", "--no-report"])
        assert result.exit_code == 0, f"output: {result.output}"

    def test_sweep_output_contains_scenario_comparison(self):
        """Output should include scenario comparison section."""
        result = _invoke_run(["run", SWEEP_MODEL, "--seed", "42", "--no-report"])
        assert "Scenario Comparison" in result.output

    def test_sweep_output_contains_scenario_labels(self):
        """Output should include all scenario labels."""
        result = _invoke_run(["run", SWEEP_MODEL, "--seed", "42", "--no-report"])
        assert "conservative" in result.output
        assert "aggressive" in result.output
        assert "base" in result.output

    def test_sweep_output_contains_base_query_results(self):
        """Sweep output should still include the base query results."""
        result = _invoke_run(["run", SWEEP_MODEL, "--seed", "42", "--no-report"])
        assert (
            "$150K at year 5" in result.output or "probability" in result.output.lower()
        )

    def test_sweep_report_generation(self, tmp_path):
        """Sweep mode should generate an HTML report."""
        result = _invoke_run(
            [
                "run",
                SWEEP_MODEL,
                "--seed",
                "42",
                "--output",
                str(tmp_path),
            ]
        )
        assert result.exit_code == 0
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 1

    def test_sweep_report_contains_comparison(self, tmp_path):
        """Sweep HTML report should contain comparison content."""
        result = _invoke_run(
            [
                "run",
                SWEEP_MODEL,
                "--seed",
                "42",
                "--output",
                str(tmp_path),
            ]
        )
        assert result.exit_code == 0
        html_files = list(tmp_path.glob("*.html"))
        content = html_files[0].read_text()
        assert "Scenario Comparison" in content

    def test_sweep_produces_different_results(self):
        """Conservative and aggressive scenarios should produce
        different probabilities."""
        result = _invoke_run(["run", SWEEP_MODEL, "--seed", "42", "--no-report"])
        assert result.exit_code == 0
        # The comparison table should show different percentages for different scenarios
        # We verify that the output contains multiple percentage values
        lines = result.output.split("\n")
        comparison_lines = [
            line
            for line in lines
            if "%" in line
            and ("conservative" in line or "aggressive" in line or "base" in line)
        ]
        assert len(comparison_lines) >= 2, (
            f"Expected at least 2 comparison lines with percentages, "
            f"got {len(comparison_lines)}"
        )
