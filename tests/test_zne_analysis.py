"""
Tests for ZNE analysis metrics computation.

Tests the functions used in analyze_zne_results.py script.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from analyze_zne_results import (
    compute_mae,
    compute_max_error,
    compute_rmse,
    analyze_results,
    generate_latex_table,
)


@pytest.fixture
def sample_results():
    """Sample benchmark results for testing."""
    return {
        "bond_lengths": [0.5, 0.74, 1.0, 1.5, 2.0],
        "fci": [-1.05, -1.137, -1.10, -1.02, -0.95],
        "sim_noiseless": [-1.048, -1.135, -1.098, -1.018, -0.948],
        "sim_noisy": [-1.04, -1.12, -1.08, -1.00, -0.93],
        "hw_raw": [-1.02, -1.10, -1.05, -0.98, -0.90],
        "hw_zne": [-1.04, -1.125, -1.09, -1.01, -0.94],
    }


class TestComputeMAE:
    """Tests for compute_mae function."""

    def test_returns_float(self):
        """Should return a float."""
        energies = np.array([-1.1, -1.2, -1.3])
        reference = np.array([-1.1, -1.2, -1.3])
        result = compute_mae(energies, reference)
        assert isinstance(result, float)

    def test_zero_for_identical(self):
        """Should return 0 for identical arrays."""
        arr = np.array([-1.1, -1.2, -1.3])
        result = compute_mae(arr, arr)
        assert result == pytest.approx(0.0)

    def test_correct_value(self):
        """Should compute correct MAE."""
        energies = np.array([0.01, 0.02, 0.03])
        reference = np.array([0.0, 0.0, 0.0])
        # MAE = (0.01 + 0.02 + 0.03) / 3 = 0.02 Ha = 20 mHa
        result = compute_mae(energies, reference)
        assert result == pytest.approx(20.0)

    def test_positive_result(self):
        """Should always return positive result."""
        energies = np.array([-1.0, -1.0, -1.0])
        reference = np.array([-1.1, -1.1, -1.1])
        result = compute_mae(energies, reference)
        assert result > 0

    def test_symmetric(self):
        """MAE should be symmetric."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        assert compute_mae(a, b) == pytest.approx(compute_mae(b, a))


class TestComputeMaxError:
    """Tests for compute_max_error function."""

    def test_returns_float(self):
        """Should return a float."""
        energies = np.array([-1.1, -1.2, -1.3])
        reference = np.array([-1.1, -1.2, -1.3])
        result = compute_max_error(energies, reference)
        assert isinstance(result, float)

    def test_zero_for_identical(self):
        """Should return 0 for identical arrays."""
        arr = np.array([-1.1, -1.2, -1.3])
        result = compute_max_error(arr, arr)
        assert result == pytest.approx(0.0)

    def test_correct_value(self):
        """Should compute correct max error."""
        energies = np.array([0.01, 0.02, 0.05])
        reference = np.array([0.0, 0.0, 0.0])
        # Max = 0.05 Ha = 50 mHa
        result = compute_max_error(energies, reference)
        assert result == pytest.approx(50.0)


class TestComputeRMSE:
    """Tests for compute_rmse function."""

    def test_returns_float(self):
        """Should return a float."""
        energies = np.array([-1.1, -1.2, -1.3])
        reference = np.array([-1.1, -1.2, -1.3])
        result = compute_rmse(energies, reference)
        assert isinstance(result, float)

    def test_zero_for_identical(self):
        """Should return 0 for identical arrays."""
        arr = np.array([-1.1, -1.2, -1.3])
        result = compute_rmse(arr, arr)
        assert result == pytest.approx(0.0)

    def test_correct_value(self):
        """Should compute correct RMSE."""
        energies = np.array([0.01, 0.02, 0.03])
        reference = np.array([0.0, 0.0, 0.0])
        # RMSE = sqrt((0.0001 + 0.0004 + 0.0009) / 3) Ha
        #      = sqrt(0.00046667) Ha
        #      ≈ 0.0216 Ha = 21.6 mHa
        result = compute_rmse(energies, reference)
        expected = np.sqrt((0.01**2 + 0.02**2 + 0.03**2) / 3) * 1000
        assert result == pytest.approx(expected)

    def test_rmse_geq_mae(self):
        """RMSE should be >= MAE for non-uniform errors."""
        energies = np.array([0.01, 0.05, 0.01])
        reference = np.array([0.0, 0.0, 0.0])
        mae = compute_mae(energies, reference)
        rmse = compute_rmse(energies, reference)
        assert rmse >= mae


class TestAnalyzeResults:
    """Tests for analyze_results function."""

    def test_returns_dict(self, sample_results):
        """Should return a dictionary."""
        metrics = analyze_results(sample_results, verbose=False)
        assert isinstance(metrics, dict)

    def test_contains_required_keys(self, sample_results):
        """Should contain required metric keys."""
        metrics = analyze_results(sample_results, verbose=False)

        assert "bond_lengths" in metrics
        assert "n_points" in metrics
        assert "equilibrium" in metrics
        assert "sim_noiseless" in metrics
        assert "sim_noisy" in metrics

    def test_equilibrium_detection(self, sample_results):
        """Should correctly identify equilibrium geometry."""
        metrics = analyze_results(sample_results, verbose=False)

        # Equilibrium is at r=0.74 (index 1)
        assert metrics["equilibrium"]["bond_length"] == pytest.approx(0.74)
        assert metrics["equilibrium"]["index"] == 1

    def test_mae_values_positive(self, sample_results):
        """MAE values should be positive."""
        metrics = analyze_results(sample_results, verbose=False)

        assert metrics["sim_noiseless"]["mae_mHa"] > 0
        assert metrics["sim_noisy"]["mae_mHa"] > 0

    def test_zne_improvement_computed(self, sample_results):
        """Should compute ZNE improvement when hardware data present."""
        metrics = analyze_results(sample_results, verbose=False)

        assert "hw_raw" in metrics
        assert "hw_zne" in metrics
        assert "zne_improvement" in metrics
        assert "mae_factor" in metrics["zne_improvement"]

    def test_per_geometry_improvement(self, sample_results):
        """Should compute per-geometry improvement."""
        metrics = analyze_results(sample_results, verbose=False)

        assert "per_geometry_improvement" in metrics
        assert len(metrics["per_geometry_improvement"]) == 5

    def test_handles_simulator_only(self):
        """Should handle simulator-only results."""
        sim_only_results = {
            "bond_lengths": [0.5, 0.74, 1.0],
            "fci": [-1.05, -1.137, -1.10],
            "sim_noiseless": [-1.048, -1.135, -1.098],
            "sim_noisy": [-1.04, -1.12, -1.08],
        }
        metrics = analyze_results(sim_only_results, verbose=False)

        assert "sim_noiseless" in metrics
        assert "sim_noisy" in metrics
        assert "hw_raw" not in metrics
        assert "zne_improvement" not in metrics


class TestGenerateLatexTable:
    """Tests for generate_latex_table function."""

    def test_returns_string(self, sample_results):
        """Should return a string."""
        metrics = analyze_results(sample_results, verbose=False)
        latex = generate_latex_table(metrics)
        assert isinstance(latex, str)

    def test_contains_latex_commands(self, sample_results):
        """Should contain LaTeX commands."""
        metrics = analyze_results(sample_results, verbose=False)
        latex = generate_latex_table(metrics)

        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert r"\begin{tabular}" in latex
        assert r"\toprule" in latex

    def test_contains_data_rows(self, sample_results):
        """Should contain data rows for each condition."""
        metrics = analyze_results(sample_results, verbose=False)
        latex = generate_latex_table(metrics)

        assert "noiseless" in latex.lower()
        assert "noisy" in latex.lower() or "noise" in latex.lower()

    def test_handles_hardware_data(self, sample_results):
        """Should include hardware rows when present."""
        metrics = analyze_results(sample_results, verbose=False)
        latex = generate_latex_table(metrics)

        assert "Hardware" in latex or "raw" in latex.lower()
        assert "ZNE" in latex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
