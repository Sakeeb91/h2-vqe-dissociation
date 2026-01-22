"""
Tests for ZNE visualization functionality.

These tests verify:
1. plot_zne_comparison creates valid figures
2. Works with simulator-only data
3. Works with full hardware data
4. Saves figures correctly
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from h2_vqe.visualization import plot_zne_comparison


@pytest.fixture
def sample_bond_lengths():
    """Sample bond lengths for testing."""
    return np.array([0.5, 0.74, 1.0, 1.5, 2.0])


@pytest.fixture
def sample_fci_energies():
    """Sample FCI energies (approximate H2 values)."""
    return np.array([-1.05, -1.137, -1.10, -1.02, -0.95])


@pytest.fixture
def sample_sim_noiseless(sample_fci_energies):
    """Sample noiseless VQE energies (close to FCI)."""
    return sample_fci_energies + np.random.uniform(0.001, 0.003, len(sample_fci_energies))


@pytest.fixture
def sample_sim_noisy(sample_fci_energies):
    """Sample noisy VQE energies (further from FCI)."""
    return sample_fci_energies + np.random.uniform(0.01, 0.05, len(sample_fci_energies))


@pytest.fixture
def sample_hw_raw(sample_fci_energies):
    """Sample hardware raw energies."""
    return sample_fci_energies + np.random.uniform(0.02, 0.08, len(sample_fci_energies))


@pytest.fixture
def sample_hw_zne(sample_fci_energies):
    """Sample hardware ZNE energies (between raw and noiseless)."""
    return sample_fci_energies + np.random.uniform(0.005, 0.02, len(sample_fci_energies))


class TestPlotZNEComparison:
    """Tests for plot_zne_comparison function."""

    def test_returns_figure(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
    ):
        """Should return a matplotlib Figure."""
        fig = plot_zne_comparison(
            bond_lengths=sample_bond_lengths,
            fci_energies=sample_fci_energies,
            sim_noiseless=sample_sim_noiseless,
            sim_noisy=sample_sim_noisy,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_works_without_hardware_data(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
    ):
        """Should work without hardware data (simulator-only mode)."""
        fig = plot_zne_comparison(
            bond_lengths=sample_bond_lengths,
            fci_energies=sample_fci_energies,
            sim_noiseless=sample_sim_noiseless,
            sim_noisy=sample_sim_noisy,
            hw_raw=None,
            hw_zne=None,
        )

        # Should have 2 subplots (no ZNE improvement panel)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_works_with_hardware_data(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
        sample_hw_raw,
        sample_hw_zne,
    ):
        """Should work with full hardware data."""
        fig = plot_zne_comparison(
            bond_lengths=sample_bond_lengths,
            fci_energies=sample_fci_energies,
            sim_noiseless=sample_sim_noiseless,
            sim_noisy=sample_sim_noisy,
            hw_raw=sample_hw_raw,
            hw_zne=sample_hw_zne,
        )

        # Should have 3 subplots with hardware data
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_saves_figure(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
    ):
        """Should save figure when save_path is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_zne.png"

            fig = plot_zne_comparison(
                bond_lengths=sample_bond_lengths,
                fci_energies=sample_fci_energies,
                sim_noiseless=sample_sim_noiseless,
                sim_noisy=sample_sim_noisy,
                save_path=str(save_path),
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0
            plt.close(fig)

    def test_creates_parent_directories(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
    ):
        """Should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "test_zne.png"

            fig = plot_zne_comparison(
                bond_lengths=sample_bond_lengths,
                fci_energies=sample_fci_energies,
                sim_noiseless=sample_sim_noiseless,
                sim_noisy=sample_sim_noisy,
                save_path=str(save_path),
            )

            assert save_path.exists()
            plt.close(fig)

    def test_handles_single_point(self):
        """Should handle single data point (edge case)."""
        fig = plot_zne_comparison(
            bond_lengths=np.array([0.74]),
            fci_energies=np.array([-1.137]),
            sim_noiseless=np.array([-1.135]),
            sim_noisy=np.array([-1.12]),
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_many_points(self):
        """Should handle many data points."""
        n = 50
        bond_lengths = np.linspace(0.4, 3.0, n)
        fci = -1.1 + 0.1 * (bond_lengths - 0.74) ** 2  # Approximate curve

        fig = plot_zne_comparison(
            bond_lengths=bond_lengths,
            fci_energies=fci,
            sim_noiseless=fci + 0.002,
            sim_noisy=fci + 0.02,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotAxesContent:
    """Tests for plot axes content and labels."""

    def test_has_correct_axis_labels(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
    ):
        """Should have correct axis labels."""
        fig = plot_zne_comparison(
            bond_lengths=sample_bond_lengths,
            fci_energies=sample_fci_energies,
            sim_noiseless=sample_sim_noiseless,
            sim_noisy=sample_sim_noisy,
        )

        # Check first subplot has correct labels
        ax = fig.axes[0]
        assert "Bond Length" in ax.get_xlabel()
        assert "Energy" in ax.get_ylabel()

        # Check second subplot (error plot)
        ax = fig.axes[1]
        assert "Bond Length" in ax.get_xlabel()
        assert "Error" in ax.get_ylabel()

        plt.close(fig)

    def test_has_legend(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
    ):
        """Should have legend in plots."""
        fig = plot_zne_comparison(
            bond_lengths=sample_bond_lengths,
            fci_energies=sample_fci_energies,
            sim_noiseless=sample_sim_noiseless,
            sim_noisy=sample_sim_noisy,
        )

        # At least one subplot should have a legend
        has_legend = any(ax.get_legend() is not None for ax in fig.axes)
        assert has_legend

        plt.close(fig)


class TestZNEImprovementPanel:
    """Tests for the ZNE improvement panel (third panel)."""

    def test_improvement_panel_exists_with_hardware(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
        sample_hw_raw,
        sample_hw_zne,
    ):
        """Should have improvement panel when hardware data is provided."""
        fig = plot_zne_comparison(
            bond_lengths=sample_bond_lengths,
            fci_energies=sample_fci_energies,
            sim_noiseless=sample_sim_noiseless,
            sim_noisy=sample_sim_noisy,
            hw_raw=sample_hw_raw,
            hw_zne=sample_hw_zne,
        )

        # Should have 3 axes
        assert len(fig.axes) == 3

        # Third axis should have improvement-related title
        ax = fig.axes[2]
        assert "Improvement" in ax.get_title() or "ZNE" in ax.get_title()

        plt.close(fig)

    def test_improvement_panel_not_present_without_hardware(
        self,
        sample_bond_lengths,
        sample_fci_energies,
        sample_sim_noiseless,
        sample_sim_noisy,
    ):
        """Should not have improvement panel without hardware data."""
        fig = plot_zne_comparison(
            bond_lengths=sample_bond_lengths,
            fci_energies=sample_fci_energies,
            sim_noiseless=sample_sim_noiseless,
            sim_noisy=sample_sim_noisy,
        )

        # Should only have 2 axes
        assert len(fig.axes) == 2

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
