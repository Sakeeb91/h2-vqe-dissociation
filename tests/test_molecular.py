"""
Tests for molecular.py - PySCF interface for H₂.

These tests verify:
1. Correct computation of molecular integrals
2. Accurate HF and FCI energies at equilibrium
3. Proper handling of edge cases
4. Integral symmetries and shapes
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from h2_vqe.molecular import (
    compute_h2_integrals,
    compute_bond_lengths,
    compute_classical_energies,
    MolecularData,
    EQUILIBRIUM_BOND_LENGTH,
)


class TestComputeH2Integrals:
    """Tests for the main compute_h2_integrals function."""

    def test_equilibrium_geometry_hf_energy(self):
        """HF energy at equilibrium should be ~-1.117 Ha."""
        data = compute_h2_integrals(EQUILIBRIUM_BOND_LENGTH)
        # HF energy for H2/STO-3G at 0.74 Å
        assert_allclose(data.hf_energy, -1.117, atol=0.01)

    def test_equilibrium_geometry_fci_energy(self):
        """FCI energy at equilibrium should be ~-1.137 Ha."""
        data = compute_h2_integrals(EQUILIBRIUM_BOND_LENGTH)
        # FCI energy for H2/STO-3G at 0.74 Å
        assert_allclose(data.fci_energy, -1.137, atol=0.01)

    def test_fci_lower_than_hf(self):
        """FCI energy must be lower than HF (variational principle)."""
        data = compute_h2_integrals(EQUILIBRIUM_BOND_LENGTH)
        assert data.fci_energy < data.hf_energy

    def test_correlation_energy_negative(self):
        """Correlation energy (FCI - HF) should be negative."""
        data = compute_h2_integrals(EQUILIBRIUM_BOND_LENGTH)
        assert data.correlation_energy() < 0

    def test_sto3g_qubit_count(self):
        """STO-3G basis for H2 should give 4 qubits."""
        data = compute_h2_integrals(0.74, basis="sto-3g")
        assert data.n_qubits == 4
        assert data.n_orbitals == 2

    def test_electron_count(self):
        """H2 should have 2 electrons."""
        data = compute_h2_integrals(0.74)
        assert data.n_electrons == 2


class TestIntegralSymmetries:
    """Tests for molecular integral symmetries."""

    def test_one_body_hermitian(self):
        """One-body integrals should be Hermitian (real symmetric for real orbitals)."""
        data = compute_h2_integrals(0.74)
        h1 = data.one_body_integrals
        # For real orbitals, Hermitian = symmetric
        assert_allclose(h1, h1.T, atol=1e-10)

    def test_one_body_shape(self):
        """One-body integrals should be n_orbitals x n_orbitals."""
        data = compute_h2_integrals(0.74)
        n = data.n_orbitals
        assert data.one_body_integrals.shape == (n, n)

    def test_two_body_shape(self):
        """Two-body integrals should be n_orbitals^4."""
        data = compute_h2_integrals(0.74)
        n = data.n_orbitals
        assert data.two_body_integrals.shape == (n, n, n, n)

    def test_two_body_symmetry(self):
        """Two-body integrals should satisfy (pq|rs) = (rs|pq) symmetry."""
        data = compute_h2_integrals(0.74)
        eri = data.two_body_integrals
        # Chemist notation: (pq|rs) = (rs|pq)
        assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_bond_length_raises(self):
        """Negative bond length should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_h2_integrals(-0.5)

    def test_zero_bond_length_raises(self):
        """Zero bond length should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_h2_integrals(0.0)

    def test_very_short_bond_length(self):
        """Very short bond length should still compute (high energy)."""
        data = compute_h2_integrals(0.3)
        # Energy should be much higher than equilibrium
        assert data.fci_energy > -1.0

    def test_very_long_bond_length(self):
        """Long bond length should approach -1.0 Ha (two H atoms)."""
        data = compute_h2_integrals(3.0)
        # At dissociation, energy approaches 2 * E(H) = -1.0 Ha
        # STO-3G has basis set limitations, so we use a larger tolerance
        assert_allclose(data.fci_energy, -1.0, atol=0.1)


class TestMolecularData:
    """Tests for the MolecularData dataclass."""

    def test_dataclass_creation(self):
        """MolecularData should be properly created."""
        data = compute_h2_integrals(0.74)
        assert isinstance(data, MolecularData)
        assert data.bond_length == 0.74
        assert data.basis == "sto-3g"

    def test_repr_contains_key_info(self):
        """repr should show bond length and energies."""
        data = compute_h2_integrals(0.74)
        repr_str = repr(data)
        assert "H₂" in repr_str
        assert "0.74" in repr_str
        assert "E_HF" in repr_str
        assert "E_FCI" in repr_str


class TestComputeBondLengths:
    """Tests for bond length array generation."""

    def test_default_range(self):
        """Default bond lengths should span 0.3 to 2.5 Å."""
        lengths = compute_bond_lengths()
        assert len(lengths) == 20
        assert lengths[0] == 0.3
        assert lengths[-1] == 2.5

    def test_custom_range(self):
        """Custom range should work correctly."""
        lengths = compute_bond_lengths(0.5, 2.0, num=4)
        expected = np.array([0.5, 1.0, 1.5, 2.0])
        assert_allclose(lengths, expected, atol=1e-10)


class TestComputeClassicalEnergies:
    """Tests for batch classical energy computation."""

    def test_multiple_bond_lengths(self):
        """Should compute energies at multiple bond lengths."""
        lengths = np.array([0.5, 0.74, 1.0])
        results = compute_classical_energies(lengths)

        assert "bond_lengths" in results
        assert "hf_energies" in results
        assert "fci_energies" in results
        assert "correlation_energies" in results

        assert len(results["hf_energies"]) == 3
        assert len(results["fci_energies"]) == 3

    def test_equilibrium_in_batch(self):
        """Batch computation should match single-point calculation."""
        data_single = compute_h2_integrals(0.74)
        results_batch = compute_classical_energies(np.array([0.74]))

        assert_allclose(results_batch["fci_energies"][0], data_single.fci_energy)
        assert_allclose(results_batch["hf_energies"][0], data_single.hf_energy)


class TestPhysicsValidation:
    """Tests to validate physical correctness."""

    def test_dissociation_curve_minimum(self):
        """FCI energy should have minimum near equilibrium."""
        lengths = np.array([0.5, 0.74, 1.0, 1.5])
        results = compute_classical_energies(lengths)
        fci = results["fci_energies"]

        # Energy at 0.74 Å should be lower than at 0.5 and 1.0 Å
        min_idx = np.argmin(fci)
        assert lengths[min_idx] == pytest.approx(0.74, abs=0.3)

    def test_nuclear_repulsion_decreases_with_distance(self):
        """Nuclear repulsion should decrease as atoms move apart."""
        data_short = compute_h2_integrals(0.5)
        data_long = compute_h2_integrals(1.5)
        assert data_short.nuclear_repulsion > data_long.nuclear_repulsion

    def test_energy_above_exact_limit(self):
        """All computed energies should be above the exact limit."""
        # The exact non-relativistic energy is about -1.174 Ha
        # FCI/STO-3G is less accurate but should still be reasonable
        data = compute_h2_integrals(0.74)
        assert data.fci_energy > -1.2  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
