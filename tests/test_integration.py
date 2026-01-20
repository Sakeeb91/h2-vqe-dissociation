"""
Integration tests for the full H₂ VQE workflow.

These tests verify the complete pipeline from molecular integrals
through VQE optimization works correctly end-to-end.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from h2_vqe import (
    compute_h2_integrals,
    build_qubit_hamiltonian,
    create_ansatz,
    run_vqe,
    create_noise_model,
    compute_dissociation_curve,
)
from h2_vqe.hamiltonian import exact_ground_state_energy
from h2_vqe.vqe import verify_variational_principle


class TestFullPipeline:
    """Tests for the complete VQE workflow."""

    def test_equilibrium_energy_pipeline(self):
        """Full pipeline at equilibrium geometry."""
        # Step 1: Compute molecular integrals
        mol_data = compute_h2_integrals(0.74)
        assert mol_data.n_qubits == 4
        assert mol_data.n_electrons == 2

        # Step 2: Build qubit Hamiltonian
        qubit_ham = build_qubit_hamiltonian(mol_data)
        assert qubit_ham.n_qubits == 4

        # Step 3: Verify Hamiltonian gives correct exact energy
        exact_energy, _ = exact_ground_state_energy(qubit_ham)
        assert_allclose(exact_energy, mol_data.fci_energy, rtol=1e-8)

        # Step 4: Run VQE
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=100)

        # Step 5: Verify result
        assert verify_variational_principle(result)
        assert result.error is not None
        assert result.error < 0.05  # Within 50 mHa

    def test_stretched_geometry_pipeline(self):
        """Full pipeline at stretched geometry."""
        mol_data = compute_h2_integrals(1.5)
        result = run_vqe(mol_data, ansatz_type="noise_aware", maxiter=100)

        assert verify_variational_principle(result)
        assert np.isfinite(result.energy)

    def test_dissociation_curve_integration(self):
        """Test dissociation curve computation."""
        results = compute_dissociation_curve(
            n_points=3,
            start=0.5,
            stop=1.5,
            ansatz_types=["noise_aware"],
            vqe_maxiter=50,
            verbose=False,
        )

        # Verify results structure
        assert len(results.bond_lengths) == 3
        assert len(results.fci_energies) == 3
        assert "noise_aware" in results.vqe_energies

        # Verify variational principle at all points
        for i in range(3):
            vqe_e = results.vqe_energies["noise_aware"][i]
            fci_e = results.fci_energies[i]
            assert vqe_e >= fci_e - 1e-8


class TestImports:
    """Tests for package imports."""

    def test_main_imports(self):
        """All main functions should be importable."""
        from h2_vqe import (
            compute_h2_integrals,
            build_qubit_hamiltonian,
            create_ansatz,
            run_vqe,
            create_noise_model,
            compute_dissociation_curve,
        )
        assert callable(compute_h2_integrals)
        assert callable(build_qubit_hamiltonian)
        assert callable(create_ansatz)
        assert callable(run_vqe)
        assert callable(create_noise_model)
        assert callable(compute_dissociation_curve)

    def test_version(self):
        """Package version should be defined."""
        from h2_vqe import __version__
        assert isinstance(__version__, str)
        assert "." in __version__


class TestNoiseModel:
    """Tests for noise model integration."""

    def test_noise_model_creation(self):
        """Noise models should be creatable."""
        for preset in ["ideal", "ibm_like", "low_noise", "high_noise"]:
            noise_model = create_noise_model(preset)
            assert noise_model is not None


class TestPhysicalCorrectness:
    """Tests for physical correctness of results."""

    def test_energy_minimum_at_equilibrium(self):
        """Energy should have minimum near equilibrium."""
        energies = []
        bond_lengths = [0.5, 0.74, 1.0, 1.5]

        for r in bond_lengths:
            mol_data = compute_h2_integrals(r)
            energies.append(mol_data.fci_energy)

        # Minimum should be at or near equilibrium (0.74 Å)
        min_idx = np.argmin(energies)
        assert bond_lengths[min_idx] == 0.74

    def test_correlation_energy_is_negative(self):
        """Correlation energy (FCI - HF) should be negative."""
        mol_data = compute_h2_integrals(0.74)
        correlation = mol_data.fci_energy - mol_data.hf_energy
        assert correlation < 0

    def test_vqe_energy_bounded(self):
        """VQE energy should be bounded above by HF and below by a floor."""
        mol_data = compute_h2_integrals(0.74)
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=100)

        # Should be below HF energy (since FCI < HF)
        assert result.energy < mol_data.hf_energy + 0.01

        # Should be above some reasonable floor
        assert result.energy > -2.0


class TestResultSerialization:
    """Tests for result serialization."""

    def test_dissociation_result_to_dict(self):
        """Results should be convertible to dict."""
        results = compute_dissociation_curve(
            n_points=3,
            start=0.5,
            stop=1.5,
            ansatz_types=["noise_aware"],
            vqe_maxiter=20,
            verbose=False,
        )

        d = results.to_dict()
        assert "bond_lengths" in d
        assert "fci_energies" in d
        assert "vqe_energies" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
