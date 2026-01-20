"""
Tests for vqe.py - VQE optimization engine.

These tests verify:
1. VQE produces valid results
2. Variational principle is satisfied
3. Energy converges reasonably
4. Different ansatze work correctly
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from h2_vqe.molecular import compute_h2_integrals
from h2_vqe.hamiltonian import build_qubit_hamiltonian
from h2_vqe.ansatz import create_ansatz, get_initial_parameters
from h2_vqe.vqe import (
    VQEEngine,
    VQEResult,
    run_vqe,
    run_vqe_multistart,
    verify_variational_principle,
)


@pytest.fixture
def mol_data():
    """Molecular data at equilibrium."""
    return compute_h2_integrals(0.74)


@pytest.fixture
def qubit_ham(mol_data):
    """Qubit Hamiltonian."""
    return build_qubit_hamiltonian(mol_data)


class TestVQEEngine:
    """Tests for VQEEngine class."""

    def test_compute_energy_returns_float(self, qubit_ham):
        """compute_energy should return a float."""
        ansatz = create_ansatz("noise_aware", n_qubits=4)
        engine = VQEEngine(qubit_ham, ansatz)

        params = np.zeros(ansatz.num_parameters)
        energy = engine.compute_energy(params)

        assert isinstance(energy, float)

    def test_energy_varies_with_parameters(self, qubit_ham):
        """Different parameters should give different energies."""
        ansatz = create_ansatz("noise_aware", n_qubits=4)
        engine = VQEEngine(qubit_ham, ansatz)

        n_params = ansatz.num_parameters
        energy1 = engine.compute_energy(np.zeros(n_params))
        energy2 = engine.compute_energy(np.ones(n_params) * 0.5)

        # Energies should generally be different
        # (unless parameters happen to give same state)
        assert not np.isclose(energy1, energy2, atol=1e-6)

    def test_optimize_returns_result(self, qubit_ham):
        """optimize should return VQEResult."""
        ansatz = create_ansatz("uccsd", n_qubits=4)
        engine = VQEEngine(qubit_ham, ansatz)

        result = engine.optimize(maxiter=10)

        assert isinstance(result, VQEResult)

    def test_optimize_decreases_energy(self, qubit_ham):
        """Optimization should decrease or maintain energy."""
        ansatz = create_ansatz("noise_aware", n_qubits=4)
        engine = VQEEngine(qubit_ham, ansatz)

        # Start from random params
        init_params = np.random.uniform(-0.5, 0.5, ansatz.num_parameters)
        initial_energy = engine.compute_energy(init_params)

        # Reset and optimize
        engine._n_evaluations = 0
        engine._energy_history = []
        result = engine.optimize(initial_params=init_params, maxiter=50)

        # Final energy should be <= initial
        assert result.energy <= initial_energy + 1e-6


class TestRunVQE:
    """Tests for run_vqe function."""

    def test_returns_vqe_result(self, mol_data):
        """run_vqe should return VQEResult."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=10)
        assert isinstance(result, VQEResult)

    def test_result_has_energy(self, mol_data):
        """Result should have energy."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=10)
        assert hasattr(result, "energy")
        assert isinstance(result.energy, float)

    def test_result_has_parameters(self, mol_data):
        """Result should have optimized parameters."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=10)
        assert hasattr(result, "parameters")
        assert isinstance(result.parameters, np.ndarray)

    def test_result_has_error(self, mol_data):
        """Result should have error if exact energy available."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=10)
        assert result.error is not None
        assert result.error >= 0

    def test_works_with_all_ansatze(self, mol_data):
        """Should work with all ansatz types."""
        for ansatz_type in ["uccsd", "hardware_efficient", "noise_aware"]:
            result = run_vqe(mol_data, ansatz_type=ansatz_type, maxiter=10)
            assert isinstance(result, VQEResult)
            assert np.isfinite(result.energy)


class TestVariationalPrinciple:
    """Tests for variational principle."""

    def test_energy_above_exact(self, mol_data):
        """VQE energy should be >= exact energy."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=50)

        # VQE energy must be >= FCI energy (variational principle)
        assert result.energy >= mol_data.fci_energy - 1e-8

    def test_verify_variational_principle_passes(self, mol_data):
        """verify_variational_principle should return True."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=50)
        assert verify_variational_principle(result)

    def test_all_ansatze_satisfy_variational(self, mol_data):
        """All ansatze should satisfy variational principle."""
        for ansatz_type in ["uccsd", "noise_aware"]:
            result = run_vqe(mol_data, ansatz_type=ansatz_type, maxiter=50)
            assert verify_variational_principle(result), (
                f"{ansatz_type} violated variational principle"
            )


class TestEnergyQuality:
    """Tests for energy accuracy."""

    def test_uccsd_reasonably_accurate(self, mol_data):
        """UCCSD should give reasonable energy."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=100)

        # Should be within 0.05 Ha of exact
        assert result.error < 0.05

    def test_energy_is_real(self, mol_data):
        """Energy should be real."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=10)
        assert np.isreal(result.energy)

    def test_energy_is_finite(self, mol_data):
        """Energy should be finite."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=10)
        assert np.isfinite(result.energy)


class TestConvergence:
    """Tests for optimization convergence."""

    def test_tracks_evaluations(self, mol_data):
        """Should track number of evaluations."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=50)
        assert result.n_evaluations > 0

    def test_tracks_iterations(self, mol_data):
        """Should track number of iterations."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=50)
        assert result.n_iterations > 0

    def test_records_energy_history(self, mol_data):
        """Should record energy history."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=50)
        assert len(result.energy_history) > 0


class TestMultistart:
    """Tests for multistart VQE."""

    def test_multistart_returns_result(self, mol_data):
        """multistart should return VQEResult."""
        result = run_vqe_multistart(mol_data, n_starts=2, maxiter=10)
        assert isinstance(result, VQEResult)

    def test_multistart_improves_result(self, mol_data):
        """Multistart may find better minimum."""
        # Single start with zeros
        single = run_vqe(
            mol_data,
            ansatz_type="noise_aware",
            initial_params=np.zeros(4),
            maxiter=50,
        )

        # Multistart
        multi = run_vqe_multistart(
            mol_data,
            ansatz_type="noise_aware",
            n_starts=3,
            maxiter=50,
        )

        # Multistart should be at least as good
        assert multi.energy <= single.energy + 1e-6


class TestVQEResult:
    """Tests for VQEResult dataclass."""

    def test_result_repr(self, mol_data):
        """Result should have readable repr."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=10)
        repr_str = repr(result)

        assert "energy" in repr_str
        assert "converged" in repr_str

    def test_result_stores_ansatz_type(self, mol_data):
        """Result should store ansatz type."""
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=10)
        assert "UCCSD" in result.ansatz_type


class TestDifferentGeometries:
    """Tests at different molecular geometries."""

    def test_works_at_stretched_geometry(self):
        """VQE should work at stretched geometry."""
        mol_data = compute_h2_integrals(1.5)
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=50)

        assert isinstance(result, VQEResult)
        assert verify_variational_principle(result)

    def test_works_at_compressed_geometry(self):
        """VQE should work at compressed geometry."""
        mol_data = compute_h2_integrals(0.5)
        result = run_vqe(mol_data, ansatz_type="uccsd", maxiter=50)

        assert isinstance(result, VQEResult)
        assert verify_variational_principle(result)


class TestOptimizers:
    """Tests for different optimizers."""

    @pytest.mark.parametrize("optimizer", ["COBYLA", "SLSQP"])
    def test_different_optimizers_work(self, mol_data, optimizer):
        """Different optimizers should work."""
        result = run_vqe(
            mol_data,
            ansatz_type="noise_aware",
            optimizer=optimizer,
            maxiter=50,
        )

        assert isinstance(result, VQEResult)
        assert np.isfinite(result.energy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
