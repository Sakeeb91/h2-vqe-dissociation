"""
Tests for hamiltonian.py - Jordan-Wigner transformation.

These tests verify:
1. Correct qubit count and Hamiltonian structure
2. Exact diagonalization matches FCI energy
3. Hamiltonian properties (Hermiticity, etc.)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from h2_vqe.molecular import compute_h2_integrals
from h2_vqe.hamiltonian import (
    build_qubit_hamiltonian,
    build_fermionic_hamiltonian,
    exact_ground_state_energy,
    verify_hamiltonian,
    get_hamiltonian_info,
    QubitHamiltonian,
)


@pytest.fixture
def mol_data_equilibrium():
    """Molecular data at equilibrium geometry."""
    return compute_h2_integrals(0.74)


@pytest.fixture
def mol_data_stretched():
    """Molecular data at stretched geometry."""
    return compute_h2_integrals(1.5)


class TestBuildQubitHamiltonian:
    """Tests for qubit Hamiltonian construction."""

    def test_returns_qubit_hamiltonian(self, mol_data_equilibrium):
        """Should return QubitHamiltonian dataclass."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        assert isinstance(ham, QubitHamiltonian)

    def test_correct_qubit_count(self, mol_data_equilibrium):
        """STO-3G H2 should have 4 qubits."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        assert ham.n_qubits == 4

    def test_positive_term_count(self, mol_data_equilibrium):
        """Hamiltonian should have positive number of terms."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        assert ham.n_terms > 0

    def test_stores_fci_energy(self, mol_data_equilibrium):
        """Should store FCI energy for reference."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        assert ham.fci_energy == mol_data_equilibrium.fci_energy

    def test_stores_nuclear_repulsion(self, mol_data_equilibrium):
        """Should store nuclear repulsion energy."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        assert ham.nuclear_repulsion == mol_data_equilibrium.nuclear_repulsion


class TestExactDiagonalization:
    """Tests for exact ground state computation."""

    def test_matches_fci_equilibrium(self, mol_data_equilibrium):
        """Exact diagonalization should match FCI at equilibrium."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        exact_energy, _ = exact_ground_state_energy(ham)
        assert_allclose(exact_energy, mol_data_equilibrium.fci_energy, rtol=1e-8)

    def test_matches_fci_stretched(self, mol_data_stretched):
        """Exact diagonalization should match FCI at stretched geometry."""
        ham = build_qubit_hamiltonian(mol_data_stretched)
        exact_energy, _ = exact_ground_state_energy(ham)
        assert_allclose(exact_energy, mol_data_stretched.fci_energy, rtol=1e-8)

    def test_returns_normalized_state(self, mol_data_equilibrium):
        """Ground state should be normalized."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        _, state = exact_ground_state_energy(ham)
        norm = np.linalg.norm(state)
        assert_allclose(norm, 1.0, rtol=1e-10)

    def test_ground_state_is_eigenvector(self, mol_data_equilibrium):
        """Ground state should be an eigenvector of H."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        exact_energy, state = exact_ground_state_energy(ham)

        # H|ψ⟩ = E|ψ⟩
        matrix = ham.operator.to_matrix()
        h_psi = matrix @ state
        e_psi = exact_energy * state

        assert_allclose(h_psi, e_psi, rtol=1e-10)


class TestHamiltonianProperties:
    """Tests for Hamiltonian physical properties."""

    def test_hamiltonian_is_hermitian(self, mol_data_equilibrium):
        """Hamiltonian must be Hermitian."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        info = get_hamiltonian_info(ham)
        assert info["is_hermitian"]

    def test_real_eigenvalues(self, mol_data_equilibrium):
        """Hermitian matrix should have real eigenvalues."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        matrix = ham.operator.to_matrix()
        eigenvalues = np.linalg.eigvalsh(matrix)

        # All eigenvalues should be real
        assert np.allclose(eigenvalues.imag, 0)

    def test_ground_state_is_minimum(self, mol_data_equilibrium):
        """Ground state energy should be the minimum eigenvalue."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        matrix = ham.operator.to_matrix()
        eigenvalues = np.linalg.eigvalsh(matrix)

        ground_energy, _ = exact_ground_state_energy(ham)
        assert_allclose(ground_energy, np.min(eigenvalues))


class TestVerifyHamiltonian:
    """Tests for Hamiltonian verification."""

    def test_verify_passes_at_equilibrium(self, mol_data_equilibrium):
        """Verification should pass at equilibrium."""
        assert verify_hamiltonian(mol_data_equilibrium)

    def test_verify_passes_at_multiple_geometries(self):
        """Verification should pass at multiple bond lengths."""
        for r in [0.5, 0.74, 1.0, 1.5, 2.0]:
            mol_data = compute_h2_integrals(r)
            assert verify_hamiltonian(mol_data), f"Failed at r={r} Å"


class TestHamiltonianInfo:
    """Tests for Hamiltonian metadata."""

    def test_info_contains_expected_keys(self, mol_data_equilibrium):
        """Info dict should have expected keys."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        info = get_hamiltonian_info(ham)

        expected_keys = ["n_qubits", "n_terms", "max_pauli_weight",
                        "pauli_counts", "is_hermitian"]
        for key in expected_keys:
            assert key in info

    def test_max_pauli_weight_reasonable(self, mol_data_equilibrium):
        """Max Pauli weight should be <= n_qubits."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        info = get_hamiltonian_info(ham)
        assert info["max_pauli_weight"] <= ham.n_qubits

    def test_pauli_counts_positive(self, mol_data_equilibrium):
        """Pauli counts should be positive."""
        ham = build_qubit_hamiltonian(mol_data_equilibrium)
        info = get_hamiltonian_info(ham)

        for pauli, count in info["pauli_counts"].items():
            assert count >= 0


class TestFermionicHamiltonian:
    """Tests for fermionic Hamiltonian construction."""

    def test_fermionic_op_created(self, mol_data_equilibrium):
        """Should create fermionic operator."""
        from qiskit_nature.second_q.operators import FermionicOp

        fermionic_op = build_fermionic_hamiltonian(mol_data_equilibrium)
        assert isinstance(fermionic_op, FermionicOp)

    def test_fermionic_op_spin_orbitals(self, mol_data_equilibrium):
        """Fermionic op should have correct number of spin orbitals."""
        fermionic_op = build_fermionic_hamiltonian(mol_data_equilibrium)
        # num_spin_orbitals is the register length
        assert fermionic_op.num_spin_orbitals == mol_data_equilibrium.n_qubits


class TestPhysicalConsistency:
    """Tests for physical consistency across geometries."""

    def test_energy_ordering_preserved(self):
        """Energy ordering should be preserved through transformation."""
        # At equilibrium, energy should be lowest
        mol_short = compute_h2_integrals(0.5)
        mol_eq = compute_h2_integrals(0.74)
        mol_long = compute_h2_integrals(1.5)

        ham_short = build_qubit_hamiltonian(mol_short)
        ham_eq = build_qubit_hamiltonian(mol_eq)
        ham_long = build_qubit_hamiltonian(mol_long)

        e_short, _ = exact_ground_state_energy(ham_short)
        e_eq, _ = exact_ground_state_energy(ham_eq)
        e_long, _ = exact_ground_state_energy(ham_long)

        # Equilibrium should have lowest energy
        assert e_eq < e_short
        assert e_eq < e_long


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
