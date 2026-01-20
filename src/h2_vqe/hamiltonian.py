"""
Jordan-Wigner Transformation for Qubit Hamiltonians
====================================================

This module transforms fermionic molecular Hamiltonians to qubit operators
using the Jordan-Wigner mapping, enabling quantum simulation of molecular
systems on qubit-based quantum computers.

The Jordan-Wigner transformation maps fermionic creation/annihilation
operators to Pauli strings:

    a†_j → (1/2)(X_j - iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
    a_j  → (1/2)(X_j + iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0

Example:
    >>> from h2_vqe.molecular import compute_h2_integrals
    >>> from h2_vqe.hamiltonian import build_qubit_hamiltonian
    >>> mol_data = compute_h2_integrals(0.74)
    >>> qubit_ham = build_qubit_hamiltonian(mol_data)
    >>> print(f"Number of Pauli terms: {len(qubit_ham)}")
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

from h2_vqe.molecular import MolecularData


@dataclass
class QubitHamiltonian:
    """
    Container for the qubit Hamiltonian and associated metadata.

    Attributes:
        operator: The qubit Hamiltonian as a SparsePauliOp
        n_qubits: Number of qubits required
        n_terms: Number of Pauli terms in the Hamiltonian
        fci_energy: Exact (FCI) energy for reference
        nuclear_repulsion: Nuclear repulsion energy included in operator
    """
    operator: SparsePauliOp
    n_qubits: int
    n_terms: int
    fci_energy: float
    nuclear_repulsion: float

    def __repr__(self) -> str:
        return (
            f"QubitHamiltonian(n_qubits={self.n_qubits}, "
            f"n_terms={self.n_terms}, E_exact={self.fci_energy:.6f})"
        )


def build_fermionic_hamiltonian(mol_data: MolecularData) -> FermionicOp:
    """
    Build the second-quantized fermionic Hamiltonian from molecular integrals.

    The electronic Hamiltonian in second quantization is:

        H = Σ_pq h_pq a†_p a_q + (1/2) Σ_pqrs g_pqrs a†_p a†_r a_s a_q + E_nuc

    where h_pq are one-electron integrals and g_pqrs are two-electron integrals.

    Args:
        mol_data: MolecularData containing molecular integrals

    Returns:
        FermionicOp representing the electronic Hamiltonian

    Notes:
        - Spin orbitals are ordered as: 0α, 0β, 1α, 1β, ...
        - Nuclear repulsion is added as a constant term
    """
    n_spatial = mol_data.n_orbitals
    h1 = mol_data.one_body_integrals
    h2 = mol_data.two_body_integrals

    # Build fermionic operator dictionary
    # Keys are tuples like ((p, '+'), (q, '-')) for a†_p a_q
    fermionic_dict = {}

    # One-body terms: h_pq a†_p a_q
    # Need to map spatial orbitals to spin orbitals
    # Spin orbital 2*p = spatial p, spin alpha
    # Spin orbital 2*p+1 = spatial p, spin beta
    for p in range(n_spatial):
        for q in range(n_spatial):
            if abs(h1[p, q]) > 1e-12:
                # Alpha spin
                key_alpha = f"+_{2*p} -_{2*q}"
                fermionic_dict[key_alpha] = fermionic_dict.get(key_alpha, 0) + h1[p, q]
                # Beta spin
                key_beta = f"+_{2*p+1} -_{2*q+1}"
                fermionic_dict[key_beta] = fermionic_dict.get(key_beta, 0) + h1[p, q]

    # Two-body terms: (1/2) g_pqrs a†_p a†_r a_s a_q
    # In physicist notation: <pq|rs> = (pr|qs) in chemist notation
    # PySCF gives us (pq|rs) in chemist notation
    # We need to convert: H = (1/2) Σ (pq|rs) a†_p a†_r a_s a_q
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    # Chemist notation (pq|rs)
                    coeff = 0.5 * h2[p, q, r, s]
                    if abs(coeff) < 1e-12:
                        continue

                    # Generate all spin combinations
                    # αα: p_α, r_α, s_α, q_α
                    if True:
                        key = f"+_{2*p} +_{2*r} -_{2*s} -_{2*q}"
                        fermionic_dict[key] = fermionic_dict.get(key, 0) + coeff

                    # ββ: p_β, r_β, s_β, q_β
                    if True:
                        key = f"+_{2*p+1} +_{2*r+1} -_{2*s+1} -_{2*q+1}"
                        fermionic_dict[key] = fermionic_dict.get(key, 0) + coeff

                    # αβ: p_α, r_β, s_β, q_α
                    if True:
                        key = f"+_{2*p} +_{2*r+1} -_{2*s+1} -_{2*q}"
                        fermionic_dict[key] = fermionic_dict.get(key, 0) + coeff

                    # βα: p_β, r_α, s_α, q_β
                    if True:
                        key = f"+_{2*p+1} +_{2*r} -_{2*s} -_{2*q+1}"
                        fermionic_dict[key] = fermionic_dict.get(key, 0) + coeff

    # Create FermionicOp
    # Filter out very small terms
    fermionic_dict = {k: v for k, v in fermionic_dict.items() if abs(v) > 1e-12}

    fermionic_op = FermionicOp(fermionic_dict, num_spin_orbitals=mol_data.n_qubits)

    return fermionic_op


def build_qubit_hamiltonian(mol_data: MolecularData) -> QubitHamiltonian:
    """
    Build the qubit Hamiltonian using Jordan-Wigner transformation.

    This is the main function for converting molecular data to a qubit
    operator suitable for VQE simulation.

    Args:
        mol_data: MolecularData containing molecular integrals

    Returns:
        QubitHamiltonian containing the Pauli operator and metadata

    Example:
        >>> data = compute_h2_integrals(0.74)
        >>> ham = build_qubit_hamiltonian(data)
        >>> print(f"Qubits: {ham.n_qubits}")
        4
        >>> print(f"Exact energy: {ham.fci_energy:.6f} Ha")
        -1.137284 Ha
    """
    # Build fermionic Hamiltonian
    fermionic_op = build_fermionic_hamiltonian(mol_data)

    # Apply Jordan-Wigner transformation
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    # Add nuclear repulsion as identity term
    n_qubits = mol_data.n_qubits
    identity_string = "I" * n_qubits
    nuclear_op = SparsePauliOp([identity_string], [mol_data.nuclear_repulsion])
    qubit_op = qubit_op + nuclear_op

    # Simplify the operator
    qubit_op = qubit_op.simplify()

    return QubitHamiltonian(
        operator=qubit_op,
        n_qubits=n_qubits,
        n_terms=len(qubit_op),
        fci_energy=mol_data.fci_energy,
        nuclear_repulsion=mol_data.nuclear_repulsion,
    )


def exact_ground_state_energy(qubit_ham: QubitHamiltonian) -> Tuple[float, np.ndarray]:
    """
    Compute exact ground state energy via full diagonalization.

    This is useful for verifying that the qubit Hamiltonian is correctly
    constructed by comparing to the FCI energy from PySCF.

    Args:
        qubit_ham: QubitHamiltonian to diagonalize

    Returns:
        Tuple of (ground_state_energy, ground_state_vector)

    Raises:
        MemoryError: If the Hamiltonian is too large (>16 qubits)

    Example:
        >>> ham = build_qubit_hamiltonian(mol_data)
        >>> energy, state = exact_ground_state_energy(ham)
        >>> print(f"Exact energy: {energy:.8f} Ha")
    """
    if qubit_ham.n_qubits > 16:
        raise MemoryError(
            f"Exact diagonalization requires 2^{qubit_ham.n_qubits} = "
            f"{2**qubit_ham.n_qubits} basis states. Use n_qubits <= 16."
        )

    # Convert to dense matrix
    matrix = qubit_ham.operator.to_matrix()

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Return ground state (lowest eigenvalue)
    return eigenvalues[0], eigenvectors[:, 0]


def verify_hamiltonian(mol_data: MolecularData, rtol: float = 1e-6) -> bool:
    """
    Verify that qubit Hamiltonian gives correct ground state energy.

    This function compares the exact diagonalization of the qubit Hamiltonian
    with the FCI energy from PySCF to ensure the transformation is correct.

    Args:
        mol_data: MolecularData to verify
        rtol: Relative tolerance for comparison

    Returns:
        True if energies match within tolerance

    Example:
        >>> data = compute_h2_integrals(0.74)
        >>> is_correct = verify_hamiltonian(data)
        >>> print(f"Hamiltonian correct: {is_correct}")
        True
    """
    qubit_ham = build_qubit_hamiltonian(mol_data)
    exact_energy, _ = exact_ground_state_energy(qubit_ham)

    return np.isclose(exact_energy, mol_data.fci_energy, rtol=rtol)


def get_hamiltonian_info(qubit_ham: QubitHamiltonian) -> dict:
    """
    Get detailed information about the Hamiltonian structure.

    Args:
        qubit_ham: QubitHamiltonian to analyze

    Returns:
        Dictionary with Hamiltonian statistics
    """
    op = qubit_ham.operator

    # Count Pauli types
    pauli_counts = {"I": 0, "X": 0, "Y": 0, "Z": 0}
    max_weight = 0

    for pauli_str in op.paulis.to_labels():
        weight = sum(1 for c in pauli_str if c != "I")
        max_weight = max(max_weight, weight)
        for c in pauli_str:
            pauli_counts[c] = pauli_counts.get(c, 0) + 1

    return {
        "n_qubits": qubit_ham.n_qubits,
        "n_terms": qubit_ham.n_terms,
        "max_pauli_weight": max_weight,
        "pauli_counts": pauli_counts,
        "is_hermitian": _is_hermitian(op),
    }


def _is_hermitian(op: SparsePauliOp, tol: float = 1e-10) -> bool:
    """Check if operator is Hermitian."""
    # SparsePauliOp with real coefficients and Pauli strings is Hermitian
    # if all coefficients are real (Pauli matrices are Hermitian)
    return np.allclose(op.coeffs.imag, 0, atol=tol)


if __name__ == "__main__":
    from h2_vqe.molecular import compute_h2_integrals

    print("Building qubit Hamiltonian for H₂ at 0.74 Å...")
    mol_data = compute_h2_integrals(0.74)
    qubit_ham = build_qubit_hamiltonian(mol_data)

    print(f"\nHamiltonian info:")
    info = get_hamiltonian_info(qubit_ham)
    for key, val in info.items():
        print(f"  {key}: {val}")

    print(f"\nEnergies:")
    print(f"  FCI (PySCF):         {mol_data.fci_energy:.8f} Ha")
    exact_e, _ = exact_ground_state_energy(qubit_ham)
    print(f"  Exact (qubit diag):  {exact_e:.8f} Ha")
    print(f"  Difference:          {abs(exact_e - mol_data.fci_energy):.2e} Ha")

    print(f"\nVerification: {'PASSED' if verify_hamiltonian(mol_data) else 'FAILED'}")
