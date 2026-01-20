"""
PySCF Interface for H₂ Molecular Integrals
==========================================

This module provides functions to compute molecular integrals for H₂
using the PySCF quantum chemistry package, along with classical
reference energies (Hartree-Fock and Full Configuration Interaction).

The molecular integrals are essential for constructing the qubit
Hamiltonian used in VQE calculations.

Example:
    >>> from h2_vqe.molecular import compute_h2_integrals
    >>> data = compute_h2_integrals(0.74)  # equilibrium bond length
    >>> print(f"HF Energy:  {data.hf_energy:.6f} Ha")
    >>> print(f"FCI Energy: {data.fci_energy:.6f} Ha")
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyscf import gto, scf, fci, ao2mo


@dataclass
class MolecularData:
    """
    Container for molecular data computed from PySCF.

    This dataclass holds all the information needed to construct
    a qubit Hamiltonian and benchmark VQE results.

    Attributes:
        bond_length: H-H bond distance in Angstroms
        basis: Basis set name (e.g., "sto-3g")
        n_electrons: Number of electrons (2 for H₂)
        n_orbitals: Number of spatial orbitals
        n_qubits: Number of qubits needed (2 * n_orbitals for spin orbitals)
        nuclear_repulsion: Nuclear repulsion energy in Hartrees
        one_body_integrals: One-electron integrals h_pq (n_orbitals x n_orbitals)
        two_body_integrals: Two-electron integrals g_pqrs (chemist notation)
        hf_energy: Hartree-Fock energy in Hartrees
        fci_energy: Full CI energy (exact within basis) in Hartrees
        mo_coefficients: Molecular orbital coefficients from HF
        orbital_energies: HF orbital energies
    """
    bond_length: float
    basis: str
    n_electrons: int
    n_orbitals: int
    n_qubits: int
    nuclear_repulsion: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    hf_energy: float
    fci_energy: float
    mo_coefficients: np.ndarray
    orbital_energies: np.ndarray

    def correlation_energy(self) -> float:
        """Compute correlation energy: E_corr = E_FCI - E_HF."""
        return self.fci_energy - self.hf_energy

    def __repr__(self) -> str:
        return (
            f"MolecularData(H₂, r={self.bond_length:.3f} Å, {self.basis}, "
            f"E_HF={self.hf_energy:.6f}, E_FCI={self.fci_energy:.6f})"
        )


def compute_h2_integrals(
    bond_length: float,
    basis: str = "sto-3g",
    verbose: int = 0
) -> MolecularData:
    """
    Compute molecular integrals for H₂ at a given bond length.

    This function performs the following steps:
    1. Build H₂ molecule geometry
    2. Run Hartree-Fock (HF) calculation
    3. Transform integrals to molecular orbital (MO) basis
    4. Run Full Configuration Interaction (FCI) for exact energy

    Args:
        bond_length: H-H bond distance in Angstroms (typical range: 0.3-3.0)
        basis: Gaussian basis set name. Common choices:
            - "sto-3g": Minimal basis (2 orbitals, 4 qubits)
            - "6-31g": Split-valence (4 orbitals, 8 qubits)
            - "cc-pvdz": Correlation-consistent (10 orbitals, 20 qubits)
        verbose: PySCF verbosity level (0=silent, 4=debug)

    Returns:
        MolecularData object containing all molecular information

    Raises:
        ValueError: If bond_length is not positive
        RuntimeError: If HF or FCI calculation fails to converge

    Example:
        >>> data = compute_h2_integrals(0.74)
        >>> print(f"Number of qubits: {data.n_qubits}")
        4
        >>> print(f"FCI Energy: {data.fci_energy:.6f} Ha")
        -1.137270 Ha

    Notes:
        - For STO-3G basis, H₂ has 2 spatial orbitals → 4 spin orbitals → 4 qubits
        - The two-body integrals use chemist notation: (pq|rs)
        - FCI is exact within the chosen basis set
    """
    # Validate input
    if bond_length <= 0:
        raise ValueError(f"bond_length must be positive, got {bond_length}")

    # Build H₂ molecule
    # Place atoms along z-axis, centered at origin
    geometry = f"H 0 0 0; H 0 0 {bond_length}"

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0  # singlet state
    mol.unit = "angstrom"
    mol.verbose = verbose
    mol.build()

    n_electrons = mol.nelectron  # Should be 2 for H₂
    n_orbitals = mol.nao  # Number of atomic/molecular orbitals

    # Run Restricted Hartree-Fock (RHF)
    mf = scf.RHF(mol)
    mf.verbose = verbose
    hf_energy = mf.kernel()

    if not mf.converged:
        raise RuntimeError(
            f"HF calculation did not converge for H₂ at r={bond_length} Å"
        )

    # Get MO coefficients and orbital energies
    mo_coeff = mf.mo_coeff
    orbital_energies = mf.mo_energy

    # Compute one-electron integrals in MO basis
    # h_pq = <p|h|q> where h = kinetic + nuclear attraction
    h1_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    one_body_integrals = mo_coeff.T @ h1_ao @ mo_coeff

    # Compute two-electron integrals in MO basis
    # (pq|rs) in chemist notation
    # Using efficient ao2mo transformation
    two_body_integrals = ao2mo.kernel(mol, mo_coeff)

    # Restore to 4D array for easier manipulation
    # ao2mo returns compact form for efficiency
    two_body_integrals = ao2mo.restore(1, two_body_integrals, n_orbitals)

    # Get nuclear repulsion energy
    nuclear_repulsion = mol.energy_nuc()

    # Run Full CI for exact ground state energy
    cisolver = fci.FCI(mf)
    cisolver.verbose = verbose
    fci_energy, _ = cisolver.kernel()

    # Number of qubits = 2 * n_orbitals (for spin-up and spin-down)
    n_qubits = 2 * n_orbitals

    return MolecularData(
        bond_length=bond_length,
        basis=basis,
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_qubits=n_qubits,
        nuclear_repulsion=nuclear_repulsion,
        one_body_integrals=one_body_integrals,
        two_body_integrals=two_body_integrals,
        hf_energy=hf_energy,
        fci_energy=fci_energy,
        mo_coefficients=mo_coeff,
        orbital_energies=orbital_energies,
    )


def compute_bond_lengths(
    start: float = 0.3,
    stop: float = 2.5,
    num: int = 20
) -> np.ndarray:
    """
    Generate array of bond lengths for dissociation curve.

    Args:
        start: Minimum bond length in Angstroms
        stop: Maximum bond length in Angstroms
        num: Number of points

    Returns:
        Array of bond lengths evenly spaced from start to stop

    Example:
        >>> lengths = compute_bond_lengths(0.5, 2.0, 4)
        >>> print(lengths)
        [0.5 1.0 1.5 2.0]
    """
    return np.linspace(start, stop, num)


def compute_classical_energies(
    bond_lengths: np.ndarray,
    basis: str = "sto-3g",
    verbose: int = 0
) -> dict:
    """
    Compute classical reference energies across multiple bond lengths.

    This is useful for benchmarking VQE results against exact classical values.

    Args:
        bond_lengths: Array of bond distances in Angstroms
        basis: Basis set name
        verbose: PySCF verbosity level

    Returns:
        Dictionary with keys:
            - "bond_lengths": Input bond lengths
            - "hf_energies": HF energies at each bond length
            - "fci_energies": FCI energies at each bond length
            - "correlation_energies": E_FCI - E_HF at each point

    Example:
        >>> lengths = np.array([0.5, 0.74, 1.0])
        >>> results = compute_classical_energies(lengths)
        >>> print(results["fci_energies"])
    """
    hf_energies = []
    fci_energies = []

    for r in bond_lengths:
        data = compute_h2_integrals(r, basis=basis, verbose=verbose)
        hf_energies.append(data.hf_energy)
        fci_energies.append(data.fci_energy)

    hf_energies = np.array(hf_energies)
    fci_energies = np.array(fci_energies)

    return {
        "bond_lengths": bond_lengths,
        "hf_energies": hf_energies,
        "fci_energies": fci_energies,
        "correlation_energies": fci_energies - hf_energies,
    }


# Physical constants for reference
EQUILIBRIUM_BOND_LENGTH = 0.74  # Angstroms (experimental value)
EXPERIMENTAL_BINDING_ENERGY = 4.75  # eV (experimental dissociation energy)


if __name__ == "__main__":
    # Quick verification
    print("Computing H₂ molecular integrals at equilibrium bond length...")
    data = compute_h2_integrals(EQUILIBRIUM_BOND_LENGTH)

    print(f"\nMolecular Data:")
    print(f"  Bond length:        {data.bond_length:.3f} Å")
    print(f"  Basis set:          {data.basis}")
    print(f"  Number of electrons: {data.n_electrons}")
    print(f"  Number of orbitals:  {data.n_orbitals}")
    print(f"  Number of qubits:    {data.n_qubits}")
    print(f"\nEnergies (Hartrees):")
    print(f"  Nuclear repulsion:  {data.nuclear_repulsion:.6f}")
    print(f"  HF Energy:          {data.hf_energy:.6f}")
    print(f"  FCI Energy:         {data.fci_energy:.6f}")
    print(f"  Correlation energy: {data.correlation_energy():.6f}")
    print(f"\nIntegral shapes:")
    print(f"  One-body: {data.one_body_integrals.shape}")
    print(f"  Two-body: {data.two_body_integrals.shape}")
