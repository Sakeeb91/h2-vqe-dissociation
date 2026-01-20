"""
H₂ VQE Dissociation Curve Package
=================================

Compute ground state energy of molecular hydrogen (H₂) across varying bond lengths
using the Variational Quantum Eigensolver (VQE), with benchmarking against classical
methods (Hartree-Fock, FCI) and noise analysis.

Modules:
    molecular: PySCF interface for molecular integrals
    hamiltonian: Jordan-Wigner transformation
    ansatz: Variational circuit ansatze
    vqe: VQE optimization engine
    noise: IBM-like noise models
    dissociation: Dissociation curve computation
    visualization: Plotting utilities

Example:
    >>> from h2_vqe import compute_h2_integrals, run_vqe
    >>> mol_data = compute_h2_integrals(0.74)
    >>> result = run_vqe(mol_data)
    >>> print(f"VQE Energy: {result.energy:.6f} Ha")
"""

__version__ = "0.1.0"

# Import main functions for convenience
from h2_vqe.molecular import compute_h2_integrals, MolecularData
from h2_vqe.hamiltonian import build_qubit_hamiltonian, QubitHamiltonian
from h2_vqe.ansatz import create_ansatz
from h2_vqe.vqe import run_vqe, VQEResult
from h2_vqe.noise import create_noise_model
from h2_vqe.dissociation import compute_dissociation_curve, DissociationCurveResult

__all__ = [
    "__version__",
    # Molecular
    "compute_h2_integrals",
    "MolecularData",
    # Hamiltonian
    "build_qubit_hamiltonian",
    "QubitHamiltonian",
    # Ansatz
    "create_ansatz",
    # VQE
    "run_vqe",
    "VQEResult",
    # Noise
    "create_noise_model",
    # Dissociation
    "compute_dissociation_curve",
    "DissociationCurveResult",
]
