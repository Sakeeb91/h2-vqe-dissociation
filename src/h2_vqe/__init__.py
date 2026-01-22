"""
H₂ VQE Dissociation Curve Package
=================================

Compute ground state energy of molecular hydrogen (H₂) across varying bond lengths
using the Variational Quantum Eigensolver (VQE), with benchmarking against classical
methods (Hartree-Fock, FCI) and noise analysis.

Modules:
    molecular: PySCF interface for molecular integrals
    hamiltonian: Jordan-Wigner transformation
    ansatz: Variational circuit ansatze (UCCSD, hardware-efficient, noise-aware)
    vqe: VQE optimization engine with noise model support
    noise: IBM-like noise models (ideal, low_noise, ibm_like, high_noise)
    dissociation: Dissociation curve computation
    visualization: Plotting utilities including ZNE comparison plots
    ibm_runtime: IBM Quantum hardware integration with ZNE support (optional)

Features:
    - Noise simulation with configurable IBM-like noise models
    - IBM Quantum hardware integration with error mitigation
    - ZNE (Zero-Noise Extrapolation) benchmarking support
    - Publication-quality visualization

Example:
    >>> from h2_vqe import compute_h2_integrals, run_vqe
    >>> mol_data = compute_h2_integrals(0.74)
    >>> result = run_vqe(mol_data)
    >>> print(f"VQE Energy: {result.energy:.6f} Ha")

Example with noise:
    >>> from h2_vqe import compute_h2_integrals, run_vqe, create_noise_model
    >>> mol_data = compute_h2_integrals(0.74)
    >>> noise = create_noise_model("ibm_like")
    >>> result = run_vqe(mol_data, noise_model=noise)

For hardware experiments:
    >>> from h2_vqe.ibm_runtime import run_vqe_on_hardware, run_vqe_resilience_sweep
    >>> result = run_vqe_on_hardware(mol_data, ansatz_type="noise_aware")
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
