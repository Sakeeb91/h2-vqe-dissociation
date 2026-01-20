"""
H₂ Dissociation Curve Computation
=================================

This module orchestrates the full dissociation curve calculation
by running VQE at multiple bond lengths and comparing with classical
reference methods (Hartree-Fock and Full CI).

A dissociation curve shows how the energy of H₂ changes as the
H-H bond is stretched from equilibrium (~0.74 Å) to dissociation.

Example:
    >>> from h2_vqe.dissociation import compute_dissociation_curve
    >>> results = compute_dissociation_curve(n_points=10)
    >>> print(f"Equilibrium energy: {min(results['fci_energies']):.4f} Ha")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from h2_vqe.molecular import (
    compute_h2_integrals,
    compute_bond_lengths,
    compute_classical_energies,
)
from h2_vqe.vqe import run_vqe, VQEResult


@dataclass
class DissociationCurveResult:
    """
    Container for full dissociation curve results.

    Attributes:
        bond_lengths: Array of bond distances (Å)
        hf_energies: Hartree-Fock energies (Ha)
        fci_energies: Full CI energies (Ha)
        vqe_energies: VQE energies (Ha) - dict keyed by ansatz type
        vqe_errors: Absolute errors |E_VQE - E_FCI| (Ha)
        ansatz_types: List of ansatz types computed
        n_points: Number of bond lengths
    """
    bond_lengths: np.ndarray
    hf_energies: np.ndarray
    fci_energies: np.ndarray
    vqe_energies: dict = field(default_factory=dict)
    vqe_errors: dict = field(default_factory=dict)
    ansatz_types: List[str] = field(default_factory=list)
    n_points: int = 0

    def __post_init__(self):
        self.n_points = len(self.bond_lengths)

    def equilibrium_distance(self) -> float:
        """Get equilibrium bond distance (minimum energy point)."""
        min_idx = np.argmin(self.fci_energies)
        return self.bond_lengths[min_idx]

    def equilibrium_energy(self) -> float:
        """Get equilibrium FCI energy."""
        return np.min(self.fci_energies)

    def dissociation_energy(self) -> float:
        """
        Compute dissociation energy: E(∞) - E(equilibrium).

        Returns energy in eV (1 Ha = 27.2114 eV).
        """
        e_eq = np.min(self.fci_energies)
        e_inf = self.fci_energies[-1]  # Assume last point is near dissociation
        return (e_inf - e_eq) * 27.2114  # Convert to eV

    def max_vqe_error(self, ansatz_type: str) -> float:
        """Get maximum VQE error for an ansatz."""
        if ansatz_type not in self.vqe_errors:
            raise ValueError(f"No data for ansatz type: {ansatz_type}")
        return np.max(self.vqe_errors[ansatz_type])

    def mean_vqe_error(self, ansatz_type: str) -> float:
        """Get mean VQE error for an ansatz."""
        if ansatz_type not in self.vqe_errors:
            raise ValueError(f"No data for ansatz type: {ansatz_type}")
        return np.mean(self.vqe_errors[ansatz_type])

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bond_lengths": self.bond_lengths.tolist(),
            "hf_energies": self.hf_energies.tolist(),
            "fci_energies": self.fci_energies.tolist(),
            "vqe_energies": {k: v.tolist() for k, v in self.vqe_energies.items()},
            "vqe_errors": {k: v.tolist() for k, v in self.vqe_errors.items()},
            "ansatz_types": self.ansatz_types,
            "n_points": self.n_points,
        }

    def save(self, filepath: str) -> None:
        """Save results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "DissociationCurveResult":
        """Load results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        return cls(
            bond_lengths=np.array(data["bond_lengths"]),
            hf_energies=np.array(data["hf_energies"]),
            fci_energies=np.array(data["fci_energies"]),
            vqe_energies={k: np.array(v) for k, v in data["vqe_energies"].items()},
            vqe_errors={k: np.array(v) for k, v in data["vqe_errors"].items()},
            ansatz_types=data["ansatz_types"],
        )


def compute_dissociation_curve(
    bond_lengths: Optional[np.ndarray] = None,
    n_points: int = 20,
    start: float = 0.3,
    stop: float = 2.5,
    ansatz_types: List[str] = None,
    vqe_maxiter: int = 100,
    verbose: bool = True,
) -> DissociationCurveResult:
    """
    Compute full H₂ dissociation curve with VQE.

    Args:
        bond_lengths: Explicit bond lengths to compute (overrides n_points/start/stop)
        n_points: Number of points (if bond_lengths not provided)
        start: Starting bond length in Å
        stop: Ending bond length in Å
        ansatz_types: List of ansatz types to compute (default: all three)
        vqe_maxiter: Maximum VQE iterations per point
        verbose: Print progress updates

    Returns:
        DissociationCurveResult with all computed data

    Example:
        >>> results = compute_dissociation_curve(n_points=10)
        >>> print(f"Min FCI energy: {results.equilibrium_energy():.6f} Ha")
    """
    # Set default ansatz types
    if ansatz_types is None:
        ansatz_types = ["uccsd", "noise_aware"]

    # Generate bond lengths
    if bond_lengths is None:
        bond_lengths = compute_bond_lengths(start, stop, n_points)
    else:
        bond_lengths = np.array(bond_lengths)

    n_points = len(bond_lengths)

    if verbose:
        print(f"Computing H₂ dissociation curve with {n_points} points")
        print(f"Bond lengths: {bond_lengths[0]:.2f} to {bond_lengths[-1]:.2f} Å")
        print(f"Ansatz types: {ansatz_types}")
        print()

    # Compute classical energies
    if verbose:
        print("Computing classical reference energies (HF, FCI)...")

    classical = compute_classical_energies(bond_lengths)
    hf_energies = classical["hf_energies"]
    fci_energies = classical["fci_energies"]

    if verbose:
        print(f"  HF equilibrium:  {np.min(hf_energies):.6f} Ha")
        print(f"  FCI equilibrium: {np.min(fci_energies):.6f} Ha")
        print()

    # Initialize VQE result storage
    vqe_energies = {ansatz: np.zeros(n_points) for ansatz in ansatz_types}
    vqe_errors = {ansatz: np.zeros(n_points) for ansatz in ansatz_types}

    # Compute VQE energies
    for ansatz_type in ansatz_types:
        if verbose:
            print(f"Running VQE with {ansatz_type} ansatz...")

        for i, r in enumerate(bond_lengths):
            mol_data = compute_h2_integrals(r)
            result = run_vqe(mol_data, ansatz_type=ansatz_type, maxiter=vqe_maxiter)

            vqe_energies[ansatz_type][i] = result.energy
            vqe_errors[ansatz_type][i] = abs(result.energy - fci_energies[i])

            if verbose and (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{n_points} points")

        if verbose:
            max_err = np.max(vqe_errors[ansatz_type]) * 1000  # to mHa
            mean_err = np.mean(vqe_errors[ansatz_type]) * 1000
            print(f"  Max error:  {max_err:.2f} mHa")
            print(f"  Mean error: {mean_err:.2f} mHa")
            print()

    return DissociationCurveResult(
        bond_lengths=bond_lengths,
        hf_energies=hf_energies,
        fci_energies=fci_energies,
        vqe_energies=vqe_energies,
        vqe_errors=vqe_errors,
        ansatz_types=ansatz_types,
    )


def compute_single_point(
    bond_length: float,
    ansatz_type: str = "uccsd",
    maxiter: int = 100,
) -> dict:
    """
    Compute VQE at a single bond length (for parallel execution).

    Args:
        bond_length: H-H distance in Å
        ansatz_type: Type of ansatz
        maxiter: Max VQE iterations

    Returns:
        Dict with bond_length, vqe_energy, fci_energy, error
    """
    mol_data = compute_h2_integrals(bond_length)
    result = run_vqe(mol_data, ansatz_type=ansatz_type, maxiter=maxiter)

    return {
        "bond_length": bond_length,
        "vqe_energy": result.energy,
        "fci_energy": mol_data.fci_energy,
        "hf_energy": mol_data.hf_energy,
        "error": abs(result.energy - mol_data.fci_energy),
    }


def quick_curve(n_points: int = 5) -> DissociationCurveResult:
    """
    Compute a quick dissociation curve for testing.

    Uses fewer points and only noise_aware ansatz for speed.

    Args:
        n_points: Number of bond lengths (default: 5)

    Returns:
        DissociationCurveResult
    """
    return compute_dissociation_curve(
        n_points=n_points,
        start=0.5,
        stop=2.0,
        ansatz_types=["noise_aware"],
        vqe_maxiter=50,
        verbose=False,
    )


def analyze_correlation_energy(results: DissociationCurveResult) -> dict:
    """
    Analyze correlation energy along dissociation curve.

    Correlation energy = E_FCI - E_HF is the energy not captured
    by the mean-field Hartree-Fock approximation.

    Args:
        results: Dissociation curve results

    Returns:
        Dictionary with correlation energy analysis
    """
    correlation = results.fci_energies - results.hf_energies

    # Find where correlation energy is most significant
    max_corr_idx = np.argmin(correlation)  # Most negative = strongest correlation

    return {
        "correlation_energies": correlation,
        "max_correlation": correlation[max_corr_idx],
        "max_correlation_distance": results.bond_lengths[max_corr_idx],
        "correlation_at_equilibrium": correlation[np.argmin(results.fci_energies)],
        "correlation_at_dissociation": correlation[-1],
    }


if __name__ == "__main__":
    import time

    print("H₂ Dissociation Curve Computation")
    print("=" * 50)

    start_time = time.time()

    # Compute curve with 10 points
    results = compute_dissociation_curve(
        n_points=10,
        ansatz_types=["uccsd", "noise_aware"],
        vqe_maxiter=100,
    )

    elapsed = time.time() - start_time

    print(f"\nComputation completed in {elapsed:.1f} seconds")
    print(f"\nSummary:")
    print(f"  Equilibrium distance: {results.equilibrium_distance():.3f} Å")
    print(f"  Equilibrium energy:   {results.equilibrium_energy():.6f} Ha")
    print(f"  Dissociation energy:  {results.dissociation_energy():.2f} eV")

    print(f"\nVQE Accuracy:")
    for ansatz in results.ansatz_types:
        max_err = results.max_vqe_error(ansatz) * 1000
        mean_err = results.mean_vqe_error(ansatz) * 1000
        print(f"  {ansatz}: max={max_err:.1f} mHa, mean={mean_err:.1f} mHa")

    # Save results
    output_file = Path(__file__).parent.parent.parent / "results" / "dissociation_curve.json"
    output_file.parent.mkdir(exist_ok=True)
    results.save(str(output_file))
    print(f"\nResults saved to: {output_file}")
