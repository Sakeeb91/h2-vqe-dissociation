#!/usr/bin/env python3
"""
ZNE Benchmarking Study
======================

Compare VQE performance across conditions to answer:
"How effective is Zero-Noise Extrapolation (ZNE) for VQE on real IBM Quantum hardware?"

Conditions compared:
1. FCI (exact) - Classical ground truth
2. Simulator (noiseless) - VQE accuracy baseline
3. Simulator (IBM-like noise) - Predicted hardware performance
4. Hardware (raw, resilience_level=0) - Actual NISQ performance
5. Hardware (ZNE, resilience_level=2) - Error-mitigated performance

Usage:
    # Dry run (test without hardware)
    python scripts/run_zne_benchmark.py --dry-run

    # Run simulator-only benchmarks (no hardware queue)
    python scripts/run_zne_benchmark.py --simulator-only

    # Full benchmark with specific bond lengths
    python scripts/run_zne_benchmark.py --bond-lengths 0.5 0.74 1.0 1.5 2.0

    # Resume from existing results (add hardware data)
    python scripts/run_zne_benchmark.py --resume results/zne_benchmark.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Default bond lengths capturing key physics
# 0.5 Å - compressed, 0.74 Å - equilibrium, 1.0-2.0 Å - stretched/dissociating
DEFAULT_BOND_LENGTHS = [0.5, 0.74, 1.0, 1.5, 2.0]


def compute_simulator_data(
    bond_lengths: List[float],
    ansatz_type: str = "noise_aware",
    maxiter: int = 100,
    shots: int = 4096,
    verbose: bool = True,
) -> dict:
    """
    Compute FCI, noiseless VQE, and noisy VQE energies.

    Returns:
        Dict with fci, sim_noiseless, sim_noisy arrays
    """
    from h2_vqe.molecular import compute_h2_integrals
    from h2_vqe.vqe import run_vqe
    from h2_vqe.noise import create_noise_model

    n_points = len(bond_lengths)
    fci_energies = np.zeros(n_points)
    sim_noiseless = np.zeros(n_points)
    sim_noisy = np.zeros(n_points)

    # Create IBM-like noise model
    noise_model = create_noise_model("ibm_like")

    if verbose:
        print("\n" + "=" * 60)
        print("Computing Simulator Benchmarks")
        print("=" * 60)

    for i, r in enumerate(bond_lengths):
        if verbose:
            print(f"\n[{i+1}/{n_points}] Bond length = {r:.2f} Å")

        # Compute molecular data
        mol_data = compute_h2_integrals(r)
        fci_energies[i] = mol_data.fci_energy

        if verbose:
            print(f"  FCI energy: {fci_energies[i]:.6f} Ha")

        # Noiseless VQE
        result_noiseless = run_vqe(
            mol_data,
            ansatz_type=ansatz_type,
            backend="statevector",
            maxiter=maxiter,
        )
        sim_noiseless[i] = result_noiseless.energy
        if verbose:
            print(f"  Noiseless VQE: {sim_noiseless[i]:.6f} Ha "
                  f"(error: {result_noiseless.error * 1000:.2f} mHa)")

        # Noisy VQE (IBM-like)
        result_noisy = run_vqe(
            mol_data,
            ansatz_type=ansatz_type,
            backend="aer_simulator",
            noise_model=noise_model,
            shots=shots,
            maxiter=maxiter,
        )
        sim_noisy[i] = result_noisy.energy
        if verbose:
            print(f"  Noisy VQE: {sim_noisy[i]:.6f} Ha "
                  f"(error: {result_noisy.error * 1000:.2f} mHa)")

    return {
        "fci": fci_energies.tolist(),
        "sim_noiseless": sim_noiseless.tolist(),
        "sim_noisy": sim_noisy.tolist(),
    }


def compute_hardware_data(
    bond_lengths: List[float],
    ansatz_type: str = "noise_aware",
    backend_name: Optional[str] = None,
    maxiter: int = 30,
    shots: int = 4096,
    verbose: bool = True,
) -> dict:
    """
    Run VQE on IBM hardware at resilience levels 0 (raw) and 2 (ZNE).

    Returns:
        Dict with hw_raw, hw_zne arrays and metadata
    """
    from h2_vqe.molecular import compute_h2_integrals
    from h2_vqe.ibm_runtime import (
        get_service,
        get_least_busy_backend,
        run_vqe_on_hardware,
    )

    # Get service and backend
    service = get_service()
    if backend_name is None:
        backend_name = get_least_busy_backend(service)

    n_points = len(bond_lengths)
    hw_raw = np.zeros(n_points)
    hw_zne = np.zeros(n_points)
    job_ids_raw = []
    job_ids_zne = []

    if verbose:
        print("\n" + "=" * 60)
        print(f"Computing Hardware Benchmarks on {backend_name}")
        print("=" * 60)

    for i, r in enumerate(bond_lengths):
        if verbose:
            print(f"\n[{i+1}/{n_points}] Bond length = {r:.2f} Å")

        mol_data = compute_h2_integrals(r)

        # Raw (no mitigation)
        if verbose:
            print("  Running resilience_level=0 (raw)...")
        result_raw = run_vqe_on_hardware(
            mol_data,
            ansatz_type=ansatz_type,
            backend_name=backend_name,
            service=service,
            maxiter=maxiter,
            shots=shots,
            resilience_level=0,
        )
        hw_raw[i] = result_raw.energy
        job_ids_raw.extend(result_raw.job_ids or [])
        if verbose:
            error_str = f" (error: {result_raw.error * 1000:.2f} mHa)" if result_raw.error else ""
            print(f"  Raw: {hw_raw[i]:.6f} Ha{error_str}")

        # ZNE (zero-noise extrapolation)
        if verbose:
            print("  Running resilience_level=2 (ZNE)...")
        result_zne = run_vqe_on_hardware(
            mol_data,
            ansatz_type=ansatz_type,
            backend_name=backend_name,
            service=service,
            maxiter=maxiter,
            shots=shots,
            resilience_level=2,
        )
        hw_zne[i] = result_zne.energy
        job_ids_zne.extend(result_zne.job_ids or [])
        if verbose:
            error_str = f" (error: {result_zne.error * 1000:.2f} mHa)" if result_zne.error else ""
            print(f"  ZNE: {hw_zne[i]:.6f} Ha{error_str}")

    return {
        "hw_raw": hw_raw.tolist(),
        "hw_zne": hw_zne.tolist(),
        "backend": backend_name,
        "job_ids_raw": job_ids_raw,
        "job_ids_zne": job_ids_zne,
    }


def save_results(results: dict, output_path: str):
    """Save benchmark results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def load_results(input_path: str) -> dict:
    """Load existing results from JSON."""
    with open(input_path, "r") as f:
        return json.load(f)


def print_summary(results: dict):
    """Print summary statistics from benchmark results."""
    bond_lengths = np.array(results["bond_lengths"])
    fci = np.array(results["fci"])

    print("\n" + "=" * 70)
    print("ZNE Benchmark Summary")
    print("=" * 70)

    print(f"\nBond lengths: {bond_lengths.tolist()} Å")

    # Compute MAE for each condition
    print("\nMean Absolute Error (MAE) vs FCI:")
    print("-" * 50)

    if "sim_noiseless" in results:
        sim_noiseless = np.array(results["sim_noiseless"])
        mae = np.mean(np.abs(sim_noiseless - fci)) * 1000
        print(f"  Simulator (noiseless):   {mae:6.2f} mHa")

    if "sim_noisy" in results:
        sim_noisy = np.array(results["sim_noisy"])
        mae = np.mean(np.abs(sim_noisy - fci)) * 1000
        print(f"  Simulator (IBM noise):   {mae:6.2f} mHa")

    if "hw_raw" in results:
        hw_raw = np.array(results["hw_raw"])
        mae_raw = np.mean(np.abs(hw_raw - fci)) * 1000
        print(f"  Hardware (raw):          {mae_raw:6.2f} mHa")

    if "hw_zne" in results:
        hw_zne = np.array(results["hw_zne"])
        mae_zne = np.mean(np.abs(hw_zne - fci)) * 1000
        print(f"  Hardware (ZNE):          {mae_zne:6.2f} mHa")

        # ZNE improvement
        if "hw_raw" in results:
            improvement = mae_raw / mae_zne if mae_zne > 0 else float("inf")
            print(f"\nZNE Improvement Factor: {improvement:.2f}x")
            if improvement > 1:
                print(f"  -> ZNE reduced error by {(1 - 1/improvement) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="ZNE Benchmarking Study for H2 VQE"
    )
    parser.add_argument(
        "--bond-lengths",
        type=float,
        nargs="+",
        default=DEFAULT_BOND_LENGTHS,
        help="Bond lengths to test (Angstroms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/zne_benchmark.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="IBM backend name (default: least busy)",
    )
    parser.add_argument(
        "--ansatz",
        type=str,
        default="noise_aware",
        help="Ansatz type (default: noise_aware)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=50,
        help="Max optimizer iterations",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of measurement shots",
    )
    parser.add_argument(
        "--simulator-only",
        action="store_true",
        help="Only run simulator benchmarks (skip hardware)",
    )
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="Only run hardware benchmarks (requires --resume)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing results file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plot after benchmarking",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ZNE Benchmarking Study")
    print("=" * 60)
    print(f"Research Question: How effective is ZNE for VQE on real hardware?")
    print("-" * 60)

    # Dry run mode
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Bond lengths: {args.bond_lengths} Å")
        print(f"Ansatz: {args.ansatz}")
        print(f"Max iterations: {args.maxiter}")
        print(f"Shots: {args.shots}")
        print(f"Output: {args.output}")
        print(f"Simulator only: {args.simulator_only}")
        print(f"Hardware only: {args.hardware_only}")
        print("\nWould run the following:")
        if not args.hardware_only:
            print("  1. FCI calculations (classical)")
            print("  2. Noiseless VQE (statevector simulator)")
            print("  3. Noisy VQE (Aer with IBM-like noise)")
        if not args.simulator_only:
            print("  4. Hardware VQE (resilience_level=0, raw)")
            print("  5. Hardware VQE (resilience_level=2, ZNE)")
        print("\nNo computations executed.")
        return

    # Initialize or load results
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        results = load_results(args.resume)
        # Update bond lengths from existing
        args.bond_lengths = results["bond_lengths"]
    else:
        results = {
            "timestamp": datetime.now().isoformat(),
            "bond_lengths": args.bond_lengths,
            "ansatz": args.ansatz,
            "maxiter": args.maxiter,
            "shots": args.shots,
        }

    # Run simulator benchmarks
    if not args.hardware_only:
        sim_data = compute_simulator_data(
            args.bond_lengths,
            ansatz_type=args.ansatz,
            maxiter=args.maxiter,
            shots=args.shots,
        )
        results.update(sim_data)
        save_results(results, args.output)

    # Run hardware benchmarks
    if not args.simulator_only:
        try:
            from h2_vqe.ibm_runtime import check_ibm_runtime
            if not check_ibm_runtime():
                print("\nSkipping hardware: qiskit-ibm-runtime not installed")
                print("Install with: pip install qiskit-ibm-runtime")
            else:
                hw_data = compute_hardware_data(
                    args.bond_lengths,
                    ansatz_type=args.ansatz,
                    backend_name=args.backend,
                    maxiter=min(args.maxiter, 30),  # Lower for hardware
                    shots=args.shots,
                )
                results.update(hw_data)
                save_results(results, args.output)
        except Exception as e:
            print(f"\nHardware error: {e}")
            print("Simulator results saved. Re-run with --hardware-only to add hardware data.")

    # Print summary
    print_summary(results)

    # Generate plot if requested
    if args.plot:
        try:
            from h2_vqe.visualization import plot_zne_comparison
            plot_path = args.output.replace(".json", ".png")
            plot_zne_comparison(
                bond_lengths=np.array(results["bond_lengths"]),
                fci_energies=np.array(results["fci"]),
                sim_noiseless=np.array(results.get("sim_noiseless", [])),
                sim_noisy=np.array(results.get("sim_noisy", [])),
                hw_raw=np.array(results.get("hw_raw", [])) if "hw_raw" in results else None,
                hw_zne=np.array(results.get("hw_zne", [])) if "hw_zne" in results else None,
                save_path=plot_path,
            )
        except Exception as e:
            print(f"\nCould not generate plot: {e}")
            print("Run analyze_zne_results.py for visualization.")


if __name__ == "__main__":
    main()
