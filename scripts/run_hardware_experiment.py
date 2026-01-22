#!/usr/bin/env python3
"""
IBM Quantum Hardware Experiment Script
======================================

Run VQE experiments on real IBM Quantum hardware to validate
noise resilience predictions from simulator experiments.

This script:
1. Runs noise-aware VQE on real hardware (2 CNOTs - best for NISQ)
2. Optionally runs hardware-efficient ansatz (6 CNOTs)
3. Compares results to simulator predictions
4. Saves results for combined analysis

Prerequisites:
    - pip install h2-vqe[ibm]
    - IBM Quantum account configured:
      QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

Usage:
    python scripts/run_hardware_experiment.py
    python scripts/run_hardware_experiment.py --backend ibm_brisbane
    python scripts/run_hardware_experiment.py --dry-run  # Test without hardware
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Run VQE on IBM Quantum hardware"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="IBM backend name (default: least busy)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/hardware_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=0.74,
        help="H2 bond length in Angstroms",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=30,
        help="Maximum optimizer iterations",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of measurement shots",
    )
    parser.add_argument(
        "--ansatz",
        type=str,
        nargs="+",
        default=["noise_aware"],
        help="Ansatz types to run (default: noise_aware only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test without submitting to hardware",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and exit",
    )

    args = parser.parse_args()

    # Import IBM runtime components
    try:
        from h2_vqe.ibm_runtime import (
            check_ibm_runtime,
            get_service,
            list_available_backends,
            get_least_busy_backend,
            run_vqe_on_hardware,
        )
    except ImportError as e:
        print(f"Error importing ibm_runtime: {e}")
        print("Install with: pip install h2-vqe[ibm]")
        sys.exit(1)

    # Check if runtime is available
    if not check_ibm_runtime():
        print("\nInstall IBM Runtime with: pip install qiskit-ibm-runtime")
        sys.exit(1)

    print("=" * 60)
    print("H2 VQE IBM Quantum Hardware Experiment")
    print("=" * 60)

    # Get service
    try:
        service = get_service()
        print("IBM Quantum service connected!")
    except Exception as e:
        print(f"Error connecting to IBM Quantum: {e}")
        sys.exit(1)

    # List backends if requested
    if args.list_backends:
        print("\nAvailable IBM Quantum Backends:")
        print("-" * 60)
        backends = list_available_backends(service, min_qubits=4)
        for b in backends:
            status = "OK" if b["operational"] else "DOWN"
            sim = " (sim)" if b["simulator"] else ""
            print(f"  {b['name']:20s} {b['n_qubits']:3d}q  {status:4s}  "
                  f"queue={b['pending_jobs']}{sim}")
        return

    # Determine backend
    if args.backend:
        backend_name = args.backend
    else:
        backend_name = get_least_busy_backend(service)
        print(f"Selected least busy backend: {backend_name}")

    # Dry run mode
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("Would run the following experiment:")
        print(f"  Backend: {backend_name}")
        print(f"  Bond length: {args.bond_length} A")
        print(f"  Ansatze: {args.ansatz}")
        print(f"  Max iterations: {args.maxiter}")
        print(f"  Shots: {args.shots}")
        print("\nNo jobs submitted.")
        return

    # Import molecular data
    from h2_vqe.molecular import compute_h2_integrals

    # Compute molecular data
    mol_data = compute_h2_integrals(args.bond_length)

    print(f"\nStarting experiment at {datetime.now().isoformat()}")
    print(f"Bond length: {args.bond_length} A")
    print(f"FCI energy: {mol_data.fci_energy:.6f} Ha")
    print(f"Backend: {backend_name}")
    print("-" * 60)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "backend": backend_name,
            "bond_length": args.bond_length,
            "fci_energy": mol_data.fci_energy,
            "maxiter": args.maxiter,
            "shots": args.shots,
        },
        "results": {}
    }

    for ansatz in args.ansatz:
        print(f"\n{'='*40}")
        print(f"Running {ansatz} ansatz on {backend_name}")
        print(f"{'='*40}")

        try:
            result = run_vqe_on_hardware(
                mol_data,
                ansatz_type=ansatz,
                backend_name=backend_name,
                service=service,
                maxiter=args.maxiter,
                shots=args.shots,
            )

            results["results"][ansatz] = {
                "energy": result.energy,
                "error_mHa": result.error * 1000 if result.error else None,
                "parameters": result.parameters.tolist(),
                "n_iterations": result.n_iterations,
                "n_evaluations": result.n_evaluations,
                "convergence": result.convergence,
                "job_ids": result.job_ids,
            }

            print(f"\n{ansatz} completed:")
            print(f"  Energy: {result.energy:.6f} Ha")
            if result.error:
                print(f"  Error: {result.error*1000:.2f} mHa")

        except Exception as e:
            print(f"Error running {ansatz}: {e}")
            results["results"][ansatz] = {"error": str(e)}

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"Backend: {backend_name}")
    print(f"FCI reference: {mol_data.fci_energy:.6f} Ha")
    print("-" * 60)
    for ansatz, data in results["results"].items():
        if "error" not in data or data.get("energy"):
            print(f"{ansatz:20s}: E={data['energy']:.6f} Ha, "
                  f"error={data.get('error_mHa', 'N/A'):.2f} mHa")
        else:
            print(f"{ansatz:20s}: FAILED - {data.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
