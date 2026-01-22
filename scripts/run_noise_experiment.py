#!/usr/bin/env python3
"""
Noise Resilience Experiment Script
==================================

Run VQE experiments across multiple ansatz types and noise levels
to generate a noise resilience heatmap.

This script:
1. Runs VQE for all ansatz/noise combinations
2. Saves raw data to JSON
3. Generates heatmap visualization

Usage:
    python scripts/run_noise_experiment.py
    python scripts/run_noise_experiment.py --quick  # Fast test run
    python scripts/run_noise_experiment.py --output results/my_experiment.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from h2_vqe.visualization import compute_noise_resilience_data, plot_noise_resilience_heatmap


def main():
    parser = argparse.ArgumentParser(
        description="Run noise resilience experiment for H2 VQE"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with reduced parameters",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/noise_resilience_simulator.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="results/noise_resilience_heatmap.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=0.74,
        help="H2 bond length in Angstroms (default: 0.74)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum optimizer iterations",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of measurement shots for noisy simulation",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=1,
        help="Number of random restarts for optimization",
    )
    parser.add_argument(
        "--ansatz",
        type=str,
        nargs="+",
        default=["uccsd", "hardware_efficient", "noise_aware"],
        help="Ansatz types to test",
    )
    parser.add_argument(
        "--noise",
        type=str,
        nargs="+",
        default=["ideal", "low_noise", "ibm_like", "high_noise"],
        help="Noise presets to test",
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.maxiter = 30
        args.shots = 1024
        args.ansatz = ["noise_aware", "hardware_efficient"]
        args.noise = ["ideal", "ibm_like"]
        print("=== QUICK TEST MODE ===")

    # Ensure output directories exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("H2 VQE Noise Resilience Experiment")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Bond length: {args.bond_length} A")
    print(f"Ansatze: {args.ansatz}")
    print(f"Noise levels: {args.noise}")
    print(f"Max iterations: {args.maxiter}")
    print(f"Shots: {args.shots}")
    print(f"Random restarts: {args.n_starts}")
    print("=" * 60)
    print()

    # Run experiments
    data = compute_noise_resilience_data(
        ansatz_types=args.ansatz,
        noise_presets=args.noise,
        bond_length=args.bond_length,
        maxiter=args.maxiter,
        shots=args.shots,
        n_starts=args.n_starts,
        verbose=True,
    )

    # Add experiment metadata
    data["experiment_info"] = {
        "timestamp": datetime.now().isoformat(),
        "quick_mode": args.quick,
    }

    # Save to JSON
    def json_serializer(obj):
        """Handle numpy arrays and other non-serializable types."""
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, default=json_serializer)
    print(f"\nData saved to: {args.output}")

    # Generate heatmap
    fig = plot_noise_resilience_heatmap(data=data, save_path=args.figure)

    print()
    print("=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print(f"End time: {datetime.now().isoformat()}")
    print(f"Data: {args.output}")
    print(f"Figure: {args.figure}")

    # Print summary table
    print("\nResults Summary (Error in mHa):")
    print("-" * 60)
    header = "Ansatz".ljust(20) + "".join(n.center(12) for n in args.noise)
    print(header)
    print("-" * 60)
    for i, ansatz in enumerate(args.ansatz):
        row = ansatz.ljust(20)
        for j in range(len(args.noise)):
            row += f"{data['error_matrix'][i][j]:8.2f}    "
        print(row)


if __name__ == "__main__":
    main()
