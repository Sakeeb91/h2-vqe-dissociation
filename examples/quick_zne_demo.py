#!/usr/bin/env python3
"""
Quick ZNE Benchmarking Demo
===========================

Demonstrates the ZNE benchmarking workflow using simulators only.
This script runs in ~30 seconds and produces a comparison figure.

Usage:
    cd /path/to/h2-vqe-dissociation
    python examples/quick_zne_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from h2_vqe.molecular import compute_h2_integrals
from h2_vqe.vqe import run_vqe
from h2_vqe.noise import create_noise_model
from h2_vqe.visualization import plot_zne_comparison


def main():
    print("=" * 60)
    print("Quick ZNE Benchmarking Demo (Simulator Only)")
    print("=" * 60)

    # Use 3 bond lengths for quick demo
    bond_lengths = np.array([0.5, 0.74, 1.5])
    n_points = len(bond_lengths)

    fci_energies = np.zeros(n_points)
    sim_noiseless = np.zeros(n_points)
    sim_noisy = np.zeros(n_points)

    # Create IBM-like noise model
    noise_model = create_noise_model("ibm_like")

    print(f"\nRunning VQE at {n_points} bond lengths...")
    print("-" * 40)

    for i, r in enumerate(bond_lengths):
        print(f"\n[{i+1}/{n_points}] Bond length = {r:.2f} Å")

        # Compute molecular data
        mol_data = compute_h2_integrals(r)
        fci_energies[i] = mol_data.fci_energy

        # Noiseless VQE
        result_noiseless = run_vqe(
            mol_data,
            ansatz_type="noise_aware",
            backend="statevector",
            maxiter=50,
        )
        sim_noiseless[i] = result_noiseless.energy

        # Noisy VQE
        result_noisy = run_vqe(
            mol_data,
            ansatz_type="noise_aware",
            backend="aer_simulator",
            noise_model=noise_model,
            shots=2048,
            maxiter=30,
        )
        sim_noisy[i] = result_noisy.energy

        print(f"  FCI:       {fci_energies[i]:.6f} Ha")
        print(f"  Noiseless: {sim_noiseless[i]:.6f} Ha (err: {result_noiseless.error*1000:.2f} mHa)")
        print(f"  Noisy:     {sim_noisy[i]:.6f} Ha (err: {result_noisy.error*1000:.2f} mHa)")

    # Compute summary metrics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    mae_noiseless = np.mean(np.abs(sim_noiseless - fci_energies)) * 1000
    mae_noisy = np.mean(np.abs(sim_noisy - fci_energies)) * 1000

    print(f"\nMean Absolute Error (MAE):")
    print(f"  Noiseless VQE: {mae_noiseless:.2f} mHa")
    print(f"  Noisy VQE:     {mae_noisy:.2f} mHa")
    print(f"\nNoise degradation: {mae_noisy / mae_noiseless:.1f}x worse")

    # Generate figure
    print("\nGenerating comparison figure...")
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "quick_zne_demo.png"

    fig = plot_zne_comparison(
        bond_lengths=bond_lengths,
        fci_energies=fci_energies,
        sim_noiseless=sim_noiseless,
        sim_noisy=sim_noisy,
        save_path=str(save_path),
    )

    print(f"\nFigure saved to: {save_path}")
    print("\nTo run full benchmark with hardware:")
    print("  python scripts/run_zne_benchmark.py")


if __name__ == "__main__":
    main()
