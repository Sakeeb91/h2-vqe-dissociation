#!/usr/bin/env python3
"""
ZNE Results Analysis Script
===========================

Analyze benchmark results and compute ZNE effectiveness metrics.

Metrics computed:
1. Mean Absolute Error (MAE) for each condition vs FCI
2. ZNE improvement ratio (error_raw / error_ZNE)
3. Noise model prediction accuracy (|sim_noisy - hw_raw|)
4. Equilibrium geometry error analysis
5. Statistical summary with confidence intervals

Usage:
    python scripts/analyze_zne_results.py results/zne_benchmark.json
    python scripts/analyze_zne_results.py results/zne_benchmark.json --plot
    python scripts/analyze_zne_results.py results/zne_benchmark.json --latex

Output:
    - Console summary with key metrics
    - Optional: Publication-quality figure
    - Optional: LaTeX table for papers
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def compute_mae(energies: np.ndarray, reference: np.ndarray) -> float:
    """Compute Mean Absolute Error in mHa."""
    return np.mean(np.abs(energies - reference)) * 1000


def compute_max_error(energies: np.ndarray, reference: np.ndarray) -> float:
    """Compute maximum error in mHa."""
    return np.max(np.abs(energies - reference)) * 1000


def compute_rmse(energies: np.ndarray, reference: np.ndarray) -> float:
    """Compute Root Mean Squared Error in mHa."""
    return np.sqrt(np.mean((energies - reference) ** 2)) * 1000


def analyze_results(results: dict, verbose: bool = True) -> dict:
    """
    Analyze ZNE benchmark results and compute all metrics.

    Args:
        results: Loaded benchmark results dict
        verbose: Print analysis to console

    Returns:
        Dict with computed metrics
    """
    # Extract data
    bond_lengths = np.array(results["bond_lengths"])
    fci = np.array(results["fci"])

    metrics = {
        "bond_lengths": bond_lengths.tolist(),
        "n_points": len(bond_lengths),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("ZNE Benchmark Analysis")
        print("=" * 70)
        print(f"\nBond lengths: {bond_lengths.tolist()} Å")
        print(f"Number of data points: {len(bond_lengths)}")

    # Equilibrium geometry analysis
    eq_idx = np.argmin(fci)
    eq_r = bond_lengths[eq_idx]
    eq_fci = fci[eq_idx]

    metrics["equilibrium"] = {
        "bond_length": float(eq_r),
        "fci_energy": float(eq_fci),
        "index": int(eq_idx),
    }

    if verbose:
        print(f"\nEquilibrium geometry: r = {eq_r:.2f} Å")
        print(f"FCI equilibrium energy: {eq_fci:.6f} Ha")

    # === Simulator analysis ===
    if verbose:
        print("\n" + "-" * 70)
        print("Simulator Results")
        print("-" * 70)

    if "sim_noiseless" in results:
        sim_noiseless = np.array(results["sim_noiseless"])
        mae = compute_mae(sim_noiseless, fci)
        max_err = compute_max_error(sim_noiseless, fci)
        rmse = compute_rmse(sim_noiseless, fci)
        eq_err = abs(sim_noiseless[eq_idx] - fci[eq_idx]) * 1000

        metrics["sim_noiseless"] = {
            "mae_mHa": float(mae),
            "max_error_mHa": float(max_err),
            "rmse_mHa": float(rmse),
            "eq_error_mHa": float(eq_err),
        }

        if verbose:
            print(f"\nNoiseless VQE:")
            print(f"  MAE:       {mae:6.2f} mHa")
            print(f"  Max error: {max_err:6.2f} mHa")
            print(f"  RMSE:      {rmse:6.2f} mHa")
            print(f"  Eq. error: {eq_err:6.2f} mHa")

    if "sim_noisy" in results:
        sim_noisy = np.array(results["sim_noisy"])
        mae = compute_mae(sim_noisy, fci)
        max_err = compute_max_error(sim_noisy, fci)
        rmse = compute_rmse(sim_noisy, fci)
        eq_err = abs(sim_noisy[eq_idx] - fci[eq_idx]) * 1000

        metrics["sim_noisy"] = {
            "mae_mHa": float(mae),
            "max_error_mHa": float(max_err),
            "rmse_mHa": float(rmse),
            "eq_error_mHa": float(eq_err),
        }

        if verbose:
            print(f"\nNoisy VQE (IBM-like noise):")
            print(f"  MAE:       {mae:6.2f} mHa")
            print(f"  Max error: {max_err:6.2f} mHa")
            print(f"  RMSE:      {rmse:6.2f} mHa")
            print(f"  Eq. error: {eq_err:6.2f} mHa")

    # === Hardware analysis ===
    has_hardware = "hw_raw" in results and "hw_zne" in results

    if has_hardware:
        hw_raw = np.array(results["hw_raw"])
        hw_zne = np.array(results["hw_zne"])

        if verbose:
            print("\n" + "-" * 70)
            print("Hardware Results")
            print("-" * 70)

        # Raw hardware
        mae_raw = compute_mae(hw_raw, fci)
        max_err_raw = compute_max_error(hw_raw, fci)
        rmse_raw = compute_rmse(hw_raw, fci)
        eq_err_raw = abs(hw_raw[eq_idx] - fci[eq_idx]) * 1000

        metrics["hw_raw"] = {
            "mae_mHa": float(mae_raw),
            "max_error_mHa": float(max_err_raw),
            "rmse_mHa": float(rmse_raw),
            "eq_error_mHa": float(eq_err_raw),
        }

        if verbose:
            print(f"\nHardware (raw, no mitigation):")
            print(f"  MAE:       {mae_raw:6.2f} mHa")
            print(f"  Max error: {max_err_raw:6.2f} mHa")
            print(f"  RMSE:      {rmse_raw:6.2f} mHa")
            print(f"  Eq. error: {eq_err_raw:6.2f} mHa")

        # ZNE hardware
        mae_zne = compute_mae(hw_zne, fci)
        max_err_zne = compute_max_error(hw_zne, fci)
        rmse_zne = compute_rmse(hw_zne, fci)
        eq_err_zne = abs(hw_zne[eq_idx] - fci[eq_idx]) * 1000

        metrics["hw_zne"] = {
            "mae_mHa": float(mae_zne),
            "max_error_mHa": float(max_err_zne),
            "rmse_mHa": float(rmse_zne),
            "eq_error_mHa": float(eq_err_zne),
        }

        if verbose:
            print(f"\nHardware (ZNE, resilience_level=2):")
            print(f"  MAE:       {mae_zne:6.2f} mHa")
            print(f"  Max error: {max_err_zne:6.2f} mHa")
            print(f"  RMSE:      {rmse_zne:6.2f} mHa")
            print(f"  Eq. error: {eq_err_zne:6.2f} mHa")

        # === ZNE Improvement Analysis ===
        if verbose:
            print("\n" + "-" * 70)
            print("ZNE Effectiveness Analysis")
            print("-" * 70)

        # Overall improvement
        zne_improvement_mae = mae_raw / mae_zne if mae_zne > 0 else float("inf")
        zne_improvement_rmse = rmse_raw / rmse_zne if rmse_zne > 0 else float("inf")
        zne_improvement_eq = eq_err_raw / eq_err_zne if eq_err_zne > 0 else float("inf")

        metrics["zne_improvement"] = {
            "mae_factor": float(zne_improvement_mae),
            "rmse_factor": float(zne_improvement_rmse),
            "eq_factor": float(zne_improvement_eq),
            "mae_reduction_pct": float((1 - mae_zne / mae_raw) * 100) if mae_raw > 0 else 0,
        }

        if verbose:
            print(f"\nZNE Improvement Factors (raw/ZNE, >1 = ZNE helped):")
            print(f"  MAE improvement:  {zne_improvement_mae:.2f}x")
            print(f"  RMSE improvement: {zne_improvement_rmse:.2f}x")
            print(f"  Eq. improvement:  {zne_improvement_eq:.2f}x")

            if zne_improvement_mae > 1:
                reduction = (1 - mae_zne / mae_raw) * 100
                print(f"\n  ZNE reduced MAE by {reduction:.1f}%")
            else:
                increase = (mae_zne / mae_raw - 1) * 100
                print(f"\n  Warning: ZNE increased MAE by {increase:.1f}%")

        # Per-geometry improvement
        err_raw = np.abs(hw_raw - fci)
        err_zne = np.abs(hw_zne - fci)
        per_point_improvement = np.where(err_zne > 1e-10, err_raw / err_zne, 0)

        metrics["per_geometry_improvement"] = per_point_improvement.tolist()

        if verbose:
            print(f"\nPer-geometry improvement factors:")
            for r, imp in zip(bond_lengths, per_point_improvement):
                status = "improved" if imp > 1 else "degraded"
                print(f"  r = {r:.2f} Å: {imp:.2f}x ({status})")

        # === Noise Model Prediction Accuracy ===
        if "sim_noisy" in results:
            sim_noisy = np.array(results["sim_noisy"])
            noise_model_error = np.mean(np.abs(sim_noisy - hw_raw)) * 1000
            noise_model_max = np.max(np.abs(sim_noisy - hw_raw)) * 1000

            metrics["noise_model_accuracy"] = {
                "mae_vs_hw_mHa": float(noise_model_error),
                "max_diff_mHa": float(noise_model_max),
            }

            if verbose:
                print(f"\nNoise Model Prediction Accuracy:")
                print(f"  MAE(sim_noisy - hw_raw): {noise_model_error:.2f} mHa")
                print(f"  Max deviation:           {noise_model_max:.2f} mHa")

    # === Summary ===
    if verbose:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)

        # Chemical accuracy threshold
        chem_acc = 1.6  # mHa

        print(f"\nChemical accuracy threshold: {chem_acc} mHa")
        print("\nMethods achieving chemical accuracy (MAE < 1.6 mHa):")

        if "sim_noiseless" in metrics and metrics["sim_noiseless"]["mae_mHa"] < chem_acc:
            print(f"  - Noiseless VQE: {metrics['sim_noiseless']['mae_mHa']:.2f} mHa")

        if "sim_noisy" in metrics and metrics["sim_noisy"]["mae_mHa"] < chem_acc:
            print(f"  - Noisy VQE:     {metrics['sim_noisy']['mae_mHa']:.2f} mHa")

        if has_hardware:
            if metrics["hw_raw"]["mae_mHa"] < chem_acc:
                print(f"  - Hardware raw:  {metrics['hw_raw']['mae_mHa']:.2f} mHa")
            if metrics["hw_zne"]["mae_mHa"] < chem_acc:
                print(f"  - Hardware ZNE:  {metrics['hw_zne']['mae_mHa']:.2f} mHa")

        if has_hardware:
            print(f"\nKey finding: ZNE {'improved' if zne_improvement_mae > 1 else 'did not improve'} "
                  f"hardware accuracy by {abs(zne_improvement_mae - 1) * 100:.1f}%")

    return metrics


def generate_latex_table(metrics: dict) -> str:
    """Generate LaTeX table from analysis metrics."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{ZNE Benchmark Results for H$_2$ VQE}",
        r"\label{tab:zne_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & MAE (mHa) & Max Error (mHa) & RMSE (mHa) & Eq. Error (mHa) \\",
        r"\midrule",
    ]

    # Add data rows
    if "sim_noiseless" in metrics:
        m = metrics["sim_noiseless"]
        lines.append(
            f"Simulator (noiseless) & {m['mae_mHa']:.2f} & {m['max_error_mHa']:.2f} & "
            f"{m['rmse_mHa']:.2f} & {m['eq_error_mHa']:.2f} \\\\"
        )

    if "sim_noisy" in metrics:
        m = metrics["sim_noisy"]
        lines.append(
            f"Simulator (IBM noise) & {m['mae_mHa']:.2f} & {m['max_error_mHa']:.2f} & "
            f"{m['rmse_mHa']:.2f} & {m['eq_error_mHa']:.2f} \\\\"
        )

    if "hw_raw" in metrics:
        lines.append(r"\midrule")
        m = metrics["hw_raw"]
        lines.append(
            f"Hardware (raw) & {m['mae_mHa']:.2f} & {m['max_error_mHa']:.2f} & "
            f"{m['rmse_mHa']:.2f} & {m['eq_error_mHa']:.2f} \\\\"
        )

    if "hw_zne" in metrics:
        m = metrics["hw_zne"]
        lines.append(
            f"Hardware (ZNE) & {m['mae_mHa']:.2f} & {m['max_error_mHa']:.2f} & "
            f"{m['rmse_mHa']:.2f} & {m['eq_error_mHa']:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ZNE benchmark results"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to benchmark results JSON file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plot",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output LaTeX table",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Save computed metrics to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)

    # Analyze
    metrics = analyze_results(results, verbose=not args.quiet)

    # Save metrics if requested
    if args.output_metrics:
        Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output_metrics}")

    # Generate LaTeX table if requested
    if args.latex:
        latex = generate_latex_table(metrics)
        print("\n" + "=" * 70)
        print("LaTeX Table")
        print("=" * 70)
        print(latex)

    # Generate plot if requested
    if args.plot:
        try:
            from h2_vqe.visualization import plot_zne_comparison
            import numpy as np

            plot_path = args.results_file.replace(".json", "_analysis.png")

            # Prepare arrays
            bond_lengths = np.array(results["bond_lengths"])
            fci = np.array(results["fci"])
            sim_noiseless = np.array(results.get("sim_noiseless", []))
            sim_noisy = np.array(results.get("sim_noisy", []))
            hw_raw = np.array(results["hw_raw"]) if "hw_raw" in results else None
            hw_zne = np.array(results["hw_zne"]) if "hw_zne" in results else None

            plot_zne_comparison(
                bond_lengths=bond_lengths,
                fci_energies=fci,
                sim_noiseless=sim_noiseless,
                sim_noisy=sim_noisy,
                hw_raw=hw_raw,
                hw_zne=hw_zne,
                save_path=plot_path,
            )
        except Exception as e:
            print(f"\nCould not generate plot: {e}")


if __name__ == "__main__":
    main()
