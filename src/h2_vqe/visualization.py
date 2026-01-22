"""
Visualization for H₂ VQE Results
================================

This module provides publication-quality plotting functions for
visualizing H₂ dissociation curves and VQE performance.

The main output is a 4-panel figure showing:
1. Full dissociation curve (energy vs bond length)
2. VQE error vs bond length
3. Correlation energy analysis
4. Convergence behavior

New advanced visualizations:
- plot_convergence_landscape: Energy landscape in parameter space
- plot_noise_resilience_heatmap: Ansatz vs noise level error analysis

Example:
    >>> from h2_vqe.dissociation import compute_dissociation_curve
    >>> from h2_vqe.visualization import create_dissociation_figure
    >>> results = compute_dissociation_curve(n_points=20)
    >>> fig = create_dissociation_figure(results)
    >>> fig.savefig("h2_dissociation.png", dpi=300)
"""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

from h2_vqe.dissociation import DissociationCurveResult


# Publication-quality style settings
STYLE_CONFIG = {
    "figure.figsize": (12, 10),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
}

# Color palette for different methods
COLORS = {
    "hf": "#4477AA",      # Blue
    "fci": "#228833",     # Green
    "uccsd": "#EE6677",   # Red
    "hardware_efficient": "#CCBB44",  # Yellow
    "noise_aware": "#AA3377",  # Purple
    "default": "#BBBBBB", # Gray
}


def apply_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_CONFIG)


def create_dissociation_figure(
    results: DissociationCurveResult,
    title: str = "H₂ Molecular Dissociation Curve via VQE",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Create publication-quality 4-panel dissociation curve figure.

    Panels:
        (a) Energy vs bond length - comparing HF, FCI, and VQE methods
        (b) VQE error vs bond length - accuracy analysis
        (c) Correlation energy - E_FCI - E_HF analysis
        (d) Energy bar chart at equilibrium - method comparison

    Args:
        results: DissociationCurveResult from compute_dissociation_curve
        title: Figure title
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    r = results.bond_lengths

    # Panel (a): Full dissociation curve
    ax_a = axes[0, 0]
    ax_a.plot(r, results.hf_energies, "-", color=COLORS["hf"],
              label="Hartree-Fock", linewidth=2)
    ax_a.plot(r, results.fci_energies, "-", color=COLORS["fci"],
              label="FCI (Exact)", linewidth=2)

    for ansatz in results.ansatz_types:
        color = COLORS.get(ansatz, COLORS["default"])
        ax_a.plot(r, results.vqe_energies[ansatz], "o--", color=color,
                  label=f"VQE ({ansatz})", markersize=5, linewidth=1.5)

    ax_a.set_xlabel("Bond Length (Å)")
    ax_a.set_ylabel("Energy (Hartree)")
    ax_a.set_title("(a) Potential Energy Curve")
    ax_a.legend(loc="upper right", framealpha=0.9)
    ax_a.grid(True, alpha=0.3)

    # Mark equilibrium
    eq_idx = np.argmin(results.fci_energies)
    ax_a.axvline(r[eq_idx], color="gray", linestyle=":", alpha=0.5)

    # Panel (b): VQE Error
    ax_b = axes[0, 1]
    for ansatz in results.ansatz_types:
        color = COLORS.get(ansatz, COLORS["default"])
        errors_mha = results.vqe_errors[ansatz] * 1000  # Convert to mHa
        ax_b.plot(r, errors_mha, "o-", color=color,
                  label=f"{ansatz}", markersize=5)

    ax_b.set_xlabel("Bond Length (Å)")
    ax_b.set_ylabel("Error (mHa)")
    ax_b.set_title("(b) VQE Error |E_VQE - E_FCI|")
    ax_b.legend(loc="upper right")
    ax_b.grid(True, alpha=0.3)

    # Add chemical accuracy line (1.6 mHa = 1 kcal/mol)
    ax_b.axhline(1.6, color="green", linestyle="--", alpha=0.7,
                 label="Chemical accuracy")

    # Panel (c): Correlation energy
    ax_c = axes[1, 0]
    correlation = results.fci_energies - results.hf_energies
    ax_c.fill_between(r, correlation * 1000, alpha=0.3, color=COLORS["fci"])
    ax_c.plot(r, correlation * 1000, "-", color=COLORS["fci"], linewidth=2)

    ax_c.set_xlabel("Bond Length (Å)")
    ax_c.set_ylabel("Correlation Energy (mHa)")
    ax_c.set_title("(c) Electron Correlation (E_FCI - E_HF)")
    ax_c.grid(True, alpha=0.3)
    ax_c.axvline(r[eq_idx], color="gray", linestyle=":", alpha=0.5)

    # Panel (d): Energy comparison at equilibrium
    ax_d = axes[1, 1]

    # Data for bar chart
    methods = ["HF"]
    energies = [results.hf_energies[eq_idx]]
    colors = [COLORS["hf"]]

    for ansatz in results.ansatz_types:
        methods.append(f"VQE\n({ansatz})")
        energies.append(results.vqe_energies[ansatz][eq_idx])
        colors.append(COLORS.get(ansatz, COLORS["default"]))

    methods.append("FCI")
    energies.append(results.fci_energies[eq_idx])
    colors.append(COLORS["fci"])

    x = np.arange(len(methods))
    bars = ax_d.bar(x, energies, color=colors, edgecolor="black", linewidth=0.5)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(methods)
    ax_d.set_ylabel("Energy (Hartree)")
    ax_d.set_title(f"(d) Energy at Equilibrium (r = {r[eq_idx]:.2f} Å)")
    ax_d.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{energy:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_convergence(
    energy_history: List[float],
    exact_energy: Optional[float] = None,
    title: str = "VQE Convergence",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot VQE optimization convergence.

    Args:
        energy_history: List of energies at each evaluation
        exact_energy: Optional exact (FCI) energy for reference
        title: Plot title
        ax: Optional matplotlib axes

    Returns:
        matplotlib Axes object
    """
    apply_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    evaluations = np.arange(1, len(energy_history) + 1)
    ax.plot(evaluations, energy_history, "b-", linewidth=1.5, label="VQE Energy")

    if exact_energy is not None:
        ax.axhline(exact_energy, color="green", linestyle="--",
                   label=f"FCI ({exact_energy:.6f} Ha)")

    ax.set_xlabel("Function Evaluation")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_error_vs_cnot_count(
    results: DissociationCurveResult,
    cnot_counts: dict,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot VQE error vs CNOT count to show noise-resilience tradeoff.

    Args:
        results: Dissociation curve results
        cnot_counts: Dict mapping ansatz name to CNOT count
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for ansatz in results.ansatz_types:
        if ansatz in cnot_counts:
            mean_error = results.mean_vqe_error(ansatz) * 1000  # mHa
            n_cnots = cnot_counts[ansatz]
            color = COLORS.get(ansatz, COLORS["default"])
            ax.scatter([n_cnots], [mean_error], s=150, color=color,
                      label=ansatz, edgecolor="black", linewidth=1)

    ax.set_xlabel("Number of CNOT Gates")
    ax.set_ylabel("Mean VQE Error (mHa)")
    ax.set_title("Accuracy vs Circuit Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_simple_curve(
    bond_lengths: np.ndarray,
    fci_energies: np.ndarray,
    vqe_energies: np.ndarray,
    title: str = "H₂ Dissociation Curve",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create a simple single-panel dissociation curve plot.

    Args:
        bond_lengths: Array of bond distances
        fci_energies: Array of FCI energies
        vqe_energies: Array of VQE energies
        title: Plot title
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(bond_lengths, fci_energies, "b-", linewidth=2, label="FCI (Exact)")
    ax.plot(bond_lengths, vqe_energies, "ro--", markersize=6, label="VQE")

    ax.set_xlabel("Bond Length (Å)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_convergence_landscape(
    bond_length: float = 0.74,
    ansatz_type: str = "uccsd",
    n_points: int = 100,
    param_range: tuple = (-np.pi, np.pi),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot VQE energy landscape in parameter space.

    For the 1-parameter UCCSD ansatz, this creates a 1D potential energy
    surface showing the optimization landscape. This helps understand
    convergence behavior and identify local minima or barren plateaus.

    Args:
        bond_length: H₂ bond length (default: equilibrium)
        ansatz_type: Ansatz to use (best for low-dimensional parameter spaces)
        n_points: Number of parameter values to sample
        param_range: (min, max) range for first parameter
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_convergence_landscape(ansatz_type="uccsd")
        >>> fig.savefig("landscape.png", dpi=300)

    Notes:
        For multi-parameter ansatze, this plots energy vs first parameter
        while keeping other parameters at their initial values.
    """
    from h2_vqe.molecular import compute_h2_integrals
    from h2_vqe.hamiltonian import build_qubit_hamiltonian
    from h2_vqe.ansatz import create_ansatz, get_initial_parameters
    from h2_vqe.vqe import VQEEngine

    apply_style()

    # Compute molecular data
    mol_data = compute_h2_integrals(bond_length)
    hamiltonian = build_qubit_hamiltonian(mol_data)

    # Create ansatz
    ansatz = create_ansatz(ansatz_type, n_qubits=4)
    n_params = ansatz.num_parameters

    print(f"Plotting convergence landscape for {ansatz_type}")
    print(f"Bond length: {bond_length} Å")
    print(f"Number of parameters: {n_params}")

    # Create VQE engine
    vqe = VQEEngine(hamiltonian, ansatz, backend="statevector")

    # Get initial parameter values
    initial_params = get_initial_parameters(ansatz)

    # Sample parameter space
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    energies = []

    print(f"Sampling {n_points} parameter values...")

    for i, p in enumerate(param_values):
        # Set first parameter, keep rest at initial values
        test_params = initial_params.copy()
        test_params[0] = p

        # Compute energy
        energy = vqe.compute_energy(test_params)
        energies.append(energy)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_points} complete")

    energies = np.array(energies)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot energy landscape
    ax.plot(param_values, energies, "b-", linewidth=2, label="VQE Energy")

    # Add FCI reference line
    ax.axhline(mol_data.fci_energy, color=COLORS["fci"], linestyle="--",
               linewidth=2, label=f"FCI Energy ({mol_data.fci_energy:.6f} Ha)")

    # Add HF reference line
    ax.axhline(mol_data.hf_energy, color=COLORS["hf"], linestyle=":",
               linewidth=2, label=f"HF Energy ({mol_data.hf_energy:.6f} Ha)")

    # Mark global minimum
    min_idx = np.argmin(energies)
    ax.plot(param_values[min_idx], energies[min_idx], "ro",
            markersize=10, label=f"Global Min ({energies[min_idx]:.6f} Ha)")

    # Annotate with error at minimum
    error_mha = (energies[min_idx] - mol_data.fci_energy) * 1000
    ax.annotate(
        f"Error: {error_mha:.2f} mHa",
        xy=(param_values[min_idx], energies[min_idx]),
        xytext=(10, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax.set_xlabel(f"Parameter θ₁ (radians)" if n_params == 1 else "First Parameter θ₁ (radians)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(f"VQE Energy Landscape: {ansatz_type} ansatz (H₂ at r = {bond_length} Å)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add parameter range annotation
    ax.text(0.02, 0.98, f"Total parameters: {n_params}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nLandscape saved to: {save_path}")

    return fig


def compute_noise_resilience_data(
    ansatz_types: List[str] = ["uccsd", "hardware_efficient", "noise_aware"],
    noise_presets: List[str] = ["ideal", "low_noise", "ibm_like", "high_noise"],
    bond_length: float = 0.74,
    maxiter: int = 100,
    shots: int = 4096,
    n_starts: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Compute noise resilience data for all ansatz/noise combinations.

    This function runs VQE experiments and returns structured data that can
    be saved to JSON and/or visualized with plot_noise_resilience_heatmap.

    Args:
        ansatz_types: List of ansatz types to compare
        noise_presets: List of noise preset names
        bond_length: H₂ bond length to compute at (default: equilibrium)
        maxiter: Maximum optimizer iterations (lower for noisy to help convergence)
        shots: Number of measurement shots for noisy simulations
        n_starts: Number of random restarts (helps with noisy optimization)
        verbose: Print progress updates

    Returns:
        Dictionary containing:
            - error_matrix: 2D numpy array (ansatz x noise)
            - energy_matrix: 2D numpy array of final energies
            - ansatz_types: List of ansatz names
            - noise_presets: List of noise preset names
            - bond_length: Bond length used
            - fci_energy: Exact FCI energy for reference
            - metadata: Dict with computation parameters

    Example:
        >>> data = compute_noise_resilience_data()
        >>> import json
        >>> with open("results.json", "w") as f:
        ...     json.dump(data, f, default=lambda x: x.tolist())
    """
    from h2_vqe.molecular import compute_h2_integrals
    from h2_vqe.vqe import run_vqe, run_vqe_multistart
    from h2_vqe.noise import create_noise_model

    # Compute molecular data once
    mol_data = compute_h2_integrals(bond_length)

    # Build matrices: rows = ansatz, cols = noise level
    n_ansatz = len(ansatz_types)
    n_noise = len(noise_presets)
    error_matrix = np.zeros((n_ansatz, n_noise))
    energy_matrix = np.zeros((n_ansatz, n_noise))

    if verbose:
        print(f"Computing noise resilience data at r = {bond_length} Å")
        print(f"FCI energy: {mol_data.fci_energy:.6f} Ha")
        print(f"Ansatze: {ansatz_types}")
        print(f"Noise levels: {noise_presets}")
        print("-" * 60)

    for i, ansatz in enumerate(ansatz_types):
        for j, noise_preset in enumerate(noise_presets):
            # Determine noise model and backend
            if noise_preset == "ideal":
                noise_model = None
                backend = "statevector"
            else:
                noise_model = create_noise_model(noise_preset)
                backend = "aer_simulator"

            # Run VQE (with multistart if n_starts > 1)
            if n_starts > 1:
                result = run_vqe_multistart(
                    mol_data,
                    ansatz_type=ansatz,
                    n_starts=n_starts,
                    noise_model=noise_model,
                    backend=backend,
                    shots=shots,
                    maxiter=maxiter,
                )
            else:
                result = run_vqe(
                    mol_data,
                    ansatz_type=ansatz,
                    noise_model=noise_model,
                    backend=backend,
                    shots=shots,
                    maxiter=maxiter,
                )

            # Store results
            error_matrix[i, j] = result.error * 1000  # Convert to mHa
            energy_matrix[i, j] = result.energy

            if verbose:
                print(f"  {ansatz:20s} | {noise_preset:12s}: "
                      f"E={result.energy:.6f} Ha, error={error_matrix[i, j]:6.2f} mHa")

    return {
        "error_matrix": error_matrix,
        "energy_matrix": energy_matrix,
        "ansatz_types": ansatz_types,
        "noise_presets": noise_presets,
        "bond_length": bond_length,
        "fci_energy": mol_data.fci_energy,
        "metadata": {
            "maxiter": maxiter,
            "shots": shots,
            "n_starts": n_starts,
        }
    }


def plot_noise_resilience_heatmap(
    data: Optional[dict] = None,
    ansatz_types: List[str] = ["uccsd", "hardware_efficient", "noise_aware"],
    noise_presets: List[str] = ["ideal", "low_noise", "ibm_like", "high_noise"],
    bond_length: float = 0.74,
    save_path: Optional[str] = None,
    **compute_kwargs,
) -> Figure:
    """
    Create a heatmap showing VQE error across ansatz types and noise levels.

    This visualization answers: "Which ansatz survives which noise regime?"
    and is directly relevant to quantum hardware benchmarking.

    Args:
        data: Pre-computed data from compute_noise_resilience_data() (optional)
              If None, will compute data using provided parameters
        ansatz_types: List of ansatz types to compare (if computing)
        noise_presets: List of noise preset names (if computing)
        bond_length: H₂ bond length to compute at (default: equilibrium)
        save_path: Optional path to save figure
        **compute_kwargs: Additional arguments for compute_noise_resilience_data()

    Returns:
        matplotlib Figure with heatmap

    Example:
        >>> # Compute and plot in one call
        >>> fig = plot_noise_resilience_heatmap()
        >>> fig.savefig("noise_resilience.png", dpi=300)

        >>> # Or compute separately and plot
        >>> data = compute_noise_resilience_data()
        >>> fig = plot_noise_resilience_heatmap(data=data)
    """
    apply_style()

    # Compute data if not provided
    if data is None:
        data = compute_noise_resilience_data(
            ansatz_types=ansatz_types,
            noise_presets=noise_presets,
            bond_length=bond_length,
            **compute_kwargs,
        )

    # Extract data
    error_matrix = np.array(data["error_matrix"])
    ansatz_list = data["ansatz_types"]
    noise_list = data["noise_presets"]
    r = data["bond_length"]

    n_ansatz = len(ansatz_list)
    n_noise = len(noise_list)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use diverging colormap centered appropriately
    im = ax.imshow(error_matrix, cmap="RdYlGn_r", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(n_noise))
    ax.set_yticks(np.arange(n_ansatz))
    ax.set_xticklabels([preset.replace("_", "\n") for preset in noise_list])
    ax.set_yticklabels(ansatz_list)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("VQE Error (mHa)", rotation=270, labelpad=20)

    # Add text annotations with adaptive coloring
    for i in range(n_ansatz):
        for j in range(n_noise):
            val = error_matrix[i, j]
            # Use white text for dark cells, black for light cells
            text_color = "white" if val > np.median(error_matrix) else "black"
            ax.text(j, i, f"{val:.1f}",
                   ha="center", va="center", color=text_color, fontsize=10,
                   fontweight="bold")

    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Ansatz Type")
    ax.set_title(f"Noise Resilience Heatmap (H₂ at r = {r} Å)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    return fig


if __name__ == "__main__":
    # Demo with sample data
    print("Creating sample visualization...")

    # Generate sample data for demo
    from h2_vqe.dissociation import quick_curve

    results = quick_curve(n_points=5)

    # Create figure
    fig = create_dissociation_figure(
        results,
        title="H₂ VQE Dissociation Curve (Demo)",
        save_path="results/demo_figure.png",
    )

    plt.show()
