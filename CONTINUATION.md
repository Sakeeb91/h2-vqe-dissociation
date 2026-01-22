# H2-VQE Project Continuation Document

> **Purpose**: This document contains all context needed to continue development on a new Claude instance.

---

## Project Overview

**Repository**: `/Users/sakeeb/Code repositories/h2-vqe-dissociation`
**GitHub**: `https://github.com/Sakeeb91/h2-vqe-dissociation`

A Variational Quantum Eigensolver (VQE) implementation for computing H₂ molecular dissociation curves, with noise simulation and IBM Quantum hardware integration.

---

## Current State (as of Jan 22, 2026)

### What's Implemented ✅

| Component | File | Status |
|-----------|------|--------|
| Molecular integrals (PySCF) | `src/h2_vqe/molecular.py` | Complete |
| Jordan-Wigner Hamiltonian | `src/h2_vqe/hamiltonian.py` | Complete |
| Ansatze (UCCSD, HW-Efficient, Noise-Aware) | `src/h2_vqe/ansatz.py` | Complete |
| VQE Engine with noise support | `src/h2_vqe/vqe.py` | Complete |
| Noise models (IBM-like presets) | `src/h2_vqe/noise.py` | Complete |
| Dissociation curve computation | `src/h2_vqe/dissociation.py` | Complete |
| Visualization (heatmaps, curves) | `src/h2_vqe/visualization.py` | Complete |
| IBM Runtime integration | `src/h2_vqe/ibm_runtime.py` | Complete |
| Noise experiment script | `scripts/run_noise_experiment.py` | Complete |
| Hardware experiment script | `scripts/run_hardware_experiment.py` | Complete |
| Tests | `tests/` | 119 tests passing |

### Recent Commits (8 atomic commits pushed)

```
7f3a8e5 Add tests for noisy VQE execution
7c9c787 Update package docstring to document noise and hardware features
fc5b47a Add IBM Quantum hardware experiment runner script
f55c44d Add IBM Quantum Runtime integration module
ee22c14 Add qiskit-ibm-runtime as optional dependency
b8c8017 Add noise resilience experiment runner script
43023bb Add compute_noise_resilience_data function and improve heatmap plotting
c1a0f01 Add noise_model parameter to VQE engine and run_vqe function
```

---

## Next Phase: ZNE Benchmarking Study

### Scientific Goal

**Research Question**: *"How effective is Zero-Noise Extrapolation (ZNE) for VQE on real IBM Quantum hardware?"*

### Study Design

Compare H₂ dissociation curves across 5 conditions:

| Condition | Source | Purpose |
|-----------|--------|---------|
| FCI (exact) | Classical | Ground truth |
| Simulator (noiseless) | Qiskit Aer statevector | VQE accuracy baseline |
| Simulator (IBM noise) | Qiskit Aer + noise model | Predicted hardware performance |
| Hardware (raw) | IBM Quantum, resilience_level=0 | Actual NISQ performance |
| Hardware (ZNE) | IBM Quantum, resilience_level=2 | Error-mitigated performance |

### Key Metrics to Compute

1. **Prediction accuracy**: |E_simulator_noisy - E_hardware_raw|
2. **ZNE improvement**: |E_hardware_raw - E_FCI| vs |E_hardware_ZNE - E_FCI|
3. **Shot overhead**: Shots needed for ZNE vs raw
4. **Dissociation curve quality**: Can we capture the equilibrium geometry?

---

## Implementation Tasks

### Task 1: Update `ibm_runtime.py` for ZNE Sweeps

Add function to run VQE at multiple resilience levels:

```python
def run_vqe_resilience_sweep(
    mol_data,
    ansatz_type: str = "noise_aware",
    resilience_levels: List[int] = [0, 1, 2],
    backend_name: Optional[str] = None,
    service: Optional["QiskitRuntimeService"] = None,
    maxiter: int = 30,
    shots: int = 4096,
) -> dict:
    """
    Run VQE at multiple error mitigation levels for comparison.

    Args:
        resilience_levels:
            0 = raw (no mitigation)
            1 = M3 readout error mitigation
            2 = ZNE (zero-noise extrapolation)

    Returns:
        Dict mapping resilience_level -> HardwareResult
    """
```

### Task 2: Create `scripts/run_zne_benchmark.py`

Script to run the full ZNE benchmarking study:

```python
#!/usr/bin/env python3
"""
ZNE Benchmarking Study
======================

Compare VQE performance across:
1. Exact (FCI)
2. Simulator (noiseless)
3. Simulator (IBM-like noise)
4. Hardware (raw, resilience_level=0)
5. Hardware (ZNE, resilience_level=2)

Usage:
    python scripts/run_zne_benchmark.py --dry-run
    python scripts/run_zne_benchmark.py --bond-lengths 0.5 0.74 1.0 1.5 2.0
"""

# Bond lengths to test (5 points capturing key physics)
BOND_LENGTHS = [0.5, 0.74, 1.0, 1.5, 2.0]

# Main workflow:
# 1. Compute FCI energies at all bond lengths
# 2. Run simulator (noiseless) VQE
# 3. Run simulator (IBM-like noise) VQE
# 4. Run hardware VQE with resilience_level=0 (raw)
# 5. Run hardware VQE with resilience_level=2 (ZNE)
# 6. Save results to JSON
# 7. Generate comparison plots
```

### Task 3: Add ZNE Comparison Visualization

Add to `visualization.py`:

```python
def plot_zne_comparison(
    bond_lengths: np.ndarray,
    fci_energies: np.ndarray,
    sim_noiseless: np.ndarray,
    sim_noisy: np.ndarray,
    hw_raw: np.ndarray,
    hw_zne: np.ndarray,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create comparison plot showing ZNE effectiveness.

    Panels:
    (a) Energy curves for all 5 conditions
    (b) Error vs bond length (|E - E_FCI|)
    (c) ZNE improvement factor
    """
```

### Task 4: Update Results Analysis

Create `scripts/analyze_zne_results.py` to compute:

```python
# Metrics to compute from results:

# 1. Mean absolute error (MAE) for each condition
mae_sim_noiseless = np.mean(np.abs(sim_noiseless - fci))
mae_sim_noisy = np.mean(np.abs(sim_noisy - fci))
mae_hw_raw = np.mean(np.abs(hw_raw - fci))
mae_hw_zne = np.mean(np.abs(hw_zne - fci))

# 2. ZNE improvement ratio
zne_improvement = mae_hw_raw / mae_hw_zne  # >1 means ZNE helped

# 3. Noise model accuracy
noise_model_error = np.mean(np.abs(sim_noisy - hw_raw))

# 4. Equilibrium geometry error
eq_idx = np.argmin(fci)
eq_error_raw = abs(hw_raw[eq_idx] - fci[eq_idx])
eq_error_zne = abs(hw_zne[eq_idx] - fci[eq_idx])
```

---

## Key Code Locations

### VQE with Noise Support

```python
# src/h2_vqe/vqe.py - run_vqe() signature
def run_vqe(
    mol_data: MolecularData,
    ansatz_type: str = "uccsd",
    optimizer: str = "COBYLA",
    maxiter: int = 200,
    initial_params: Optional[np.ndarray] = None,
    backend: Literal["statevector", "aer_simulator"] = "statevector",
    shots: int = 1024,
    noise_model: Optional[NoiseModel] = None,  # <-- Added
    **ansatz_kwargs,
) -> VQEResult:
```

### IBM Runtime Integration

```python
# src/h2_vqe/ibm_runtime.py - key function
def run_vqe_on_hardware(
    mol_data,
    ansatz_type: str = "noise_aware",
    backend_name: Optional[str] = None,
    service: Optional["QiskitRuntimeService"] = None,
    optimizer: str = "COBYLA",
    maxiter: int = 50,
    shots: int = 4096,
    optimization_level: int = 3,
    resilience_level: int = 1,  # <-- 0=raw, 1=M3, 2=ZNE
) -> HardwareResult:
```

### Noise Model Presets

```python
# src/h2_vqe/noise.py - available presets
NOISE_PRESETS = {
    "ideal": NoiseParameters(single_qubit_error=0.0, two_qubit_error=0.0, ...),
    "ibm_like": NoiseParameters(single_qubit_error=5e-4, two_qubit_error=1e-2, ...),
    "low_noise": NoiseParameters(single_qubit_error=1e-4, two_qubit_error=5e-3, ...),
    "high_noise": NoiseParameters(single_qubit_error=1e-3, two_qubit_error=5e-2, ...),
}
```

---

## Project Structure

```
h2-vqe-dissociation/
├── src/h2_vqe/
│   ├── __init__.py
│   ├── molecular.py      # PySCF interface
│   ├── hamiltonian.py    # Jordan-Wigner transformation
│   ├── ansatz.py         # UCCSD, HW-Efficient, Noise-Aware
│   ├── vqe.py            # VQE engine (supports noise_model)
│   ├── noise.py          # IBM-like noise models
│   ├── dissociation.py   # Dissociation curve computation
│   ├── visualization.py  # Plotting (heatmaps, curves)
│   └── ibm_runtime.py    # IBM Quantum hardware integration
├── scripts/
│   ├── run_noise_experiment.py    # Simulator noise study
│   └── run_hardware_experiment.py # IBM hardware runs
├── tests/                # 119 tests
├── results/              # Output directory (gitignored)
├── pyproject.toml        # Dependencies (including [ibm] optional)
└── CONTINUATION.md       # This file
```

---

## Dependencies

```toml
# pyproject.toml
dependencies = [
    "qiskit>=1.0,<2.0",
    "qiskit-aer>=0.13",
    "qiskit-nature>=0.7",
    "pyscf>=2.4",
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
ibm = ["qiskit-ibm-runtime>=0.20"]
```

---

## IBM Quantum Setup

User has IBM Quantum account. To configure:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```

---

## Running Tests

```bash
cd /Users/sakeeb/Code\ repositories/h2-vqe-dissociation
source .venv/bin/activate
python -m pytest tests/ -v
```

---

## Quick Commands

```bash
# Run simulator noise experiment (quick test)
python scripts/run_noise_experiment.py --quick

# List available IBM backends
python scripts/run_hardware_experiment.py --list-backends

# Dry run hardware experiment
python scripts/run_hardware_experiment.py --dry-run

# Run on hardware (requires IBM credentials)
python scripts/run_hardware_experiment.py --ansatz noise_aware
```

---

## Expected Deliverables

1. **Data**: `results/zne_benchmark.json` with all energies
2. **Figure**: Comparison plot showing ZNE effectiveness
3. **Table**: Summary statistics (MAE, improvement ratios)
4. **Analysis**: Does our noise model predict hardware performance?

---

## Notes

- Use `noise_aware` ansatz only for hardware (2 CNOTs, most NISQ-friendly)
- 5 bond lengths is sufficient: [0.5, 0.74, 1.0, 1.5, 2.0] Å
- Queue times vary - 127-qubit devices often have better availability
- ZNE (resilience_level=2) requires ~3x more circuits than raw
