# H2-VQE Project Continuation Document

> **Purpose**: This document contains all context needed to continue development on a new Claude instance.

---

## Project Overview

**Repository**: `/Users/sakeeb/Code repositories/h2-vqe-dissociation`
**GitHub**: `https://github.com/Sakeeb91/h2-vqe-dissociation`

A Variational Quantum Eigensolver (VQE) implementation for computing H₂ molecular dissociation curves, with noise simulation, IBM Quantum hardware integration, and ZNE benchmarking support.

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
| Visualization (heatmaps, curves, ZNE) | `src/h2_vqe/visualization.py` | Complete |
| IBM Runtime integration | `src/h2_vqe/ibm_runtime.py` | Complete |
| **ZNE benchmark script** | `scripts/run_zne_benchmark.py` | **Complete** |
| **ZNE analysis script** | `scripts/analyze_zne_results.py` | **Complete** |
| Tests | `tests/` | 161 tests passing |

### ZNE Benchmarking Study - IMPLEMENTED ✅

The ZNE benchmarking study is now fully implemented:

| Component | Status |
|-----------|--------|
| `run_vqe_resilience_sweep()` | ✅ Added to ibm_runtime.py |
| `run_zne_benchmark.py` | ✅ Full benchmarking script |
| `plot_zne_comparison()` | ✅ 3-panel publication figure |
| `analyze_zne_results.py` | ✅ Metrics + LaTeX + CSV export |
| Tests | ✅ 42 new tests (161 total) |
| Examples | ✅ Quick demo + sample data |

---

## Quick Start Commands

### Run Simulator-Only ZNE Benchmark
```bash
# Quick demo (~30 seconds)
python examples/quick_zne_demo.py

# Full simulator benchmark
python scripts/run_zne_benchmark.py --simulator-only

# Dry run (show what would run)
python scripts/run_zne_benchmark.py --dry-run
```

### Analyze Results
```bash
# Analyze sample results
python scripts/analyze_zne_results.py examples/sample_zne_results.json

# Generate LaTeX table
python scripts/analyze_zne_results.py results/zne_benchmark.json --latex

# Export to CSV
python scripts/analyze_zne_results.py results/zne_benchmark.json --csv results/data.csv

# Generate plot
python scripts/analyze_zne_results.py results/zne_benchmark.json --plot
```

### Run Hardware Benchmark (requires IBM credentials)
```bash
# Full benchmark (5 bond lengths, raw + ZNE)
python scripts/run_zne_benchmark.py

# Resume with hardware after simulator run
python scripts/run_zne_benchmark.py --resume results/zne_benchmark.json --hardware-only
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
│   ├── visualization.py  # Plotting (heatmaps, curves, ZNE)
│   └── ibm_runtime.py    # IBM Quantum hardware + ZNE
├── scripts/
│   ├── run_noise_experiment.py    # Simulator noise study
│   ├── run_hardware_experiment.py # IBM hardware runs
│   ├── run_zne_benchmark.py       # ZNE benchmarking study
│   ├── analyze_zne_results.py     # Results analysis
│   └── test_zne_scripts.py        # Integration tests
├── examples/
│   ├── quick_zne_demo.py          # Quick simulator demo
│   └── sample_zne_results.json    # Sample data for testing
├── tests/                         # 161 tests
├── results/                       # Output directory (gitignored)
├── pyproject.toml                 # Dependencies
└── CONTINUATION.md                # This file
```

---

## Key Functions

### ZNE Resilience Sweep
```python
from h2_vqe.ibm_runtime import run_vqe_resilience_sweep
from h2_vqe.molecular import compute_h2_integrals

mol_data = compute_h2_integrals(0.74)
results = run_vqe_resilience_sweep(
    mol_data,
    resilience_levels=[0, 1, 2],  # raw, M3, ZNE
    ansatz_type="noise_aware",
)
# results[0] = raw, results[2] = ZNE
```

### ZNE Comparison Plot
```python
from h2_vqe.visualization import plot_zne_comparison

fig = plot_zne_comparison(
    bond_lengths=np.array([0.5, 0.74, 1.0, 1.5, 2.0]),
    fci_energies=fci,
    sim_noiseless=sim_noiseless,
    sim_noisy=sim_noisy,
    hw_raw=hw_raw,    # Optional
    hw_zne=hw_zne,    # Optional
    save_path="zne_comparison.png",
)
```

---

## Running Tests

```bash
cd /Users/sakeeb/Code\ repositories/h2-vqe-dissociation
source .venv/bin/activate

# All tests
python -m pytest tests/ -v

# ZNE-specific tests
python -m pytest tests/test_zne_*.py -v

# Quick integration test
python scripts/test_zne_scripts.py
```

---

## Next Steps / Future Work

1. **Run actual hardware benchmark**: Execute on IBM device when queue times allow
2. **Additional analysis**: Add statistical error bars from shot noise
3. **More ansatze comparison**: Test hardware-efficient ansatz on hardware
4. **Documentation**: Write up results as blog post or paper

---

## Notes

- Use `noise_aware` ansatz for hardware (2 CNOTs, most NISQ-friendly)
- 5 bond lengths capture key physics: [0.5, 0.74, 1.0, 1.5, 2.0] Å
- ZNE (resilience_level=2) requires ~3x more circuits than raw
- 161 tests ensure code quality and prevent regressions
