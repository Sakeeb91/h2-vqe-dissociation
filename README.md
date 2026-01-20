# H₂ Molecular Dissociation Curve via VQE

Compute the ground state energy of molecular hydrogen (H₂) across varying bond lengths using the Variational Quantum Eigensolver (VQE), with benchmarking against classical methods (Hartree-Fock, FCI) and noise analysis.

## Overview

This project implements a complete VQE workflow for studying the dissociation of the H₂ molecule:

1. **Molecular Integrals**: Using PySCF to compute one- and two-electron integrals
2. **Qubit Hamiltonian**: Jordan-Wigner transformation to map fermionic operators to qubits
3. **Variational Ansatze**: Three circuit ansatze with different complexity/accuracy tradeoffs
4. **VQE Optimization**: Classical-quantum hybrid optimization loop
5. **Noise Models**: IBM-like noise models for realistic NISQ simulation
6. **Visualization**: Publication-quality dissociation curve figures

## Installation

```bash
# Clone the repository
git clone https://github.com/username/h2-vqe-dissociation.git
cd h2-vqe-dissociation

# Create virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e ".[dev]"
```

## Quick Start

```python
from h2_vqe import compute_h2_integrals, run_vqe

# Compute molecular data at equilibrium bond length
mol_data = compute_h2_integrals(0.74)  # 0.74 Å
print(f"HF Energy:  {mol_data.hf_energy:.6f} Ha")
print(f"FCI Energy: {mol_data.fci_energy:.6f} Ha")

# Run VQE
result = run_vqe(mol_data, ansatz_type="uccsd")
print(f"VQE Energy: {result.energy:.6f} Ha")
print(f"Error:      {result.error*1000:.2f} mHa")
```

## Ansatz Types

| Ansatz | Parameters | CNOTs | Best For |
|--------|------------|-------|----------|
| `uccsd` | 1 | 22 | Chemical accuracy |
| `hardware_efficient` | 16 | 6 | Flexibility |
| `noise_aware` | 4 | 2 | NISQ devices |

## Full Dissociation Curve

```python
from h2_vqe import compute_dissociation_curve
from h2_vqe.visualization import create_dissociation_figure

# Compute curve at 20 bond lengths
results = compute_dissociation_curve(
    n_points=20,
    start=0.3,  # Å
    stop=2.5,   # Å
    ansatz_types=["uccsd", "noise_aware"],
)

# Create publication-quality figure
fig = create_dissociation_figure(results)
fig.savefig("h2_dissociation.png", dpi=300)
```

## Project Structure

```
h2-vqe-dissociation/
├── src/h2_vqe/
│   ├── molecular.py      # PySCF interface
│   ├── hamiltonian.py    # Jordan-Wigner mapping
│   ├── ansatz.py         # Variational circuits
│   ├── vqe.py            # VQE optimization
│   ├── noise.py          # Noise models
│   ├── dissociation.py   # Curve computation
│   └── visualization.py  # Plotting
├── tests/                # Comprehensive test suite
├── notebooks/            # Demo notebooks
└── results/              # Output data
```

## Key Results

At equilibrium (r = 0.74 Å):
- **Hartree-Fock Energy**: -1.117 Ha
- **FCI Energy (Exact)**: -1.137 Ha
- **VQE Energy (UCCSD)**: -1.117 to -1.137 Ha (depending on optimization)

The correlation energy (E_FCI - E_HF ≈ -20 mHa) represents the electron-electron interaction not captured by mean-field theory.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=h2_vqe --cov-report=html
```

## Requirements

- Python ≥3.10
- qiskit ≥1.0, <2.0
- qiskit-aer ≥0.13
- qiskit-nature ≥0.7
- pyscf ≥2.4
- numpy, scipy, matplotlib

## Theory Background

### Variational Quantum Eigensolver (VQE)

VQE finds the ground state energy by minimizing the expectation value:

```
E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
```

The **variational principle** guarantees: E_VQE ≥ E_exact

### Jordan-Wigner Transformation

Maps fermionic operators to Pauli strings:
- a†_j → (X_j - iY_j)/2 ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
- a_j → (X_j + iY_j)/2 ⊗ Z_{j-1} ⊗ ... ⊗ Z_0

For H₂ in STO-3G basis: 2 spatial orbitals → 4 spin orbitals → 4 qubits

## License

MIT License

## References

1. Peruzzo et al., "A variational eigenvalue solver on a quantum processor" (2014)
2. McArdle et al., "Quantum computational chemistry" (2020)
3. Kandala et al., "Hardware-efficient variational quantum eigensolver" (2017)
