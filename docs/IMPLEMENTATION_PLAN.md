# H₂ Molecular Dissociation Curve via VQE - Implementation Plan

## Project Summary
Compute ground state energy of molecular hydrogen (H₂) across varying bond lengths using the Variational Quantum Eigensolver (VQE), benchmarking against classical methods (Hartree-Fock, FCI) and analyzing noise effects.

**Repository:** `h2-dissociation-vqe`
**Expert Role:** Quantum Computing Engineer
**Target:** Junior developer with 6 months Python experience
**Budget:** $0 (free tools only)

---

## Implementation Status

| Phase | Module | Status | Tests |
|-------|--------|--------|-------|
| 1 | molecular.py | ✅ Complete | 23 tests |
| 2 | hamiltonian.py | ✅ Complete | 20 tests |
| 3 | ansatz.py | ✅ Complete | 38 tests |
| 4 | vqe.py | ✅ Complete | 26 tests |
| 5 | noise.py | ✅ Complete | Integrated |
| 6 | dissociation.py | ✅ Complete | Integration |
| 7 | visualization.py | ✅ Complete | Manual |

**Total Tests:** 117 passing

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        H₂ VQE DISSOCIATION CURVE                            │
└─────────────────────────────────────────────────────────────────────────────┘

  Bond Length (Å)
        │
        ▼
┌───────────────┐    one-/two-electron    ┌───────────────┐
│    PySCF      │ ──────integrals──────▶  │ Jordan-Wigner │
│  Molecular    │                         │   Transform   │
│  Integrals    │                         │               │
└───────────────┘                         └───────────────┘
                                                 │
                                          Pauli strings
                                                 │
                                                 ▼
┌───────────────┐     circuit      ┌───────────────────────┐
│    Ansatz     │ ◀───params────── │    VQE Optimizer      │
│   Circuit     │                  │                       │
│  (θ₁,θ₂,...)  │ ────energy────▶  │  min⟨ψ(θ)|H|ψ(θ)⟩    │
└───────────────┘                  └───────────────────────┘
        │                                    │
        ▼                             converged?
┌───────────────┐                           │
│   Quantum     │                           ▼
│   Backend     │                  ┌───────────────┐
│  (simulate)   │                  │  Ground State │
└───────────────┘                  │    Energy     │
                                   └───────────────┘
```

---

## Technology Stack

| Package | Purpose | Cost |
|---------|---------|------|
| qiskit >=1.0 | Quantum circuit construction | $0 |
| qiskit-aer >=0.13 | Quantum simulation (noise models) | $0 |
| qiskit-nature >=0.7 | Chemistry-quantum interface | $0 |
| pyscf >=2.4 | Classical quantum chemistry | $0 |
| numpy, scipy | Numerical computing, optimization | $0 |
| matplotlib | Visualization | $0 |

---

## Key Results

At equilibrium (r = 0.74 Å):
- **Hartree-Fock Energy**: -1.116759 Ha
- **FCI Energy (Exact)**: -1.137284 Ha
- **Correlation Energy**: -20.5 mHa

Ansatz Comparison:
| Ansatz | Parameters | CNOTs | Typical Error |
|--------|------------|-------|---------------|
| UCCSD | 1 | 22 | <50 mHa |
| Hardware-Efficient | 16 | 6 | Variable |
| Noise-Aware | 4 | 2 | <50 mHa |

---

## Usage

```python
from h2_vqe import compute_h2_integrals, run_vqe

# Single point calculation
mol_data = compute_h2_integrals(0.74)
result = run_vqe(mol_data, ansatz_type="uccsd")
print(f"VQE Energy: {result.energy:.6f} Ha")

# Full dissociation curve
from h2_vqe import compute_dissociation_curve
from h2_vqe.visualization import create_dissociation_figure

results = compute_dissociation_curve(n_points=20)
fig = create_dissociation_figure(results)
fig.savefig("h2_dissociation.png")
```
