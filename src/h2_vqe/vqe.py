"""
VQE Optimization Engine
=======================

This module implements the Variational Quantum Eigensolver (VQE) algorithm
for finding the ground state energy of molecular Hamiltonians.

VQE works by:
1. Preparing a parameterized quantum state |ψ(θ)⟩
2. Measuring the expectation value ⟨ψ(θ)|H|ψ(θ)⟩
3. Using classical optimization to minimize this expectation value

The variational principle guarantees: E_VQE ≥ E_exact

Example:
    >>> from h2_vqe.molecular import compute_h2_integrals
    >>> from h2_vqe.vqe import run_vqe
    >>> mol_data = compute_h2_integrals(0.74)
    >>> result = run_vqe(mol_data)
    >>> print(f"VQE Energy: {result.energy:.6f} Ha")
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Literal, List
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.primitives import StatevectorEstimator

from h2_vqe.molecular import MolecularData
from h2_vqe.hamiltonian import build_qubit_hamiltonian, QubitHamiltonian
from h2_vqe.ansatz import create_ansatz, get_initial_parameters, BaseAnsatz


@dataclass
class VQEResult:
    """
    Container for VQE optimization results.

    Attributes:
        energy: Final optimized energy in Hartrees
        parameters: Optimal parameter values
        n_iterations: Number of optimizer iterations
        n_evaluations: Number of energy evaluations
        convergence: Whether optimizer converged
        energy_history: Energy at each evaluation (if tracked)
        exact_energy: Exact ground state energy for comparison
        error: Absolute error |E_VQE - E_exact|
        ansatz_type: Type of ansatz used
    """
    energy: float
    parameters: np.ndarray
    n_iterations: int
    n_evaluations: int
    convergence: bool
    energy_history: List[float] = field(default_factory=list)
    exact_energy: Optional[float] = None
    error: Optional[float] = None
    ansatz_type: str = ""

    def __repr__(self) -> str:
        error_str = f", error={self.error:.2e}" if self.error else ""
        return (
            f"VQEResult(energy={self.energy:.6f} Ha, "
            f"converged={self.convergence}, "
            f"iterations={self.n_iterations}{error_str})"
        )


class VQEEngine:
    """
    Core VQE optimization engine.

    This class handles the quantum circuit execution and classical
    optimization loop for finding ground state energies.
    """

    def __init__(
        self,
        hamiltonian: QubitHamiltonian,
        ansatz: BaseAnsatz,
        backend: Literal["statevector", "aer_simulator"] = "statevector",
        shots: int = 1024,
        noise_model: Optional[NoiseModel] = None,
    ):
        """
        Initialize VQE engine.

        Args:
            hamiltonian: Qubit Hamiltonian to minimize
            ansatz: Variational ansatz circuit
            backend: Simulation backend
                - "statevector": Exact statevector simulation (noiseless)
                - "aer_simulator": Shot-based simulation (supports noise)
            shots: Number of measurement shots (for aer_simulator)
            noise_model: Optional Qiskit NoiseModel for realistic simulation
        """
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.backend_type = backend
        self.shots = shots
        self.noise_model = noise_model

        self._n_evaluations = 0
        self._energy_history: List[float] = []

        # Setup backend
        if backend == "statevector":
            self._estimator = StatevectorEstimator()
        else:
            # Create AerSimulator with optional noise model
            if noise_model is not None:
                self._simulator = AerSimulator(noise_model=noise_model)
            else:
                self._simulator = AerSimulator()

    def compute_energy(self, parameters: np.ndarray) -> float:
        """
        Compute expectation value ⟨ψ(θ)|H|ψ(θ)⟩.

        Args:
            parameters: Variational parameter values

        Returns:
            Energy expectation value in Hartrees
        """
        self._n_evaluations += 1

        # Bind parameters to circuit
        bound_circuit = self.ansatz.bind_parameters(parameters)

        if self.backend_type == "statevector":
            # Exact statevector computation
            energy = self._compute_statevector_energy(bound_circuit)
        else:
            # Shot-based estimation
            energy = self._compute_shot_energy(bound_circuit)

        self._energy_history.append(energy)
        return energy

    def _compute_statevector_energy(self, circuit: QuantumCircuit) -> float:
        """Compute energy using exact statevector."""
        # Get statevector
        sv = Statevector.from_instruction(circuit)

        # Compute expectation value
        ham_matrix = self.hamiltonian.operator.to_matrix()
        state = sv.data
        energy = np.real(state.conj() @ ham_matrix @ state)

        return float(energy)

    def _compute_shot_energy(self, circuit: QuantumCircuit) -> float:
        """Compute energy using shot-based measurement with optional noise."""
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator

        # Create Aer Estimator with the simulator (which may have noise)
        estimator = AerEstimator.from_backend(self._simulator)

        # Set default precision based on shots (precision ~ 1/sqrt(shots))
        default_precision = 1.0 / np.sqrt(self.shots)

        # Run estimation
        job = estimator.run([(circuit, self.hamiltonian.operator)], precision=default_precision)
        result = job.result()
        energy = result[0].data.evs

        return float(energy)

    def optimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        method: str = "COBYLA",
        maxiter: int = 200,
        tol: float = 1e-6,
        callback: Optional[Callable] = None,
    ) -> VQEResult:
        """
        Run VQE optimization.

        Args:
            initial_params: Initial parameter values (default: zeros)
            method: Scipy optimizer method (COBYLA, L-BFGS-B, SLSQP, etc.)
            maxiter: Maximum iterations
            tol: Convergence tolerance
            callback: Function called after each iteration

        Returns:
            VQEResult with optimization results
        """
        # Initialize parameters
        if initial_params is None:
            initial_params = get_initial_parameters(self.ansatz, strategy="zeros")

        # Reset counters
        self._n_evaluations = 0
        self._energy_history = []

        # Track iterations
        iteration_count = [0]

        def iteration_callback(params):
            iteration_count[0] += 1
            if callback:
                callback(params)

        # Run optimization
        result = minimize(
            self.compute_energy,
            initial_params,
            method=method,
            options={"maxiter": maxiter, "disp": False},
            tol=tol,
            callback=iteration_callback,
        )

        # Compute final energy (ensure we use optimized params)
        final_energy = self.compute_energy(result.x)

        # Calculate error if exact energy is available
        error = None
        if self.hamiltonian.fci_energy is not None:
            error = abs(final_energy - self.hamiltonian.fci_energy)

        return VQEResult(
            energy=final_energy,
            parameters=result.x,
            n_iterations=iteration_count[0],
            n_evaluations=self._n_evaluations,
            convergence=result.success,
            energy_history=self._energy_history.copy(),
            exact_energy=self.hamiltonian.fci_energy,
            error=error,
            ansatz_type=self.ansatz.__class__.__name__,
        )


def run_vqe(
    mol_data: MolecularData,
    ansatz_type: str = "uccsd",
    optimizer: str = "COBYLA",
    maxiter: int = 200,
    initial_params: Optional[np.ndarray] = None,
    backend: Literal["statevector", "aer_simulator"] = "statevector",
    shots: int = 1024,
    noise_model: Optional[NoiseModel] = None,
    **ansatz_kwargs,
) -> VQEResult:
    """
    Run VQE calculation for molecular system.

    This is the main entry point for VQE calculations. It handles
    building the Hamiltonian, creating the ansatz, and running
    the optimization.

    Args:
        mol_data: Molecular data from compute_h2_integrals
        ansatz_type: Type of ansatz ("uccsd", "hardware_efficient", "noise_aware")
        optimizer: Scipy optimizer method
        maxiter: Maximum optimizer iterations
        initial_params: Initial parameter values (default: zeros)
        backend: Simulation backend ("statevector" or "aer_simulator")
        shots: Number of measurement shots (for aer_simulator)
        noise_model: Optional NoiseModel for realistic simulation (requires aer_simulator backend)
        **ansatz_kwargs: Additional arguments for ansatz constructor

    Returns:
        VQEResult with optimization results

    Example:
        >>> data = compute_h2_integrals(0.74)
        >>> result = run_vqe(data, ansatz_type="uccsd")
        >>> print(f"Energy: {result.energy:.6f} Ha")
        >>> print(f"Error:  {result.error:.2e} Ha")

    Example with noise:
        >>> from h2_vqe.noise import create_noise_model
        >>> noise = create_noise_model("ibm_like")
        >>> result = run_vqe(data, backend="aer_simulator", noise_model=noise)
    """
    # If noise_model is provided, force aer_simulator backend
    if noise_model is not None and backend == "statevector":
        backend = "aer_simulator"

    # Build qubit Hamiltonian
    qubit_ham = build_qubit_hamiltonian(mol_data)

    # Create ansatz
    ansatz = create_ansatz(ansatz_type, n_qubits=mol_data.n_qubits, **ansatz_kwargs)

    # Create VQE engine
    engine = VQEEngine(
        hamiltonian=qubit_ham,
        ansatz=ansatz,
        backend=backend,
        shots=shots,
        noise_model=noise_model,
    )

    # Run optimization
    result = engine.optimize(
        initial_params=initial_params,
        method=optimizer,
        maxiter=maxiter,
    )

    return result


def run_vqe_multistart(
    mol_data: MolecularData,
    ansatz_type: str = "uccsd",
    n_starts: int = 5,
    noise_model: Optional[NoiseModel] = None,
    **kwargs,
) -> VQEResult:
    """
    Run VQE with multiple random initializations.

    This helps avoid local minima by trying multiple starting points
    and returning the best result. Especially useful with noisy simulations
    where optimization landscapes can be more complex.

    Args:
        mol_data: Molecular data
        ansatz_type: Type of ansatz
        n_starts: Number of random initializations
        noise_model: Optional NoiseModel for realistic simulation
        **kwargs: Additional arguments for run_vqe

    Returns:
        Best VQEResult across all starts
    """
    best_result = None

    # Create ansatz to get parameter count
    ansatz = create_ansatz(ansatz_type, n_qubits=mol_data.n_qubits)

    for i in range(n_starts):
        if i == 0:
            # First start: use zeros
            init_params = np.zeros(ansatz.num_parameters)
        else:
            # Random initialization
            init_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

        result = run_vqe(
            mol_data,
            ansatz_type=ansatz_type,
            initial_params=init_params,
            noise_model=noise_model,
            **kwargs,
        )

        if best_result is None or result.energy < best_result.energy:
            best_result = result

    return best_result


def verify_variational_principle(result: VQEResult, tol: float = 1e-8) -> bool:
    """
    Verify that VQE energy satisfies variational principle.

    The variational principle states: E_VQE >= E_exact

    Args:
        result: VQE result to verify
        tol: Numerical tolerance for comparison

    Returns:
        True if variational principle is satisfied
    """
    if result.exact_energy is None:
        raise ValueError("Exact energy not available in result")

    return result.energy >= result.exact_energy - tol


if __name__ == "__main__":
    from h2_vqe.molecular import compute_h2_integrals

    print("Running VQE for H₂ at 0.74 Å\n")

    # Compute molecular data
    mol_data = compute_h2_integrals(0.74)
    print(f"Classical reference energies:")
    print(f"  HF:  {mol_data.hf_energy:.6f} Ha")
    print(f"  FCI: {mol_data.fci_energy:.6f} Ha")

    # Run VQE with each ansatz
    for ansatz_type in ["uccsd", "hardware_efficient", "noise_aware"]:
        print(f"\n{ansatz_type.upper()} Ansatz:")
        result = run_vqe(mol_data, ansatz_type=ansatz_type)

        print(f"  Energy:      {result.energy:.6f} Ha")
        print(f"  Error:       {result.error:.2e} Ha")
        print(f"  Iterations:  {result.n_iterations}")
        print(f"  Evaluations: {result.n_evaluations}")
        print(f"  Converged:   {result.convergence}")
        print(f"  Variational: {verify_variational_principle(result)}")
