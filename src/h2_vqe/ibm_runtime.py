"""
IBM Quantum Runtime Integration
===============================

This module provides integration with IBM Quantum hardware through
the Qiskit Runtime service. It enables running VQE experiments on
real quantum processors.

Prerequisites:
    - Install: pip install qiskit-ibm-runtime
    - Save credentials: QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

Example:
    >>> from h2_vqe.ibm_runtime import get_service, run_vqe_on_hardware
    >>> service = get_service()
    >>> result = run_vqe_on_hardware(mol_data, ansatz_type="noise_aware")
"""

from dataclasses import dataclass
from typing import Optional, Literal, List
import numpy as np

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2
    from qiskit_ibm_runtime.options import EstimatorOptions
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False


@dataclass
class HardwareResult:
    """Result from running VQE on IBM hardware."""
    energy: float
    parameters: np.ndarray
    n_iterations: int
    n_evaluations: int
    convergence: bool
    error: Optional[float] = None
    exact_energy: Optional[float] = None
    backend_name: str = ""
    job_ids: List[str] = None

    def __repr__(self) -> str:
        error_str = f", error={self.error:.2e}" if self.error else ""
        return (
            f"HardwareResult(energy={self.energy:.6f} Ha, "
            f"backend={self.backend_name}, "
            f"converged={self.convergence}{error_str})"
        )


def check_ibm_runtime() -> bool:
    """Check if qiskit-ibm-runtime is installed and configured."""
    if not IBM_RUNTIME_AVAILABLE:
        print("qiskit-ibm-runtime is not installed.")
        print("Install with: pip install qiskit-ibm-runtime")
        return False
    return True


def get_service(
    channel: Literal["ibm_quantum", "ibm_cloud"] = "ibm_quantum",
    instance: Optional[str] = None,
) -> "QiskitRuntimeService":
    """
    Get the IBM Quantum Runtime service.

    Args:
        channel: IBM Quantum channel ("ibm_quantum" for free tier)
        instance: Optional instance specification (e.g., "hub/group/project")

    Returns:
        QiskitRuntimeService instance

    Raises:
        RuntimeError: If qiskit-ibm-runtime is not installed
        Exception: If credentials are not configured

    Example:
        >>> service = get_service()
        >>> backends = service.backends()
        >>> print([b.name for b in backends])
    """
    if not check_ibm_runtime():
        raise RuntimeError("qiskit-ibm-runtime not available")

    # Try to load saved credentials
    try:
        if instance:
            service = QiskitRuntimeService(channel=channel, instance=instance)
        else:
            service = QiskitRuntimeService(channel=channel)
        return service
    except Exception as e:
        print(f"Error loading IBM Quantum credentials: {e}")
        print("\nTo save your credentials, run:")
        print('  from qiskit_ibm_runtime import QiskitRuntimeService')
        print('  QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")')
        print("\nGet your token from: https://quantum.ibm.com/account")
        raise


def get_least_busy_backend(
    service: "QiskitRuntimeService",
    min_qubits: int = 4,
    operational: bool = True,
    simulator: bool = False,
) -> str:
    """
    Find the least busy IBM Quantum backend with sufficient qubits.

    Args:
        service: QiskitRuntimeService instance
        min_qubits: Minimum number of qubits required
        operational: Only consider operational backends
        simulator: Include simulators (default: False for real hardware)

    Returns:
        Backend name string

    Example:
        >>> service = get_service()
        >>> backend = get_least_busy_backend(service)
        >>> print(f"Using backend: {backend}")
    """
    if not check_ibm_runtime():
        raise RuntimeError("qiskit-ibm-runtime not available")

    backends = service.backends(
        min_num_qubits=min_qubits,
        operational=operational,
        simulator=simulator,
    )

    if not backends:
        raise ValueError(f"No backends found with >= {min_qubits} qubits")

    # Sort by queue length (pending jobs)
    backends_with_jobs = []
    for backend in backends:
        try:
            status = backend.status()
            pending = status.pending_jobs
            backends_with_jobs.append((backend.name, pending))
        except Exception:
            continue

    if not backends_with_jobs:
        # Fall back to first available
        return backends[0].name

    # Return least busy
    backends_with_jobs.sort(key=lambda x: x[1])
    return backends_with_jobs[0][0]


def list_available_backends(
    service: "QiskitRuntimeService",
    min_qubits: int = 4,
) -> List[dict]:
    """
    List available IBM Quantum backends with their status.

    Args:
        service: QiskitRuntimeService instance
        min_qubits: Minimum number of qubits required

    Returns:
        List of dicts with backend info
    """
    if not check_ibm_runtime():
        raise RuntimeError("qiskit-ibm-runtime not available")

    backends = service.backends(min_num_qubits=min_qubits)
    result = []

    for backend in backends:
        try:
            status = backend.status()
            config = backend.configuration()
            result.append({
                "name": backend.name,
                "n_qubits": config.n_qubits,
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
                "simulator": config.simulator,
            })
        except Exception:
            continue

    return result


def run_vqe_on_hardware(
    mol_data,
    ansatz_type: str = "noise_aware",
    backend_name: Optional[str] = None,
    service: Optional["QiskitRuntimeService"] = None,
    optimizer: str = "COBYLA",
    maxiter: int = 50,
    shots: int = 4096,
    optimization_level: int = 3,
    resilience_level: int = 1,
) -> HardwareResult:
    """
    Run VQE on IBM Quantum hardware.

    This function executes VQE optimization on a real quantum processor
    using the Qiskit Runtime Estimator primitive.

    Args:
        mol_data: Molecular data from compute_h2_integrals
        ansatz_type: Type of ansatz (recommend "noise_aware" for NISQ)
        backend_name: Specific backend to use (None = least busy)
        service: QiskitRuntimeService (None = create new)
        optimizer: Classical optimizer method
        maxiter: Maximum optimizer iterations
        shots: Number of measurement shots
        optimization_level: Transpilation optimization (0-3, higher = more optimized)
        resilience_level: Error mitigation level (0-2, higher = more mitigation)

    Returns:
        HardwareResult with optimization results

    Example:
        >>> from h2_vqe.molecular import compute_h2_integrals
        >>> mol_data = compute_h2_integrals(0.74)
        >>> result = run_vqe_on_hardware(mol_data)
        >>> print(f"Hardware energy: {result.energy:.6f} Ha")
    """
    if not check_ibm_runtime():
        raise RuntimeError("qiskit-ibm-runtime not available")

    from scipy.optimize import minimize
    from h2_vqe.hamiltonian import build_qubit_hamiltonian
    from h2_vqe.ansatz import create_ansatz, get_initial_parameters

    # Get service
    if service is None:
        service = get_service()

    # Get backend
    if backend_name is None:
        backend_name = get_least_busy_backend(service)
        print(f"Using least busy backend: {backend_name}")

    backend = service.backend(backend_name)

    # Build Hamiltonian and ansatz
    qubit_ham = build_qubit_hamiltonian(mol_data)
    ansatz = create_ansatz(ansatz_type, n_qubits=mol_data.n_qubits)

    print(f"Running VQE on {backend_name}")
    print(f"  Ansatz: {ansatz_type} ({ansatz.num_parameters} parameters)")
    print(f"  Max iterations: {maxiter}")
    print(f"  Shots: {shots}")

    # Configure estimator options
    options = EstimatorOptions()
    options.default_shots = shots
    options.optimization_level = optimization_level
    options.resilience_level = resilience_level

    # Track optimization
    n_evaluations = [0]
    energy_history = []
    job_ids = []

    def cost_function(params):
        """Evaluate energy on hardware."""
        n_evaluations[0] += 1
        bound_circuit = ansatz.bind_parameters(params)

        with Session(service=service, backend=backend_name) as session:
            estimator = EstimatorV2(session=session, options=options)
            job = estimator.run([(bound_circuit, qubit_ham.operator)])
            job_ids.append(job.job_id())
            result = job.result()
            energy = float(result[0].data.evs)

        energy_history.append(energy)
        print(f"  Eval {n_evaluations[0]}: E = {energy:.6f} Ha")
        return energy

    # Initialize parameters
    initial_params = get_initial_parameters(ansatz, strategy="zeros")

    # Run optimization
    print("\nStarting optimization...")
    result = minimize(
        cost_function,
        initial_params,
        method=optimizer,
        options={"maxiter": maxiter, "disp": False},
    )

    final_energy = result.fun

    # Calculate error if exact energy available
    error = None
    if qubit_ham.fci_energy is not None:
        error = abs(final_energy - qubit_ham.fci_energy)

    print(f"\nOptimization complete!")
    print(f"  Final energy: {final_energy:.6f} Ha")
    if error:
        print(f"  Error: {error*1000:.2f} mHa")

    return HardwareResult(
        energy=final_energy,
        parameters=result.x,
        n_iterations=result.nit if hasattr(result, 'nit') else n_evaluations[0],
        n_evaluations=n_evaluations[0],
        convergence=result.success,
        error=error,
        exact_energy=qubit_ham.fci_energy,
        backend_name=backend_name,
        job_ids=job_ids,
    )


def run_vqe_resilience_sweep(
    mol_data,
    ansatz_type: str = "noise_aware",
    resilience_levels: List[int] = [0, 1, 2],
    backend_name: Optional[str] = None,
    service: Optional["QiskitRuntimeService"] = None,
    maxiter: int = 30,
    shots: int = 4096,
    optimization_level: int = 3,
) -> dict:
    """
    Run VQE at multiple error mitigation levels for comparison.

    This function enables benchmarking of IBM's built-in error mitigation
    strategies by running the same VQE optimization at different resilience
    levels and comparing results.

    Args:
        mol_data: Molecular data from compute_h2_integrals
        ansatz_type: Type of ansatz (recommend "noise_aware" for NISQ)
        resilience_levels: List of resilience levels to test
            0 = raw (no mitigation)
            1 = M3 readout error mitigation
            2 = ZNE (zero-noise extrapolation)
        backend_name: Specific backend to use (None = least busy)
        service: QiskitRuntimeService (None = create new)
        maxiter: Maximum optimizer iterations
        shots: Number of measurement shots
        optimization_level: Transpilation optimization (0-3)

    Returns:
        Dict mapping resilience_level -> HardwareResult

    Example:
        >>> from h2_vqe.molecular import compute_h2_integrals
        >>> mol_data = compute_h2_integrals(0.74)
        >>> results = run_vqe_resilience_sweep(mol_data, resilience_levels=[0, 2])
        >>> print(f"Raw: {results[0].energy:.6f} Ha")
        >>> print(f"ZNE: {results[2].energy:.6f} Ha")

    Notes:
        - ZNE (level 2) requires approximately 3x more circuit executions
        - Running all levels sequentially may take significant queue time
        - Consider using the same initial parameters for fair comparison
    """
    if not check_ibm_runtime():
        raise RuntimeError("qiskit-ibm-runtime not available")

    # Get service
    if service is None:
        service = get_service()

    # Get backend
    if backend_name is None:
        backend_name = get_least_busy_backend(service)
        print(f"Using least busy backend: {backend_name}")

    print(f"Running resilience sweep on {backend_name}")
    print(f"  Resilience levels: {resilience_levels}")
    print(f"  Ansatz: {ansatz_type}")
    print(f"  Max iterations: {maxiter}")
    print(f"  Shots: {shots}")
    print("-" * 50)

    results = {}

    for level in resilience_levels:
        level_name = {0: "raw", 1: "M3 (readout)", 2: "ZNE"}
        print(f"\n[{level_name.get(level, f'level {level}')}] Running resilience_level={level}...")

        result = run_vqe_on_hardware(
            mol_data,
            ansatz_type=ansatz_type,
            backend_name=backend_name,
            service=service,
            maxiter=maxiter,
            shots=shots,
            optimization_level=optimization_level,
            resilience_level=level,
        )

        results[level] = result
        print(f"  Energy: {result.energy:.6f} Ha")
        if result.error is not None:
            print(f"  Error:  {result.error * 1000:.2f} mHa")

    print("\n" + "=" * 50)
    print("Resilience sweep complete!")
    print("-" * 50)
    for level, result in results.items():
        level_name = {0: "raw", 1: "M3", 2: "ZNE"}
        error_str = f", error={result.error * 1000:.2f} mHa" if result.error else ""
        print(f"  Level {level} ({level_name.get(level, '?'):4s}): E={result.energy:.6f} Ha{error_str}")

    return results


def run_single_point_hardware(
    mol_data,
    parameters: np.ndarray,
    ansatz_type: str = "noise_aware",
    backend_name: Optional[str] = None,
    service: Optional["QiskitRuntimeService"] = None,
    shots: int = 8192,
) -> float:
    """
    Run a single energy evaluation on hardware with fixed parameters.

    Useful for validating optimized parameters or benchmarking.

    Args:
        mol_data: Molecular data
        parameters: Fixed parameter values
        ansatz_type: Type of ansatz
        backend_name: Backend to use
        service: QiskitRuntimeService
        shots: Number of measurement shots

    Returns:
        Energy in Hartrees
    """
    if not check_ibm_runtime():
        raise RuntimeError("qiskit-ibm-runtime not available")

    from h2_vqe.hamiltonian import build_qubit_hamiltonian
    from h2_vqe.ansatz import create_ansatz

    # Get service and backend
    if service is None:
        service = get_service()
    if backend_name is None:
        backend_name = get_least_busy_backend(service)

    # Build circuit
    qubit_ham = build_qubit_hamiltonian(mol_data)
    ansatz = create_ansatz(ansatz_type, n_qubits=mol_data.n_qubits)
    bound_circuit = ansatz.bind_parameters(parameters)

    # Configure and run
    options = EstimatorOptions()
    options.default_shots = shots
    options.optimization_level = 3
    options.resilience_level = 1

    with Session(service=service, backend=backend_name) as session:
        estimator = EstimatorV2(session=session, options=options)
        job = estimator.run([(bound_circuit, qubit_ham.operator)])
        result = job.result()
        energy = float(result[0].data.evs)

    return energy


if __name__ == "__main__":
    print("IBM Quantum Runtime Integration")
    print("=" * 40)

    if not check_ibm_runtime():
        print("\nInstall with: pip install h2-vqe[ibm]")
    else:
        print("\nqiskit-ibm-runtime is installed!")
        print("\nTo use IBM Quantum hardware:")
        print("1. Get your API token from https://quantum.ibm.com/account")
        print("2. Save credentials:")
        print('   QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")')
        print("3. Run VQE:")
        print("   from h2_vqe.ibm_runtime import run_vqe_on_hardware")
        print("   result = run_vqe_on_hardware(mol_data)")
