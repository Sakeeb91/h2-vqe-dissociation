"""
Noise Models for Realistic VQE Simulation
=========================================

This module provides realistic noise models for simulating VQE
on NISQ (Noisy Intermediate-Scale Quantum) devices. The noise models
are inspired by IBM Quantum hardware characteristics.

Noise sources modeled:
1. Single-qubit gate errors (depolarizing)
2. Two-qubit gate errors (depolarizing, typically 10x worse)
3. Readout errors (bit-flip during measurement)
4. T1/T2 decoherence (amplitude and phase damping)

Example:
    >>> from h2_vqe.noise import create_noise_model
    >>> noise_model = create_noise_model("ibm_like")
    >>> result = run_vqe(mol_data, noise_model=noise_model)
"""

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import (
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    pauli_error,
    ReadoutError,
)


@dataclass
class NoiseParameters:
    """
    Parameters defining a noise model.

    Attributes:
        single_qubit_error: Error rate for single-qubit gates (typical: 1e-4 to 1e-3)
        two_qubit_error: Error rate for two-qubit gates (typical: 1e-3 to 1e-2)
        readout_error: Probability of bit-flip during measurement (typical: 1e-3 to 5e-2)
        t1: T1 relaxation time in microseconds (typical: 50-200 μs)
        t2: T2 dephasing time in microseconds (typical: 20-100 μs)
        gate_time_1q: Single-qubit gate time in nanoseconds (typical: 35 ns)
        gate_time_2q: Two-qubit gate time in nanoseconds (typical: 300 ns)
    """
    single_qubit_error: float = 1e-4
    two_qubit_error: float = 1e-3
    readout_error: float = 1e-2
    t1: float = 100.0  # microseconds
    t2: float = 50.0   # microseconds
    gate_time_1q: float = 35.0   # nanoseconds
    gate_time_2q: float = 300.0  # nanoseconds

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "single_qubit_error": self.single_qubit_error,
            "two_qubit_error": self.two_qubit_error,
            "readout_error": self.readout_error,
            "t1": self.t1,
            "t2": self.t2,
            "gate_time_1q": self.gate_time_1q,
            "gate_time_2q": self.gate_time_2q,
        }


# Predefined noise parameter sets
NOISE_PRESETS = {
    "ideal": NoiseParameters(
        single_qubit_error=0.0,
        two_qubit_error=0.0,
        readout_error=0.0,
        t1=np.inf,
        t2=np.inf,
    ),
    "ibm_like": NoiseParameters(
        single_qubit_error=5e-4,
        two_qubit_error=1e-2,
        readout_error=2e-2,
        t1=100.0,
        t2=50.0,
    ),
    "low_noise": NoiseParameters(
        single_qubit_error=1e-4,
        two_qubit_error=5e-3,
        readout_error=1e-2,
        t1=150.0,
        t2=80.0,
    ),
    "high_noise": NoiseParameters(
        single_qubit_error=1e-3,
        two_qubit_error=5e-2,
        readout_error=5e-2,
        t1=50.0,
        t2=25.0,
    ),
}


def create_noise_model(
    preset: Literal["ideal", "ibm_like", "low_noise", "high_noise"] = "ibm_like",
    custom_params: Optional[NoiseParameters] = None,
) -> NoiseModel:
    """
    Create a Qiskit noise model.

    Args:
        preset: Predefined noise level
            - "ideal": No noise (for benchmarking)
            - "ibm_like": Typical IBM Quantum hardware noise
            - "low_noise": Better-than-average NISQ device
            - "high_noise": Worse-case NISQ device
        custom_params: Override preset with custom parameters

    Returns:
        Qiskit NoiseModel object

    Example:
        >>> noise_model = create_noise_model("ibm_like")
        >>> print(noise_model)
    """
    if custom_params is not None:
        params = custom_params
    elif preset in NOISE_PRESETS:
        params = NOISE_PRESETS[preset]
    else:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(NOISE_PRESETS.keys())}")

    # Handle ideal case
    if params.single_qubit_error == 0 and params.two_qubit_error == 0:
        return NoiseModel()

    noise_model = NoiseModel()

    # Single-qubit depolarizing errors
    if params.single_qubit_error > 0:
        error_1q = depolarizing_error(params.single_qubit_error, 1)
        # Apply to common single-qubit gates
        for gate in ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz"]:
            noise_model.add_all_qubit_quantum_error(error_1q, gate)

    # Two-qubit depolarizing errors
    if params.two_qubit_error > 0:
        error_2q = depolarizing_error(params.two_qubit_error, 2)
        # Apply to CNOT and other two-qubit gates
        for gate in ["cx", "cz", "swap", "ecr"]:
            noise_model.add_all_qubit_quantum_error(error_2q, gate)

    # Readout errors
    if params.readout_error > 0:
        # P(0|1) and P(1|0) measurement errors
        p01 = params.readout_error  # Prob of reading 0 when true state is 1
        p10 = params.readout_error  # Prob of reading 1 when true state is 0
        read_error = ReadoutError([[1 - p10, p10], [p01, 1 - p01]])
        noise_model.add_all_qubit_readout_error(read_error)

    return noise_model


def get_noise_info(noise_model: NoiseModel) -> dict:
    """
    Get information about a noise model.

    Args:
        noise_model: Qiskit NoiseModel

    Returns:
        Dictionary with noise model statistics
    """
    return {
        "basis_gates": list(noise_model.basis_gates),
        "noise_instructions": list(noise_model.noise_instructions),
        "has_readout_error": len(noise_model._local_readout_errors) > 0
                            or len(noise_model._default_readout_error) > 0,
        "num_quantum_errors": (
            len(noise_model._local_quantum_errors)
            + len(noise_model._default_quantum_errors)
        ),
    }


def estimate_error_rate(
    noise_params: NoiseParameters,
    n_single_gates: int,
    n_two_gates: int,
    n_measurements: int = 4,
) -> float:
    """
    Estimate total circuit error rate from noise parameters.

    Uses a simple multiplicative error model:
        P(success) ≈ (1 - e1)^n1 * (1 - e2)^n2 * (1 - em)^nm

    Args:
        noise_params: Noise parameters
        n_single_gates: Number of single-qubit gates
        n_two_gates: Number of two-qubit gates
        n_measurements: Number of measurement operations

    Returns:
        Estimated total error probability

    Example:
        >>> params = NOISE_PRESETS["ibm_like"]
        >>> error = estimate_error_rate(params, n_single_gates=20, n_two_gates=10)
        >>> print(f"Estimated error: {error:.1%}")
    """
    p_success_1q = (1 - noise_params.single_qubit_error) ** n_single_gates
    p_success_2q = (1 - noise_params.two_qubit_error) ** n_two_gates
    p_success_read = (1 - noise_params.readout_error) ** n_measurements

    p_success_total = p_success_1q * p_success_2q * p_success_read
    return 1 - p_success_total


def compare_noise_levels(n_qubits: int = 4) -> dict:
    """
    Compare error rates across noise presets for a typical VQE circuit.

    Args:
        n_qubits: Number of qubits in circuit

    Returns:
        Dictionary mapping preset names to estimated error rates
    """
    # Typical gate counts for H2 VQE with different ansatze
    circuit_specs = {
        "uccsd": {"n_single": 30, "n_two": 22},
        "hardware_efficient": {"n_single": 16, "n_two": 6},
        "noise_aware": {"n_single": 4, "n_two": 2},
    }

    results = {}
    for preset_name, params in NOISE_PRESETS.items():
        results[preset_name] = {}
        for ansatz_name, specs in circuit_specs.items():
            error = estimate_error_rate(
                params,
                specs["n_single"],
                specs["n_two"],
                n_qubits,
            )
            results[preset_name][ansatz_name] = error

    return results


def create_thermal_relaxation_model(
    t1: float,
    t2: float,
    gate_times: dict,
    n_qubits: int = 4,
) -> NoiseModel:
    """
    Create noise model with T1/T2 thermal relaxation.

    This models amplitude damping (T1) and phase damping (T2) that
    occur during gate execution due to interaction with the environment.

    Args:
        t1: T1 relaxation time in microseconds
        t2: T2 dephasing time in microseconds (must be <= 2*T1)
        gate_times: Dict mapping gate names to times in nanoseconds
        n_qubits: Number of qubits

    Returns:
        NoiseModel with thermal relaxation

    Notes:
        The T2 time models both energy relaxation (T1) and pure dephasing (T_phi):
        1/T2 = 1/(2*T1) + 1/T_phi

        Physical constraint: T2 <= 2*T1
    """
    from qiskit_aer.noise import thermal_relaxation_error

    # Validate T2 <= 2*T1
    if t2 > 2 * t1:
        raise ValueError(f"T2 ({t2}) must be <= 2*T1 ({2*t1})")

    noise_model = NoiseModel()

    # Convert times to same units (microseconds)
    t1_us = t1
    t2_us = t2

    for gate_name, gate_time_ns in gate_times.items():
        # Convert gate time to microseconds
        gate_time_us = gate_time_ns / 1000.0

        # Create thermal relaxation error
        error = thermal_relaxation_error(t1_us, t2_us, gate_time_us)

        # Determine if single or two-qubit gate
        if gate_name in ["cx", "cz", "swap", "ecr"]:
            # Two-qubit gate: apply error to both qubits
            error_2q = error.tensor(error)
            noise_model.add_all_qubit_quantum_error(error_2q, gate_name)
        else:
            # Single-qubit gate
            noise_model.add_all_qubit_quantum_error(error, gate_name)

    return noise_model


if __name__ == "__main__":
    print("Noise Model Comparison\n")
    print("=" * 60)

    # Compare noise presets
    comparison = compare_noise_levels()

    print("\nEstimated Error Rates by Preset and Ansatz:")
    print("-" * 60)
    print(f"{'Preset':<15} {'UCCSD':<15} {'HW-Efficient':<15} {'Noise-Aware':<15}")
    print("-" * 60)

    for preset, ansatz_errors in comparison.items():
        uccsd = ansatz_errors["uccsd"]
        he = ansatz_errors["hardware_efficient"]
        na = ansatz_errors["noise_aware"]
        print(f"{preset:<15} {uccsd:>12.1%}   {he:>12.1%}   {na:>12.1%}")

    print("\n" + "=" * 60)
    print("\nNoise Parameter Details:")
    for name, params in NOISE_PRESETS.items():
        print(f"\n{name}:")
        print(f"  1Q error: {params.single_qubit_error:.1e}")
        print(f"  2Q error: {params.two_qubit_error:.1e}")
        print(f"  Readout:  {params.readout_error:.1e}")
        print(f"  T1/T2:    {params.t1}/{params.t2} μs")
