"""
Variational Ansatz Implementations for VQE
==========================================

This module implements three types of variational ansatze for the
Variational Quantum Eigensolver (VQE):

1. UCCSD (Unitary Coupled Cluster Singles and Doubles):
   - Chemically motivated, starts from Hartree-Fock reference
   - Compact for small molecules, but deep circuits

2. Hardware-Efficient Ansatz:
   - Alternating layers of single-qubit rotations and entanglers
   - Flexible but may have barren plateaus

3. Noise-Aware Ansatz:
   - Minimal CNOT count for noise resilience
   - Shallow circuit depth for NISQ devices

Example:
    >>> from h2_vqe.ansatz import create_ansatz
    >>> circuit = create_ansatz("uccsd", n_qubits=4)
    >>> print(f"Parameters: {circuit.num_parameters}")
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Literal
from dataclasses import dataclass
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


@dataclass
class AnsatzInfo:
    """
    Metadata about an ansatz circuit.

    Attributes:
        name: Ansatz type identifier
        n_qubits: Number of qubits
        n_parameters: Number of variational parameters
        n_cnots: Number of CNOT gates
        depth: Circuit depth
        description: Human-readable description
    """
    name: str
    n_qubits: int
    n_parameters: int
    n_cnots: int
    depth: int
    description: str


class BaseAnsatz(ABC):
    """Abstract base class for all ansatze."""

    def __init__(self, n_qubits: int):
        """
        Initialize ansatz.

        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
        self._circuit: Optional[QuantumCircuit] = None
        self._parameters: Optional[ParameterVector] = None

    @abstractmethod
    def build_circuit(self) -> QuantumCircuit:
        """Build and return the parameterized circuit."""
        pass

    @property
    def circuit(self) -> QuantumCircuit:
        """Get the circuit, building it if necessary."""
        if self._circuit is None:
            self._circuit = self.build_circuit()
        return self._circuit

    @property
    def num_parameters(self) -> int:
        """Number of variational parameters."""
        return self.circuit.num_parameters

    def get_info(self) -> AnsatzInfo:
        """Get ansatz metadata."""
        qc = self.circuit
        return AnsatzInfo(
            name=self.__class__.__name__,
            n_qubits=self.n_qubits,
            n_parameters=qc.num_parameters,
            n_cnots=qc.count_ops().get("cx", 0),
            depth=qc.depth(),
            description=self.__class__.__doc__ or "",
        )

    def bind_parameters(self, values: np.ndarray) -> QuantumCircuit:
        """
        Bind parameter values to create concrete circuit.

        Args:
            values: Array of parameter values

        Returns:
            Circuit with bound parameters
        """
        if len(values) != self.num_parameters:
            raise ValueError(
                f"Expected {self.num_parameters} parameters, got {len(values)}"
            )
        return self.circuit.assign_parameters(values)


class UCCSDAnswatz(BaseAnsatz):
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    This chemically-motivated ansatz applies the exponential of
    cluster operators to a Hartree-Fock reference state:

        |ψ(θ)⟩ = e^{T(θ) - T†(θ)} |HF⟩

    where T = T₁ + T₂ includes single and double excitations.

    For H₂ in STO-3G basis (4 qubits, 2 electrons), there are:
    - 0 single excitations (all spatial orbitals occupied)
    - 1 double excitation: |0α 0β⟩ → |1α 1β⟩

    The circuit uses Trotterization to approximate the exponential.
    """

    def __init__(self, n_qubits: int = 4, n_electrons: int = 2):
        """
        Initialize UCCSD ansatz.

        Args:
            n_qubits: Number of qubits (must be 4 for H₂/STO-3G)
            n_electrons: Number of electrons (2 for H₂)
        """
        super().__init__(n_qubits)
        self.n_electrons = n_electrons

    def build_circuit(self) -> QuantumCircuit:
        """
        Build UCCSD circuit for H₂.

        For H₂, the ansatz prepares HF reference and applies
        the double excitation operator.

        Returns:
            Parameterized quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)

        # Prepare Hartree-Fock reference |1100⟩
        # Qubits: 0=0α, 1=0β, 2=1α, 3=1β
        # HF occupation: 0α↑ 0β↑ (qubits 0,1 are |1⟩)
        qc.x(0)  # Occupy 0α
        qc.x(1)  # Occupy 0β

        # Double excitation: |0α 0β⟩ → |1α 1β⟩
        # Implemented as exp(-iθ(X₀Y₁X₂X₃ - Y₀X₁X₂X₃ + ...)/2)
        # Using fermionic excitation gate decomposition

        theta = Parameter("θ_double")

        # The double excitation can be decomposed into CNOT ladder
        # with rotations. Using the standard decomposition:
        self._add_double_excitation(qc, [0, 1, 2, 3], theta)

        self._parameters = ParameterVector("θ", 1)
        return qc

    def _add_double_excitation(
        self,
        qc: QuantumCircuit,
        qubits: List[int],
        theta: Parameter
    ) -> None:
        """
        Add double excitation operator to circuit.

        This implements exp(-iθ(a†₂a†₃a₁a₀ - h.c.)/2) using
        the standard decomposition into CNOT gates and rotations.

        Args:
            qc: Circuit to modify
            qubits: List of 4 qubit indices [i, j, k, l]
            theta: Rotation parameter
        """
        i, j, k, l = qubits

        # Decomposition following arXiv:1805.04340
        # This creates the fermionic double excitation

        qc.cx(k, l)
        qc.cx(i, k)
        qc.cx(j, i)
        qc.cx(l, j)

        # Ry rotation
        qc.ry(theta / 8, i)

        qc.cx(l, i)
        qc.ry(-theta / 8, i)
        qc.cx(k, i)
        qc.ry(theta / 8, i)
        qc.cx(l, i)
        qc.ry(-theta / 8, i)

        # Reverse the CNOT ladder
        qc.cx(l, j)
        qc.cx(j, i)
        qc.cx(i, k)
        qc.cx(k, l)

        # Additional layer for the imaginary part
        qc.cx(k, l)
        qc.cx(i, k)
        qc.cx(j, i)
        qc.cx(l, j)

        qc.rx(np.pi / 2, i)
        qc.ry(theta / 8, i)

        qc.cx(l, i)
        qc.ry(-theta / 8, i)
        qc.cx(k, i)
        qc.ry(theta / 8, i)
        qc.cx(l, i)
        qc.ry(-theta / 8, i)
        qc.rx(-np.pi / 2, i)

        qc.cx(l, j)
        qc.cx(j, i)
        qc.cx(i, k)
        qc.cx(k, l)


class HardwareEfficientAnsatz(BaseAnsatz):
    """
    Hardware-Efficient Ansatz with alternating rotation and entanglement layers.

    Structure:
        [Ry-Rz layer] - [CNOT entangling layer] - [Ry-Rz layer] - ...

    This ansatz is hardware-native and flexible, but may suffer from
    barren plateaus at deeper depths.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        entanglement: Literal["linear", "circular", "full"] = "linear"
    ):
        """
        Initialize hardware-efficient ansatz.

        Args:
            n_qubits: Number of qubits
            n_layers: Number of rotation-entanglement layers
            entanglement: Entanglement pattern
                - "linear": nearest-neighbor CNOTs
                - "circular": linear + last-to-first CNOT
                - "full": all-to-all CNOTs
        """
        super().__init__(n_qubits)
        self.n_layers = n_layers
        self.entanglement = entanglement

    def build_circuit(self) -> QuantumCircuit:
        """
        Build hardware-efficient circuit.

        Returns:
            Parameterized quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)

        # 2 parameters per qubit per layer (Ry, Rz)
        n_params = 2 * self.n_qubits * self.n_layers
        params = ParameterVector("θ", n_params)
        self._parameters = params

        param_idx = 0

        for layer in range(self.n_layers):
            # Rotation layer: Ry and Rz on each qubit
            for q in range(self.n_qubits):
                qc.ry(params[param_idx], q)
                param_idx += 1
                qc.rz(params[param_idx], q)
                param_idx += 1

            # Entanglement layer
            self._add_entanglement_layer(qc)

        return qc

    def _add_entanglement_layer(self, qc: QuantumCircuit) -> None:
        """Add CNOT entanglement based on pattern."""
        if self.entanglement == "linear":
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        elif self.entanglement == "circular":
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(self.n_qubits - 1, 0)
        elif self.entanglement == "full":
            for q1 in range(self.n_qubits):
                for q2 in range(q1 + 1, self.n_qubits):
                    qc.cx(q1, q2)


class NoiseAwareAnsatz(BaseAnsatz):
    """
    Noise-Aware Ansatz with minimal CNOT count.

    This ansatz minimizes the number of two-qubit gates to reduce
    noise accumulation on NISQ devices. It uses only essential
    entanglement for capturing correlation in H₂.

    Structure:
        HF reference → Single CNOT → Ry rotations

    For H₂, the dominant correlation is captured by a single
    parameter controlling the superposition of |1100⟩ and |0011⟩.
    """

    def __init__(self, n_qubits: int = 4):
        """
        Initialize noise-aware ansatz.

        Args:
            n_qubits: Number of qubits (4 for H₂/STO-3G)
        """
        super().__init__(n_qubits)

    def build_circuit(self) -> QuantumCircuit:
        """
        Build minimal-CNOT circuit for H₂.

        This implements a physically motivated ansatz that captures
        the essential electron correlation with minimal gates.

        Returns:
            Parameterized quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)

        # Parameters for controlled rotations
        params = ParameterVector("θ", 4)
        self._parameters = params

        # Initialize in superposition state
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        # Minimal entanglement
        qc.cx(0, 2)
        qc.cx(1, 3)

        # Final rotations for fine-tuning
        qc.ry(params[2], 2)
        qc.ry(params[3], 3)

        return qc


# Ansatz registry for factory function
ANSATZ_REGISTRY = {
    "uccsd": UCCSDAnswatz,
    "hardware_efficient": HardwareEfficientAnsatz,
    "he": HardwareEfficientAnsatz,  # Alias
    "noise_aware": NoiseAwareAnsatz,
    "minimal": NoiseAwareAnsatz,  # Alias
}


def create_ansatz(
    ansatz_type: str,
    n_qubits: int = 4,
    **kwargs
) -> BaseAnsatz:
    """
    Factory function to create ansatz by name.

    Args:
        ansatz_type: Type of ansatz ("uccsd", "hardware_efficient", "noise_aware")
        n_qubits: Number of qubits
        **kwargs: Additional arguments passed to ansatz constructor

    Returns:
        Initialized ansatz object

    Raises:
        ValueError: If ansatz_type is not recognized

    Example:
        >>> ansatz = create_ansatz("uccsd", n_qubits=4)
        >>> print(f"Parameters: {ansatz.num_parameters}")
        1
    """
    ansatz_type = ansatz_type.lower()

    if ansatz_type not in ANSATZ_REGISTRY:
        valid = list(set(ANSATZ_REGISTRY.keys()))
        raise ValueError(
            f"Unknown ansatz type: {ansatz_type}. Valid types: {valid}"
        )

    ansatz_class = ANSATZ_REGISTRY[ansatz_type]
    return ansatz_class(n_qubits=n_qubits, **kwargs)


def get_initial_parameters(
    ansatz: BaseAnsatz,
    strategy: Literal["zeros", "random", "hf"] = "zeros"
) -> np.ndarray:
    """
    Get initial parameter values for optimization.

    Args:
        ansatz: Ansatz object
        strategy: Initialization strategy
            - "zeros": All parameters set to 0
            - "random": Uniform random in [-π, π]
            - "hf": Hartree-Fock-like initialization

    Returns:
        Array of initial parameter values
    """
    n_params = ansatz.num_parameters

    if strategy == "zeros":
        return np.zeros(n_params)
    elif strategy == "random":
        return np.random.uniform(-np.pi, np.pi, n_params)
    elif strategy == "hf":
        # Small perturbation from HF reference
        return np.random.uniform(-0.1, 0.1, n_params)
    else:
        raise ValueError(f"Unknown initialization strategy: {strategy}")


if __name__ == "__main__":
    print("Ansatz Comparison for H₂ (4 qubits)\n")
    print("-" * 60)

    for name in ["uccsd", "hardware_efficient", "noise_aware"]:
        ansatz = create_ansatz(name, n_qubits=4)
        info = ansatz.get_info()

        print(f"\n{info.name}")
        print(f"  Parameters: {info.n_parameters}")
        print(f"  CNOTs:      {info.n_cnots}")
        print(f"  Depth:      {info.depth}")
        print(f"\n  Circuit:")
        print(ansatz.circuit.draw(output="text", fold=80))
