"""
Tests for ansatz.py - Variational circuit ansatze.

These tests verify:
1. Correct circuit construction
2. Parameter counts and binding
3. Circuit properties (CNOTs, depth)
4. Factory function functionality
"""

import pytest
import numpy as np

from qiskit import QuantumCircuit

from h2_vqe.ansatz import (
    UCCSDAnswatz,
    HardwareEfficientAnsatz,
    NoiseAwareAnsatz,
    create_ansatz,
    get_initial_parameters,
    AnsatzInfo,
    ANSATZ_REGISTRY,
)


class TestUCCSDAnswatz:
    """Tests for UCCSD ansatz."""

    def test_creates_circuit(self):
        """Should create a quantum circuit."""
        ansatz = UCCSDAnswatz(n_qubits=4)
        assert isinstance(ansatz.circuit, QuantumCircuit)

    def test_correct_qubit_count(self):
        """Circuit should have correct number of qubits."""
        ansatz = UCCSDAnswatz(n_qubits=4)
        assert ansatz.circuit.num_qubits == 4

    def test_has_one_parameter(self):
        """H2 UCCSD should have 1 parameter (double excitation)."""
        ansatz = UCCSDAnswatz(n_qubits=4)
        # For H2, only double excitation contributes
        assert ansatz.num_parameters == 1

    def test_prepares_hf_reference(self):
        """Circuit should start with HF reference state."""
        ansatz = UCCSDAnswatz(n_qubits=4)
        qc = ansatz.circuit

        # First two operations should be X gates on qubits 0 and 1
        ops = list(qc.data[:2])
        assert ops[0].operation.name == "x"
        assert ops[1].operation.name == "x"

    def test_parameter_binding_works(self):
        """Should be able to bind parameters."""
        ansatz = UCCSDAnswatz(n_qubits=4)
        bound = ansatz.bind_parameters(np.array([0.5]))
        assert bound.num_parameters == 0


class TestHardwareEfficientAnsatz:
    """Tests for hardware-efficient ansatz."""

    def test_creates_circuit(self):
        """Should create a quantum circuit."""
        ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=2)
        assert isinstance(ansatz.circuit, QuantumCircuit)

    def test_correct_parameter_count(self):
        """Should have 2 * n_qubits * n_layers parameters."""
        ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=2)
        expected = 2 * 4 * 2  # 16 parameters
        assert ansatz.num_parameters == expected

    def test_single_layer(self):
        """Single layer should have fewer parameters."""
        ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=1)
        expected = 2 * 4 * 1  # 8 parameters
        assert ansatz.num_parameters == expected

    def test_linear_entanglement(self):
        """Linear entanglement should have n-1 CNOTs per layer."""
        ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=1, entanglement="linear")
        info = ansatz.get_info()
        assert info.n_cnots == 3  # n_qubits - 1

    def test_circular_entanglement(self):
        """Circular entanglement should have n CNOTs per layer."""
        ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=1, entanglement="circular")
        info = ansatz.get_info()
        assert info.n_cnots == 4  # n_qubits

    def test_full_entanglement(self):
        """Full entanglement should have n(n-1)/2 CNOTs per layer."""
        ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=1, entanglement="full")
        info = ansatz.get_info()
        expected = 4 * 3 // 2  # 6 CNOTs
        assert info.n_cnots == expected


class TestNoiseAwareAnsatz:
    """Tests for noise-aware ansatz."""

    def test_creates_circuit(self):
        """Should create a quantum circuit."""
        ansatz = NoiseAwareAnsatz(n_qubits=4)
        assert isinstance(ansatz.circuit, QuantumCircuit)

    def test_has_four_parameters(self):
        """Should have 4 parameters for H2."""
        ansatz = NoiseAwareAnsatz(n_qubits=4)
        assert ansatz.num_parameters == 4

    def test_minimal_cnots(self):
        """Should have minimal CNOT count."""
        ansatz = NoiseAwareAnsatz(n_qubits=4)
        info = ansatz.get_info()
        # Only 2 CNOTs for correlation
        assert info.n_cnots == 2

    def test_shallow_depth(self):
        """Should have shallow circuit depth."""
        ansatz = NoiseAwareAnsatz(n_qubits=4)
        info = ansatz.get_info()
        assert info.depth <= 5


class TestCreateAnsatz:
    """Tests for the factory function."""

    def test_creates_uccsd(self):
        """Should create UCCSD ansatz."""
        ansatz = create_ansatz("uccsd", n_qubits=4)
        assert isinstance(ansatz, UCCSDAnswatz)

    def test_creates_hardware_efficient(self):
        """Should create hardware-efficient ansatz."""
        ansatz = create_ansatz("hardware_efficient", n_qubits=4)
        assert isinstance(ansatz, HardwareEfficientAnsatz)

    def test_creates_noise_aware(self):
        """Should create noise-aware ansatz."""
        ansatz = create_ansatz("noise_aware", n_qubits=4)
        assert isinstance(ansatz, NoiseAwareAnsatz)

    def test_alias_he(self):
        """'he' should be alias for hardware_efficient."""
        ansatz = create_ansatz("he", n_qubits=4)
        assert isinstance(ansatz, HardwareEfficientAnsatz)

    def test_alias_minimal(self):
        """'minimal' should be alias for noise_aware."""
        ansatz = create_ansatz("minimal", n_qubits=4)
        assert isinstance(ansatz, NoiseAwareAnsatz)

    def test_case_insensitive(self):
        """Should be case insensitive."""
        ansatz = create_ansatz("UCCSD", n_qubits=4)
        assert isinstance(ansatz, UCCSDAnswatz)

    def test_invalid_type_raises(self):
        """Unknown ansatz type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown ansatz type"):
            create_ansatz("unknown", n_qubits=4)

    def test_passes_kwargs(self):
        """Should pass kwargs to constructor."""
        ansatz = create_ansatz("hardware_efficient", n_qubits=4, n_layers=3)
        expected = 2 * 4 * 3  # 24 parameters
        assert ansatz.num_parameters == expected


class TestGetInitialParameters:
    """Tests for parameter initialization."""

    def test_zeros_strategy(self):
        """Zeros strategy should return all zeros."""
        ansatz = create_ansatz("uccsd", n_qubits=4)
        params = get_initial_parameters(ansatz, strategy="zeros")
        assert np.allclose(params, 0)

    def test_random_strategy(self):
        """Random strategy should return values in [-π, π]."""
        ansatz = create_ansatz("hardware_efficient", n_qubits=4)
        params = get_initial_parameters(ansatz, strategy="random")
        assert np.all(params >= -np.pi)
        assert np.all(params <= np.pi)

    def test_hf_strategy(self):
        """HF strategy should return small values."""
        ansatz = create_ansatz("noise_aware", n_qubits=4)
        params = get_initial_parameters(ansatz, strategy="hf")
        assert np.all(np.abs(params) < 0.2)

    def test_correct_length(self):
        """Should return correct number of parameters."""
        ansatz = create_ansatz("hardware_efficient", n_qubits=4, n_layers=2)
        params = get_initial_parameters(ansatz, strategy="zeros")
        assert len(params) == ansatz.num_parameters


class TestAnsatzInfo:
    """Tests for ansatz metadata."""

    def test_info_dataclass(self):
        """get_info should return AnsatzInfo."""
        ansatz = create_ansatz("uccsd", n_qubits=4)
        info = ansatz.get_info()
        assert isinstance(info, AnsatzInfo)

    def test_info_has_name(self):
        """Info should have ansatz name."""
        ansatz = create_ansatz("uccsd", n_qubits=4)
        info = ansatz.get_info()
        assert "UCCSD" in info.name

    def test_info_has_all_fields(self):
        """Info should have all expected fields."""
        ansatz = create_ansatz("noise_aware", n_qubits=4)
        info = ansatz.get_info()

        assert info.n_qubits == 4
        assert info.n_parameters > 0
        assert info.n_cnots >= 0
        assert info.depth > 0


class TestParameterBinding:
    """Tests for parameter binding functionality."""

    def test_binding_removes_parameters(self):
        """Bound circuit should have no free parameters."""
        ansatz = create_ansatz("uccsd", n_qubits=4)
        n_params = ansatz.num_parameters
        values = np.random.random(n_params)
        bound = ansatz.bind_parameters(values)
        assert bound.num_parameters == 0

    def test_wrong_param_count_raises(self):
        """Wrong parameter count should raise ValueError."""
        ansatz = create_ansatz("uccsd", n_qubits=4)
        with pytest.raises(ValueError, match="Expected"):
            ansatz.bind_parameters(np.array([0.1, 0.2, 0.3]))


class TestCircuitValidity:
    """Tests to ensure circuits are valid for simulation."""

    @pytest.mark.parametrize("ansatz_type", ["uccsd", "hardware_efficient", "noise_aware"])
    def test_circuit_is_valid(self, ansatz_type):
        """All ansatze should produce valid circuits."""
        ansatz = create_ansatz(ansatz_type, n_qubits=4)
        qc = ansatz.circuit

        # Should have correct qubit count
        assert qc.num_qubits == 4

        # Should be able to get depth (validates structure)
        assert qc.depth() > 0

        # Should be able to bind parameters
        n_params = ansatz.num_parameters
        bound = ansatz.bind_parameters(np.zeros(n_params))
        assert bound.num_parameters == 0

    @pytest.mark.parametrize("ansatz_type", ["uccsd", "hardware_efficient", "noise_aware"])
    def test_can_simulate_statevector(self, ansatz_type):
        """Should be able to simulate circuit with statevector."""
        from qiskit.quantum_info import Statevector

        ansatz = create_ansatz(ansatz_type, n_qubits=4)
        params = get_initial_parameters(ansatz, strategy="zeros")
        bound = ansatz.bind_parameters(params)

        # Simulate
        sv = Statevector.from_instruction(bound)

        # Should be normalized
        assert np.isclose(np.abs(sv.data).sum() ** 2 /
                          np.abs(sv.data).sum() ** 2, 1.0)
        assert np.isclose(np.linalg.norm(sv.data), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
