import unittest
from .gates import GateX
from .quantumcircuit import Circuit, CircuitGate


class TestCircuit(unittest.TestCase):

    def test_circuit_init(self):
        # Test initializing a circuit with valid gates
        gates = [CircuitGate(GateX(), (0,)), CircuitGate(GateX(), (1,))]
        circuit = Circuit(gates)
        self.assertEqual(len(circuit), 2)

        # Test initializing a circuit with invalid gates
        with self.assertRaises(TypeError):
            circuit = Circuit([1, 2, 3])

    def test_add_gate(self):
        # Test adding a valid gate to a circuit
        circuit = Circuit([])
        gate = CircuitGate(GateX(), (0,))
        circuit.add_gate(gate)
        self.assertEqual(len(circuit), 1)

        # Test adding an invalid gate to a circuit
        with self.assertRaises(TypeError):
            circuit.add_gate(1)

    def test_remove_gate(self):
        # Test removing a valid gate from a circuit
        gates = [CircuitGate(GateX(), (0,)), CircuitGate(GateX(), (1,))]
        circuit = Circuit(gates)
        circuit.remove_gate(0)
        self.assertEqual(len(circuit), 1)

        # Test removing an invalid gate from a circuit
        with self.assertRaises(IndexError):
            circuit.remove_gate(5)

    def test_get_gate(self):
        # Test getting a valid gate from a circuit
        gates = [CircuitGate(GateX(), (0,)), CircuitGate(GateX(), (1,))]
        circuit = Circuit(gates)
        gate = circuit.get_gate(1)
        self.assertEqual(gate.qubits, (1,))
        self.assertIsInstance(gate.gate, GateX)

    def test_circuit_str(self):
        # Test generating a string representation of a circuit
        gates = [CircuitGate(GateX(), (0,)), CircuitGate(GateX(), (1,))]
        circuit = Circuit(gates)
        circuit_str = str(circuit)
        expected_str = '| X |   |\n|   | X |\n'
        self.assertEqual(circuit_str, expected_str)

if __name__ == '__main__':
    unittest.main()
