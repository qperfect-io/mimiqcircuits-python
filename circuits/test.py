import unittest
from .quantumcircuit import Circuit, CircuitGate
from .gates import GateX, GateY

class TestCircuit(unittest.TestCase):
    def test_add_gate(self):
        circuit = Circuit()
        gate = GateX()
        circuit.add_gate(gate, 0)
        self.assertEqual(circuit.gates[0].gate, gate)
        self.assertEqual(circuit.gates[0].qubits, (0,))
    
    def test_add_circuit_gate(self):
        circuit = Circuit()
        circuit_gate = CircuitGate(GateX(), (0,))
        circuit.add_gate(circuit_gate)
        self.assertEqual(circuit.gates[0], circuit_gate)
    
    def test_remove_gate(self):
        circuit = Circuit()
        gate1 = CircuitGate(GateX(), (0,))
        gate2 = CircuitGate(GateY(), (1,))
        circuit.add_gate(gate1)
        circuit.add_gate(gate2)
        circuit.remove_gate(0)
        self.assertEqual(len(circuit.gates), 1)
        self.assertEqual(circuit.gates[0], gate2)
    
    def test_get_gate(self):
        circuit = Circuit()
        gate1 = CircuitGate(GateX(), (0,))
        gate2 = CircuitGate(GateY(), (1,))
        circuit.add_gate(gate1)
        circuit.add_gate(gate2)
        self.assertEqual(circuit.get_gate(0), gate1)
        self.assertEqual(circuit.get_gate(1), gate2)
    
    def test_init_with_gates(self):
        gates = [CircuitGate(GateX(), (0,)), CircuitGate(GateY(), (1,))]
        circuit = Circuit(gates)
        self.assertEqual(circuit.gates, gates)
    
    def test_init_with_invalid_gates(self):
        with self.assertRaises(TypeError):
            Circuit([GateX(), (GateY(), (0,))])
    
    def test_init_with_invalid_gate(self):
        with self.assertRaises(TypeError):
            Circuit([1])
    
    def test_add_invalid_gate(self):
        circuit = Circuit()
        with self.assertRaises(TypeError):
            circuit.add_gate(1)
    
    def test_add_invalid_qubits(self):
        circuit = Circuit()
        with self.assertRaises(TypeError):
            circuit.add_gate(GateX(), [1])
        with self.assertRaises(ValueError):
            circuit.add_gate(GateX(), ())
            circuit.add_gate(GateX(), (1, 2, 3))

    def test_add_invalid_qubits_tuple(self):
        circuit = Circuit()
        with self.assertRaises(ValueError):
            circuit.add_gate(GateX(), ())
    
    def test_add_invalid_qubits_length(self):
        circuit = Circuit()
        with self.assertRaises(ValueError):
            circuit.add_gate(GateX(), (0, 1, 2))
    
    def test_add_invalid_circuit_gate(self):
        circuit = Circuit()
        with self.assertRaises(TypeError):
            circuit.add_gate((GateX(), (0,)))
    
    def get_inverse_circuit(self):
        inverse = Circuit()
        for gate in reversed(self.gates):
            inverse.add_gate(gate.get_inverse(), gate.qubits)
        return inverse
    def test_circuit_str(self):
        # Test generating a string representation of a circuit
        gates = [CircuitGate(GateX(), (0,)), CircuitGate(GateX(), (1,))]
        circuit = Circuit(gates)
        circuit_str = str(circuit)
        expected_str = '| X |   |\n|   | X |\n'
        self.assertEqual(circuit_str, expected_str)

if __name__ == '__main__':
    unittest.main()