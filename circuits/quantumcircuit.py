
# Copyright © 2023 University of Strasbourg. All Rights Reserved.
# See AUTHORS.md for the list of authors.
#

import numpy as np
from .gates import Gate


class CircuitGate:
    """
    Class representing a gate in a quantum circuit.

    Attributes:
    gate (Gate): The gate to apply.
    qubits (tuple of int): The qubits to apply the gate to.
    """
    def __init__(self, gate, *args):
        """
        Initializes a CircuitGate object.
        Args:
        gate (Gate): The gate to apply.
        qubits (tuple of int): The qubits to apply the gate to.

        Raises:
        TypeError: If gate is not a subclass of Gate or qubits is not a tuple.
        ValueError: If qubits contains less than 1 or more than 2 elements.
        """
        if not isinstance(gate, Gate):
            raise TypeError("gate must be a subclass of Gate")
        
        if len(args) != gate.num_qubits:
            raise ValueError("Wrong number of target qubits")
        
        self._gate = gate
        self._qubits = args
    
    @property
    def gate(self):
        return self._gate

    @gate.setter
    def gate(self, _):
        raise AttributeError("gate is a read-only attribute")

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, _):
        raise AttributeError("qubits is a read-only attribute")
    
    def __str__(self):
        return f"{self.gate.__class__.__name__}({self.qubits})"

    def __repr__(self):
        return str(self)

    def CircuitGate_inverse(self):
       return CircuitGate(self.gate.get_inverse(), self.qubits)
    

class Circuit:
    """
    Class representing a quantum circuit.

    Attributes:
    gates (list of CircuitGate): The gates in the circuit.
    """
    def __init__(self, gates=None):
        """
        Initializes a Circuit object.

        Args:
        gates (list of CircuitGate): The gates to apply in the circuit.

        Raises:
        TypeError: If gates is not a list of CircuitGate objects.
        """
        self.gates = []
        if gates is None:
            gates = []
        if not isinstance(gates, list):
            raise TypeError("gates must be a list of CircuitGate objects")
        for gate in gates:
            if not isinstance(gate, CircuitGate):
                if isinstance(gate, tuple) and len(gate) == 2:
                    if isinstance(gate[0], Gate):
                        if not isinstance(gate[1], tuple) and not isinstance(gate[1], int):
                            raise TypeError("qubits must be a tuple or int")
                        if isinstance(gate[1], int):
                            qubits = (gate[1],)
                        else:
                            qubits = gate[1]
                        if len(qubits) < 1 or len(qubits) > 2:
                            raise ValueError("qubits must contain 1 or 2 elements")
                        circuit_gate = CircuitGate(gate[0], qubits)
                        self.gates.append(circuit_gate)
                    else:
                        raise TypeError("gate must be a Gate object")
                else:
                    raise TypeError("gates must be a list of CircuitGate or (Gate, qubits) tuples")
            else:
                self.gates.append(gate)


    def add_gate(self, gate, *args):
        """
        Adds a gate to the end of the circuit.

        Args:
        gate (Gate or CircuitGate): The gate to add.
        qubits (tuple or int): The qubits to apply the gate to.

        Raises:
        TypeError: If gate is not a Gate or CircuitGate object or qubits is not a tuple or int.
        ValueError: If qubits contains less than 1 or more than 2 elements.
        """
        if not isinstance(gate, Gate):
            raise TypeError("acceps only a Gate")

        circuit_gate = CircuitGate(gate, *args)
        self.gates.append(circuit_gate)
        
    def  add_circuitgate(self, circuitgate):
        if not isinstance(circuitgate, CircuitGate):
            raise TypeError("accepts only a CircuitGate")
        
        self.gates.append(circuitgate)


    def remove_gate(self, index):
        """
        Removes a gate at a specific index from the circuit.

        Args:
        index (int): The index of the gate to remove.

        Raises:
        IndexError: If index is out of range.
        """
        del self.gates[index]


    def get_gate(self, index):
        """
        Get a gate at a specific index from the circuit.

        Args:
        index (int): The index of the gate to get.
        """
        return self.gates[index]


    def __len__(self):
        return len(self.gates)


    def __iter__(self):
        return iter(self.gates)


    def __getitem__(self, index):
        return self.gates[index]


    def __str__(self):
        """Generate a JSON string representation of the circuit."""
        qubits = set(qubit for gate in self.gates for qubit in gate.qubits)
        num_qubits = max(qubits) + 1
        # Initialize an empty matrix representing the circuit
        matrix = np.empty((num_qubits, len(self.gates)), dtype=object)
        matrix[:, :] = ' '
        # Fill in the matrix with the gates
        for i, gate in enumerate(self.gates):
            for qubit in gate.qubits:
                matrix[qubit, i] = str(gate.gate)
        # Generate the string representation of the circuit
        output = ''
        for row in matrix:
            output += '|'
            for entry in row:
                output += f'{entry:^3}|'
            output += '\n'

        return output
    
    
    def depth(self):
        time_steps = [[] for _ in range(len(self.gates))]
        for i, gate in enumerate(self.gates):
            time_steps[i] = list(gate.qubits)
        max_qubits = 0
        for qubits in time_steps:
            if qubits:
                max_qubits = max(max_qubits, len(qubits))
        depth = 0
        for qubits in time_steps:
            depth += max_qubits - len(qubits)
        return depth