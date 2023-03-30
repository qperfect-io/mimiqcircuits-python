#
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
    def __init__(self, gate, qubits):
        """
        Initializes a CircuitGate object.

        Args:
        gate (Gate): The gate to apply.
        qubits (tuple of int): The qubits to apply the gate to.

        Raises:
        TypeError: If gate is not a subclass of Gate or qubits is not a tuple.
        ValueError: If qubits contains less than 1 or more than 2 elements.
        """
        if not issubclass(type(gate), Gate):
            raise TypeError("gate must be a subclass of Gate")
        if not isinstance(qubits, tuple):
            raise TypeError("qubits must be a tuple")
        if len(qubits) < 1 or len(qubits) > 2:
            raise ValueError("qubits must contain 1 or 2 elements")
        self._gate = gate
        self._qubits = qubits

    @property
    def gate(self):
        self._gate

    @gate.setter
    def gate(self, _):
        raise AttributeError("gate is a read-only attribute")

    @property
    def qubits(self):
        self._qubits

    @qubits.setter
    def qubits(self, _):
        raise AttributeError("qubits is a read-only attribute")


class Circuit:
    """
    Class representing a quantum circuit.

    Attributes:
    gates (list of CircuitGate): The gates in the circuit.
    """
    def __init__(self, gates):
        """
        Initializes a Circuit object.

        Args:
        gates (list of CircuitGate): The gates to apply in the circuit.

        Raises:
        TypeError: If gates is not a list of CircuitGate objects.
        """
        if not isinstance(gates, list):
            raise TypeError("gates must be a list of CircuitGate objects")
        for gate in gates:
            if not isinstance(gate, CircuitGate):
                raise TypeError("gates must be a list of CircuitGate objects")
        self.gates = gates

    def add_gate(self, gate):
        """
        Adds a gate to the end of the circuit.

        Args:
        gate (CircuitGate): The gate to add.

        Raises:
        TypeError: If gate is not a CircuitGate object.
        """
        if not isinstance(gate, CircuitGate):
            raise TypeError("gate must be a CircuitGate object")
        self.gates.append(gate)

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

