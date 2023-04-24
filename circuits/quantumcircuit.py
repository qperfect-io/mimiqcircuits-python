
# Copyright © 2023 University of Strasbourg. All Rights Reserved.
# See AUTHORS.md for the list of authors.
#

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
        base = f'{self.gate}'
        targets = ', '.join(map(lambda q: f'q{q}', self.qubits))
        return base + ' @ ' + targets

    def __repr__(self):
        return str(self)

    def inverse(self):
        return CircuitGate(self.gate.inverse(), self.qubits)


class Circuit:
    """
    Class representing a quantum circuit.

    Attributes:
    gates (list of CircuitGate): The gates in the circuit.
    """

    def __init__(self, gates: list = []):
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
                raise TypeError(
                    "`gates` must be a list of CircuitGate objects")

        self.gates = gates

    def num_qubits(self):
        """
        Returns the number of qubits in the circuit.
        """
        n = -1
        for gate in self.gates:
            m = max(gate.qubits)
            if m > n:
                n = m

        return n+1

    def empty(self):
        """
        Checks if the circuit is empty.
        """
        self.gates.empty()

    def add_gate(self, gate: Gate, *args):
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

    def add_circuitgate(self, circuitgate):
        if not isinstance(circuitgate, CircuitGate):
            raise TypeError("accepts only a CircuitGate")

        self.gates.append(circuitgate)

    def remove_gate(self, index: int):
        """
        Removes a gate at a specific index from the circuit.

        Args:
        index (int): The index of the gate to remove.

        Raises:
        IndexError: If index is out of range.
        """
        del self.gates[index]

    def get_gate(self, index: int):
        """
        Get a gate at a specific index from the circuit.

        Args:
        index (int): The index of the gate to get.
        """
        return self.gates[index]

    def inverse(self):
        invgates = map(lambda x: x.inverse(), self.gates.reverse())
        return Circuit(invgates)

    def __len__(self):
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def __getitem__(self, index):
        return self.get_gate(index)

    def __str__(self):
        n = len(self)
        nq = self.num_qubits()
        output = f'{nq}-qubit circuit with {n} gates'

        # iterate from the second gate
        for g in self.gates[:-1]:
            output += f'\n ├── {g}'

        g = self.gates[-1]
        output += f'\n └── {g}'

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
