#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from mimiqcircuits.gates import Gate


class CircuitGate:
    """
    Class representing a gate in a quantum circuit.

    Attributes:
    gate (Gate): The gate to apply.
    qubits (tuple of int): The qubits to apply the gate to.
    """
    _gate = None
    _qubits = None

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
            raise ValueError(
                f"Wrong number of target qubits for gate {gate} wanted  {gate.num_qubits}, given {len(args)}")

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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def inverse(self):
        return CircuitGate(self.gate.inverse(), self.qubits)

    def to_json(self):
        d = self.gate.to_json()
        # in JSON files we count from 1 (Julia convention)
        d['targets'] = [t+1 for t in self.qubits]
        return d

    @staticmethod
    def from_json(d):
        qubits = tuple([t-1 for t in d['targets']])
        gate = Gate.from_json(d)
        return CircuitGate(gate, *qubits)


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

        if not isinstance(gates, list) and gates is not None:
            raise TypeError(
                "Circuit should be initialized with a list of CircuitGate")

        if gates is None:
            self.gates = []

        else:
            for gate in gates:
                if not isinstance(gate, CircuitGate):
                    raise TypeError(
                        "Non Gate object passed to constructor.")

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
        return len(self.gates) == 0

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
            raise TypeError(
                f"Acceps only a Gate. Given {gate} of type {type(gate)}")

        circuit_gate = CircuitGate(gate, *args)
        self.gates.append(circuit_gate)

    def append(self, circuit):
        """
        Appends all the gates of the given circuit at the end of the current circuit.
        """
        if not isinstance(circuit, Circuit):
            raise TypeError("accepts only a Circuit")

        for g in circuit.gates:
            self.add_circuitgate(g)

    def append_circuitgates(self, gates):
        """
        Appends the list of given circuit gates at the end of the current circuit.
        """
        if not isinstance(gates, list):
            raise TypeError("accepts only a list of CircuitGate")

        for g in gates:
            self.add_circuitgate(g)

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

        if n == 0:
            return 'empty circuit'

        nq = self.num_qubits()
        output = f'{nq}-qubit circuit with {n} gates'

        # iterate from the second gate
        for g in self.gates[:-1]:
            output += f'\n ├── {g}'

        g = self.gates[-1]
        output += f'\n └── {g}'

        return output

    def __eq__(self, other):
        return self.gates == other.gates

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

    def to_json(self):
        return {'gates': [g.to_json() for g in self.gates]}

    @staticmethod
    def from_json(d):
        return Circuit(
            [CircuitGate.from_json(g) for g in d['gates']]
        )


# export the cirucit classes
__all__ = ['CircuitGate', 'Circuit']
