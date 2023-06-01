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

from mimiqcircuits.operation import Operation
from mimiqcircuits.gates import Gate
from mimiqcircuits.barrier import Barrier
from mimiqcircuits.json_utils import operation_from_json


def allunique(lst):
    seen = set()
    for element in lst:
        if element in seen:
            return False
        seen.add(element)
    return True


class Instruction:
    """
    Class representing a gate in a quantum circuit.

    Attributes:
    gate (Gate): The gate to apply.
    qubits (tuple of int): The qubits to apply the gate to.
    """
    _operation = None
    _qubits = None
    _bits = None

    def __init__(self, operation, qubits=None, bits=None):
        """
        Initializes a Instruction object.
        Args:
        operation (Operation): The operation to apply.
        qubits (tuple of int): The qubits to apply the quantum operation to.
        bits (tuple of int): The classical bits to apply the quantum operation to.

        Raises:
        TypeError: If operation is not a subclass of Gate or qubits is not a tuple.
        ValueError: If qubits contains less than 1 or more than 2 elements.
        """
        if qubits is None:
            qubits = tuple()

        if bits is None:
            bits = tuple()

        if (not isinstance(qubits, tuple)):
            raise TypeError(
                f"Target qubits should be given in a tuple of integers. Given {qubits} of type {type(qubits)}.")

        if (not isinstance(bits, tuple)):
            raise TypeError(
                f"Target bits should be given in a tuple of integers. Given {bits} of type {type(bits)}.")

        if not isinstance(operation, Operation):
            raise TypeError(
                f"Operation must be a subclass of Operation. Given {operation} of type f{type(operation)}")

        if not allunique(qubits):
            raise ValueError("Duplicated qubit target in instruction")

        if not allunique(bits):
            raise ValueError("Duplicated classical bit target in instruction")

        for qi in qubits:
            if qi < 0:
                raise ValueError("Qubit target index cannot be negative")

        for bi in bits:
            if bi < 0:
                raise ValueError("Bit target index cannot be negative")

        if isinstance(operation, Gate) and len(qubits) != operation.num_qubits:
            raise ValueError(
                f"Wrong number of target qubits for gate {operation} wanted  {operation.num_qubits}, given {len(qubits)}")

        if isinstance(operation, Gate) and len(bits) != 0:
            raise ValueError(f"A gate cannot target classical bits.")

        self._operation = operation
        self._qubits = qubits
        self._bits = bits

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, _):
        raise AttributeError("operation is a read-only attribute")

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, _):
        raise AttributeError("qubits is a read-only attribute")

    @property
    def bits(self):
        return self._bits

    @bits.setter
    def bits(self, _):
        raise AttributeError("bits is a read-only attribute")

    def __str__(self):
        base = f'{self.operation}'
        qtargets = ', '.join(map(lambda q: f'q{q}', self.qubits))
        ctargets = ', '.join(map(lambda q: f'c{q}', self.bits))
        targets = ', '.join(filter(lambda x: x != '', [qtargets, ctargets]))
        return base + ' @ ' + targets

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def inverse(self):
        return Instruction(self.operation.inverse(), self.qubits, self.bits)

    def to_json(self):
        d = self.operation.to_json()
        # in JSON files we count from 1 (Julia convention)
        d['qtargets'] = [t+1 for t in self.qubits]
        d['ctargets'] = [t+1 for t in self.bits]
        return d

    @ staticmethod
    def from_json(d):
        qubits = tuple([t-1 for t in d['qtargets']])
        bits = tuple([t-1 for t in d['ctargets']])
        operation = operation_from_json(d)
        return Instruction(operation, qubits, bits)


class Circuit:
    """
    Class representing a quantum circuit.

    Attributes:
    gates (list of Instruction): The gates in the circuit.
    """

    def __init__(self, instructions=None):
        """
        Initializes a Circuit object.

        Args:
        gates (list of Instruction): The gates to apply in the circuit.

        Raises:
        TypeError: If gates is not a list of Instruction objects.
        """

        if instructions is None:
            instructions = []

        if not isinstance(instructions, list):
            raise TypeError(
                "Circuit should be initialized with a list of Instruction")

        for instruction in instructions:
            if not isinstance(instruction, Instruction):
                raise TypeError(
                    "Non Gate object passed to constructor.")

        self.instructions = instructions

    def __repr__(self):
        return self.__str__()

    def num_qubits(self):
        """
        Returns the number of qubits in the circuit.
        """
        n = -1
        for instruction in self.instructions:
            m = max(instruction.qubits)
            if m > n:
                n = m

        return n+1

    def num_bits(self):
        """
        Returns the number of qubits in the circuit.
        """
        n = -1
        for instruction in self.instructions:
            m = max(instruction.bits)
            if m > n:
                n = m

        return n+1

    def empty(self):
        """
        Checks if the circuit is empty.
        """
        return len(self.instructions) == 0

    def add(self, operation, qargs=None, cargs=None):
        instruction = Instruction(operation, qargs, cargs)
        self.instructions.append(instruction)

    def add_barrier(self, *args):
        self.instructions.append(Instruction(Barrier(), args))

    def add_gate(self, gate: Gate, *args):
        """
        Adds a gate to the end of the circuit.

        Args:
        gate (Gate or Instruction): The gate to add.
        qubits (tuple or int): The qubits to apply the gate to.

        Raises:
        TypeError: If gate is not a Gate or Instruction object or qubits is not a tuple or int.
        ValueError: If qubits contains less than 1 or more than 2 elements.
        """
        if not isinstance(gate, Gate):
            raise TypeError(
                f"Acceps only a Gate. Given {gate} of type {type(gate)}")

        instruction = Instruction(gate, args)
        self.instructions.append(instruction)

    def append(self, circuit):
        """
        Appends all the gates of the given circuit at the end of the current circuit.
        """
        if not isinstance(circuit, Circuit):
            raise TypeError("accepts only a Circuit")

        for g in circuit.gates:
            self.add_circuitgate(g)

    def append_instructions(self, instructions):
        """
        Appends the list of given circuit gates at the end of the current circuit.
        """
        if not isinstance(instructions, list):
            raise TypeError("accepts only a list of Instruction")

        if len(instructions) != 0 and not isinstance(instructions[0], Instruction):
            raise TypeError("accepts only a list of Instruction")

        for g in instructions:
            self.add_instruction(g)

    def add_instruction(self, instruction):
        if not isinstance(instruction, Instruction):
            raise TypeError("accepts only a Instruction")

        self.instructions.append(instruction)

    def remove_gate(self, index: int):
        """
        Removes a gate at a specific index from the circuit.

        Args:
        index (int): The index of the gate to remove.

        Raises:
        IndexError: If index is out of range.
        """
        del self.instructions[index]

    def get_gate(self, index: int):
        """
        Get a gate at a specific index from the circuit.

        Args:
        index (int): The index of the gate to get.
        """
        return self.instructions[index]

    def inverse(self):
        invgates = map(lambda x: x.inverse(), self.instructions.reverse())
        return Circuit(invgates)

    def __len__(self):
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)

    def __getitem__(self, index):
        return self.get_gate(index)

    def __str__(self):
        n = len(self)

        if n == 0:
            return 'empty circuit'

        nq = self.num_qubits()
        output = f'{nq}-qubit circuit with {n} gates'

        # iterate from the second gate
        for g in self.instructions[:-1]:
            output += f'\n ├── {g}'

        g = self.instructions[-1]
        output += f'\n └── {g}'

        return output

    def __eq__(self, other):
        return self.instructions == other.instructions
    

    def depth(self):
        time_steps = [[] for _ in range(len(self.instructions))]
        for i, instruction in enumerate(self.instructions):
            time_steps[i] = list(instruction.qubits)

        max_qubits = 0
        for qubits, instruction in zip(time_steps, self.instructions):
            if qubits and not isinstance(instruction.operation, Barrier):
                max_qubits = max(max_qubits, len(qubits))

        depth = 0
        for qubits, instruction in zip(time_steps, self.instructions):
            if qubits and not isinstance(instruction.operation, Barrier):
                depth += max_qubits - len(qubits)

        return depth

    def to_json(self):
        return {'instructions': [g.to_json() for g in self.instructions]}

    @ staticmethod
    def from_json(d):
        return Circuit(
            [Instruction.from_json(g) for g in d['instructions']]
        )


# export the cirucit classes
__all__ = ['Instruction', 'Circuit']
