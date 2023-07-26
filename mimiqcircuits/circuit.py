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
from mimiqcircuits.measure import Measure
from mimiqcircuits.reset import Reset
from mimiqcircuits.json_utils import operation_from_json
import copy
from collections.abc import Iterable
import itertools


def allunique(lst):
    seen = set()
    for element in lst:
        if element in seen:
            return False
        seen.add(element)
    return True


class Instruction:
    """
    Class representing an instruction of a quantum circuit.

    Attributes:
        operation (Operation): The operation applied by the instruction.
        qubits (tuple of int): The qubits to apply the operation to.
        bits (tuple of int): The classical bits to apply the operation to
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

        if len(qubits) != operation.num_qubits:
            raise ValueError(
                f"Wrong number of target qubits for operation {operation} wanted  {operation.num_qubits}, given {len(qubits)}")

        if len(bits) != operation.num_bits:
            raise ValueError(
                f"Wrong number of target bits for operation {operation} wanted  {operation.num_bits}, given {len(bits)}")

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
        if not isinstance(other, Instruction):
            return False
        return (self.operation == other.operation) and (self.qubits == other.qubits) and (self.bits == other.bits)

    def inverse(self):
        return Instruction(self.operation.inverse(), self.qubits, self.bits)

    def to_json(self):
        d = {}
        d['op'] = self.operation.to_json()
        # in JSON files we count from 1 (Julia convention)
        d['qtargets'] = [t+1 for t in self.qubits]
        d['ctargets'] = [t+1 for t in self.bits]
        return d

    @ staticmethod
    def from_json(d):
        operation = operation_from_json(d['op'])
        qubits = tuple([t-1 for t in d['qtargets']])
        bits = tuple([t-1 for t in d['ctargets']])
        return Instruction(operation, qubits, bits)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


class Circuit:
    """
    Class representing a quantum circuit.

    Attributes:
        instructions (list of Instruction): The instructions in the circuit.
    """

    def __init__(self, instructions=None):
        """
        Initializes a Circuit object.

        Args:
            instructions (list of Instruction): The instructiuons to add at construction to the circuit.

        Raises:
            TypeError: If  is not a list of Instruction objects.
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

    def num_qubits(self):
        """
        Returns the number of qubits in the circuit.
        """
        n = -1
        for instruction in self.instructions:
            qubits = instruction.qubits
            if len(qubits) == 0:
                continue
            m = max(qubits)
            if m > n:
                n = m

        return n+1

    def num_bits(self):
        """
        Returns the number of qubits in the circuit.
        """
        n = -1
        for instruction in self.instructions:
            bits = instruction.bits
            if len(bits) == 0:
                continue
            m = max(bits)
            if m > n:
                n = m

        return n+1

    def empty(self):
        """
        Checks if the circuit is empty.
        """
        return len(self.instructions) == 0

    def push(self, operation, *args):
        """
        Adds an Operation or an Instruction to the end of the circuit.

        Args:
            operation (Operation or Instruction): the quantum operation to add.
            *args (integers or ranges): Target qubits and bits for the operation (not instruction), given as variable number of arguments.

        Raises:
            TypeError: If operation is not an Operation object or the arguments are invalid.
            ValueError: If the number of arguments is incorrect or qubits contain less than 1 or more than the operation's number of qubits.
        """
        N = 0
        M = 0
        L = len(args)

        if isinstance(operation, Instruction):
            if L != 0:
                raise(ValueError("No extra arguments allowed when pushing an instruction."))

            self.instructions.append(operation)
            return self

        if operation == Barrier:
            N = L
        else:
            if not isinstance(operation, Operation):
                raise(TypeError("Non Operation object passed to push."))

            N = operation.num_qubits
            M = operation.num_bits

        if L != N + M:
            raise(ValueError(
                f"Wrong number of target qubits and bits, given {L} for a {N} qubits + {M} bits operation."
            ))

        target_ranges = []
        for arg in args:
            if isinstance(arg, int):
                target_ranges.append([arg])
            elif isinstance(arg, Iterable):
                target_ranges.append(list(arg))
            else:
                raise TypeError(f"Invalid target type: {type(arg)}")

        for i in range(N):
            for j in range(i+1, N):
                if not set(target_ranges[i]).isdisjoint(target_ranges[j]):
                    raise ValueError("Duplicated qubit target.")

        for i in range(M):
            for j in range(i+1, M):
                if not set(target_ranges[N+i]).isdisjoint(target_ranges[N+j]):
                    raise ValueError("Duplicated bit target.")

        for targets in itertools.product(*target_ranges):
            if operation == Barrier:
                self.instructions.append(Instruction(Barrier(N), (*targets,), ()))
            else:
                self.instructions.append(Instruction(operation, (*targets[:N],), (*targets[N:],)))

        return self

    def insert(self, index: int, operation, *args):
        """
        Inserts an operation at a specific index in the circuit.

        Args:
            index (int): The index at which the operation should be inserted.
            operation (Operation): The quantum operation to insert.
            *args: The target qubits or classical bits for the operation.

        Raises:
            TypeError: If operation is not an Operation object or the arguments are invalid.
            ValueError: If the number of arguments is incorrect or qubits contain less than 1 or more than the operation's number of qubits.
        """
        N = 0
        M = 0
        L = len(args)

        if isinstance(operation, Instruction):
            if L != 0:
                raise(ValueError("No extra arguments allowed when inserting an instruction."))

            self.instructions.insert(index, operation)
            return self

        if operation == Barrier:
            N = L
        else:
            if not isinstance(operation, Operation):
                raise(TypeError("Non Operation object passed to push."))

            N = operation.num_qubits()
            M = operation.num_bits()

        if L != N + M:
            raise(ValueError(
                f"Wrong number of target qubits and bits, given {L} for a {N} qubits + {M} bits operation."
            ))

        if operation == Barrier:
            self.instructions.insert(index, Instruction(operation, args))
        else:
            self.instructions.insert(index, Instruction(Barrier(N), args))

    def append(self, other):
        """
        Appends all the gates of the given circuit at the end of the current circuit.

        Args:
            circuit (Circuit): the circuit to append.
        """
        instructions = None
        if isinstance(other, Circuit):
            instructions = other.instructions
        elif isinstance(other, list):
            instructions = other
        else:
            raise TypeError("Only allowed to append a circuit or a list of instructions")

        for inst in instructions:
            self.instructions.append(inst)

    def remove(self, index: int):
        """
        Removes an instruction at a specific index from the circuit.

        Args:
            index (int): The index of the gate to remove.

        Raises:
            IndexError: If index is out of range.
        """
        del self.instructions[index]

    def inverse(self):
        """
        Returns the inverse of the circuit.
        """
        invgates = [x.inverse() for x in self.instructions]
        invgates.reverse()
        return Circuit(invgates)

    def __len__(self):
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)

    def __getitem__(self, index):
        if type(index) is slice:
            return Circuit(self.instructions[index])
        return self.instructions[index]

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

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Circuit):
            return False
        for (i, j) in zip(self.instructions, other.instructions):
            if i != j:
                return False

        return True

    def depth(self):
        """
        Computes the depth of the quantum circuit.
        """
        if self.empty() or self.num_qubits() == 0:
            return 0

        d = [0 for _ in range(self.num_qubits() + self.num_bits())]

        for g in self:
            if isinstance(g.operation, Barrier):
                continue
            nq = self.num_qubits()
            optargets = g.qubits + tuple(map(lambda x: x + nq, g.bits))
            dm = max([d[t] for t in optargets])
            for t in g.qubits:
                d[t] = dm + 1
            for t in g.bits:
                d[t + nq] = dm + 1

        return max(d)

    def to_json(self):
        return {'instructions': [g.to_json() for g in self.instructions]}

    @ staticmethod
    def from_json(d):
        return Circuit(
            [Instruction.from_json(g) for g in d['instructions']]
        )

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


# export the cirucit classes
__all__ = ['Instruction', 'Circuit']
