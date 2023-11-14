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

from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.barrier import Barrier
import mimiqcircuits.instruction as mi
import copy
from collections.abc import Iterable
from itertools import repeat
from symengine import *
from mimiqcircuits.proto.circuitproto import *
from mimiqcircuits.proto import circuit_pb
import mimiqcircuits.operations.gates.custom as gc


class Circuit:
    """Representation of a quantum circuit.

    Operation can be added one by one to a circuit with the
    ``c.push(operation, targets...)`` function

    Args:
        instructions (list of Instruction): Instructiuons to add at
            construction.

    Raises:
        TypeError: If initialization list contains non-Instruction objects.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi

        Create a new circuit object

        >>> c = Circuit()

        Add a GateX (Pauli-X) gate on qubit 0

        >>> c.push(GateX(), 0)
        1-qubit circuit with 1 instructions:
        └── X @ q0

        Add a Controlled-NOT (CX) gate with control qubit 0 and target qubit 1

        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── X @ q0
        └── CX @ q0, q1

        Add a Parametric GateR gate with parameters np.pi and np.pi

        >>> c.push(GateRX(pi / 4))

        Add a Reset gate on qubit 0

        >>> c.push(Reset(), 0)
        2-qubit circuit with 3 instructions:
        ├── X @ q0
        ├── CX @ q0, q1
        └── Reset @ q0

        Add a Barrier gate on qubits 0 and 1

        >>> c.push(Barrier(2), 0, 1)
        2-qubit circuit with 4 instructions:
        ├── X @ q0
        ├── CX @ q0, q1
        ├── Reset @ q0
        └── Barrier @ q0, q1

        Add a Measurement gate on qubit 0, storing the result in bit 0.

        >>> c.push(Measure(), 0, 0)
        2-qubit circuit with 5 instructions:
        ├── X @ q0
        ├── CX @ q0, q1
        ├── Reset @ q0
        ├── Barrier @ q0, q1
        └── Measure @ q0, c0

        Add a Control gate with multi-GateX as the target gates. The first 3
        targets are the control qubits.

        >>> c.push(Control(3, GateX()), 0, 1, 2, 3)
        4-qubit circuit with 6 instructions:
        ├── X @ q0
        ├── CX @ q0, q1
        ├── Reset @ q0
        ├── Barrier @ q0, q1
        ├── Measure @ q0, c0
        └── C₃X @ q0, q1, q2, q3

        Add a 3-qubit Parallel gate with GateX

        >>> c.push(Parallel(3,GateX()),0, 1, 2)
        4-qubit circuit with 7 instructions:
        ├── X @ q0
        ├── CX @ q0, q1
        ├── Reset @ q0
        ├── Barrier @ q0, q1
        ├── Measure @ q0, c0
        ├── C₃X @ q0, q1, q2, q3
        └── Parallel(3, X) @ q0, q1, q2

        To add operations without constructing them first, use the
        `c.emplace(...)` function.

    Available operations
    --------------------

    **Gates**

    **Single qubit gates**
        :func:`GateX` :func:`GateY` :func`GateZ` :func:`GateH`
        :func:`GateS` :func:`GateSDG`
        :func:`GateT` :func:`GateTDG`
        :func:`GateSX` :func:`GateSXDG`
        :func:`GateID`

    **Single qubit gates (parametric)**
        :func:`GateU` :func:GateUPpase
        :func:`GateP`
        :func:`GateRX` :func:`GateRY` :func:`GateRZ` :func:`GateP`

    **Two qubit gates**
        :func:`GateCX` :func:`GateCY` :func:`GateCZ`
        :func:`GateCH`
        :func:`GateSWAP` :func:`GateISWAP`
        :func:`GateCS`, :func:`GateCSX`

    **Two qubit gates (parametric)**
        :func:`GateCU`
        :func:`GateCP`
        :func:`GateCRX` :func:`GateCRY` :func:`GateCRZ`
        :func:`GateRXX` :func:`GateRYY` :func:`GateRZZ`
        :func:`GateXXplusYY` :func:`GateXXminusYY`

    **Other**
        :func:`GateCustom`

    **No-ops**
        :func:`Barrier`

    **Non-unitary operations**
        :func:`Measure` :func:`Reset`

    **Composite operations**
        :func:`Control` :func:`Parallel`

    **Generalized gates
        :func:`GPhase` :func:`QFT` :func:`PhaseGradient`
    """

    def __init__(self, instructions=None):

        if instructions is None:
            instructions = []

        if not isinstance(instructions, list):
            raise TypeError(
                "Circuit should be initialized with a list of Instruction")

        for instruction in instructions:
            if not isinstance(instruction, mi.Instruction):
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
        Returns the number of bits in the circuit.
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

            args (integers or iterables): Target qubits and bits for the operation
                (not instruction), given as variable number of arguments.

        Raises:
            TypeError: If operation is not an Operation object.

            ValueError: If the number of arguments is incorrect or the
                 target qubits specified are invalid.

        Examples:
            Adding multiple operations to the Circuit (The args can be
            integers or integer-valued iterables)

            >>> from mimiqcircuits import *
            >>> from symengine import pi
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instructions:
            └── H @ q0
            >>> c.push(GateT(), 0)
            1-qubit circuit with 2 instructions:
            ├── H @ q0
            └── T @ q0
            >>> c.push(GateH(), [0,2])
            3-qubit circuit with 4 instructions:
            ├── H @ q0
            ├── T @ q0
            ├── H @ q0
            └── H @ q2
            >>> c.push(GateS(), 0)
            3-qubit circuit with 5 instructions:
            ├── H @ q0
            ├── T @ q0
            ├── H @ q0
            ├── H @ q2
            └── S @ q0
            >>> c.push(GateCX(), [2, 0], 1)
            3-qubit circuit with 7 instructions:
            ├── H @ q0
            ├── T @ q0
            ├── H @ q0
            ├── H @ q2
            ├── S @ q0
            ├── CX @ q2, q1
            └── CX @ q0, q1
            >>> c.push(GateH(), 0)
            3-qubit circuit with 8 instructions:
            ├── H @ q0
            ├── T @ q0
            ├── H @ q0
            ├── H @ q2
            ├── S @ q0
            ├── CX @ q2, q1
            ├── CX @ q0, q1
            └── H @ q0
            >>> c.push(Barrier(3), *range(3)) # equivalent to c.push(Barrier(3), 0, 1, 2)
            3-qubit circuit with 9 instructions:
            ├── H @ q0
            ├── T @ q0
            ├── H @ q0
            ├── H @ q2
            ├── S @ q0
            ├── CX @ q2, q1
            ├── CX @ q0, q1
            ├── H @ q0
            └── Barrier @ q0, q1, q2
            >>> c.push(Measure(), range(3), range(3))
            3-qubit circuit with 12 instructions:
            ├── H @ q0
            ├── T @ q0
            ├── H @ q0
            ├── H @ q2
            ├── S @ q0
            ├── CX @ q2, q1
            ├── CX @ q0, q1
            ├── H @ q0
            ├── Barrier @ q0, q1, q2
            ├── Measure @ q0, c0
            ├── Measure @ q1, c1
            └── Measure @ q2, c2
            >>> c
            3-qubit circuit with 12 instructions:
            ├── H @ q0
            ├── T @ q0
            ├── H @ q0
            ├── H @ q2
            ├── S @ q0
            ├── CX @ q2, q1
            ├── CX @ q0, q1
            ├── H @ q0
            ├── Barrier @ q0, q1, q2
            ├── Measure @ q0, c0
            ├── Measure @ q1, c1
            └── Measure @ q2, c2
        """
        N = 0
        M = 0
        L = len(args)

        if isinstance(operation, mi.Instruction):
            if L != 0:
                raise (ValueError(
                    "No extra arguments allowed when pushing an instruction."))

            self.instructions.append(operation)
            return self

        if not isinstance(operation, Operation):
            raise (TypeError("Non Operation object passed to push."))

        N = operation.num_qubits
        M = operation.num_bits

        if L != N + M:
            raise (ValueError(
                f"Wrong number of target qubits and bits, given {L} for a {N} qubits + {M} bits operation."
            ))

        targets = []
        hasiterable = False
        for arg in args[:-1]:
            if isinstance(arg, int):
                targets.append(repeat(arg))
            elif isinstance(arg, Iterable):
                targets.append(arg)
                hasiterable = True
            else:
                raise TypeError(
                    f"Invalid target type for {arg} of type {type(arg)}")

        larg = args[-1]
        if isinstance(larg, int):
            if hasiterable:
                targets.append(repeat(larg))
            else:
                targets.append([larg])
        elif isinstance(larg, Iterable):
            targets.append(larg)
        else:
            raise TypeError(
                f"Invalid target type for {arg} of type {type(larg)}")

        for tg in zip(*targets):
            if operation == Barrier:
                self.instructions.append(
                    mi.Instruction(Barrier(N), (*tg,), ()))
            else:
                self.instructions.append(mi.Instruction(
                    operation, (*tg[:N],), (*tg[N:],)))

        return self

    def emplace(self, operation, *args):
        """
        Constructs and adds an Operation to the end of the circuit.

        It is useful to add to the circuit operations that are dependent on the
        number of qubits.

        Arguments:
            operation (Type subclass of Operation): the type of operation to
                add.
            args (vararg of list): A variable number of arguments compriseing a
                list of parameters (if the operation is parametric), one list
                of qubits for each quantum register,  and one list of bits of
                every classical register supported.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.emplace(GateX, [0])
            1-qubit circuit with 1 instructions:
            └── X @ q0
            >>> c.emplace(GateCX, [0, 1])
            >>> c.emplace(GateRX, [0.2], [0])
            1-qubit circuit with 2 instructions:
            ├── X @ q0
            └── RX(0.2) @ q0
            >>> c.emplace(QFT, range(10))
            10-qubit circuit with 3 instructions:
            ├── X @ q0
            ├── RX(0.2) @ q0
            └── QFT @ q0, q1, q2, q3, q4, q5, q6, q7, q8, q9
        """
        if not issubclass(operation, Operation):
            raise TypeError("Non Operation type passed to emplace.")

        nqr = operation._num_qregs
        ncr = operation._num_cregs

        if len(operation._parnames) == 0:
            np = 0
        else:
            np = 1

        if len(args) != nqr + ncr + np:
            raise ValueError(
                f"Wrong number of arguments. Expected {nqr + ncr + np} lists, got {len(args)}")

        opargs = []
        if operation._num_qubits is None:
            opargs.extend([len(x) for x in args[np:nqr+np]])
        if operation._num_bits is None:
            opargs.extend([len(x) for x in args[nqr+np:]])
        if np != 0:
            opargs.extend(args[0])

        op = operation(*opargs)

        qubits = []
        for q in args[np:np+nqr]:
            qubits.extend(q)

        bits = []
        for b in args[np+nqr:]:
            bits.extend(b)

        self.push(op, *qubits, *bits)

        return self

    def insert(self, index: int, operation, *args):
        """
        Inserts an operation at a specific index in the circuit.

        Args:
            index (int): The index at which the operation should be inserted.

            operation (Operation or Instruction): the quantum operation to add.

            args (integers or iterables): Target qubits and bits for the operation
                (not instruction), given as variable number of arguments.

        Raises:
            TypeError: If operation is not an Operation object.

            ValueError: If the number of arguments is incorrect or the
                 target qubits specified are invalid.

        Examples:
            Inserting an operation to the specify index of the circuit

            >>> from mimiqcircuits import *
            >>> c= Circuit()
            >>> c.push(GateX(), 0)
            1-qubit circuit with 1 instructions:
            └── X @ q0
            >>> c.push(GateCX(),0,1)
            2-qubit circuit with 2 instructions:
            ├── X @ q0
            └── CX @ q0, q1
            >>> c.insert(1, GateH(), 0)
            2-qubit circuit with 3 instructions:
            ├── X @ q0
            ├── H @ q0
            └── CX @ q0, q1
        """
        N = 0
        M = 0
        L = len(args)

        if isinstance(operation, mi.Instruction):
            if L != 0:
                raise (ValueError(
                    "No extra arguments allowed when inserting an instruction."))

            self.instructions.insert(index, operation)
            return self

        if operation == Barrier:
            N = L
        else:
            if not isinstance(operation, Operation):
                raise (TypeError("Non Operation object passed to push."))

            N = operation.num_qubits
            M = operation.num_bits

        if L != N + M:
            raise (ValueError(
                f"Wrong number of target qubits and bits, given {L} for a {N} qubits + {M} bits operation."
            ))

        if operation == Barrier:
            self.instructions.insert(
                index, mi.Instruction(Barrier(N), (*args,), ()))
        else:
            self.instructions.insert(index, mi.Instruction(
                operation, (*args[:N],), (*args[N:],)))

        return self

    def append(self, other):
        """
        Appends all the gates of the given circuit at the end of the current
        circuit.

        Args:
            other (Circuit): the circuit to append.
        """
        instructions = None
        if isinstance(other, Circuit):
            instructions = other.instructions
        elif isinstance(other, list):
            instructions = other
        else:
            raise TypeError(
                "Only allowed to append a circuit or a list of instructions")

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

    def _decompose(self, circ):
        for instruction in self.instructions:
            instruction._decompose(circ)
        return circ

    def decompose(self):
        """
        Decompose all the gates in the circuit.

        If applied multiple times, will reduce the circuit to a basis set of U,
        CX and GPhase gates.
        """
        return self._decompose(Circuit())

    def evaluate(self, d):
        c = Circuit()

        for inst in self.instructions:
            c.push(inst.evaluate(d))

        return c

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
        output = f'{nq}-qubit circuit with {n} instructions:'

        # iterate from the second gate
        for g in self.instructions[:-1]:
            output += f'\n├── {g}'

        g = self.instructions[-1]
        output += f'\n└── {g}'

        return output

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Circuit):
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

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def saveproto(self, filename):
        """
        Saves the circuit as a protobuf (binary) file.

        Arguments:
            filename (str): The name of the file to save the circuit to.

        Returns:
            int: The number of bytes written to the file.
        """
        with open(filename, "wb") as f:
            return f.write(toproto_circuit(self).SerializeToString())

    @staticmethod
    def loadproto(filename):
        """
        Loads a circuit from a protobuf (binary) file.

        Arguments:
            filename (str): The name of the file to load the circuit from.

        Returns:
            Circuit: The circuit loaded from the file.
        """
        with open(filename, "rb") as f:
            circuit_proto = circuit_pb.Circuit()
            circuit_proto.ParseFromString(f.read())
            return fromproto_circuit(circuit_proto)


# export the cirucit classes
__all__ = ['Instruction', 'Circuit']
