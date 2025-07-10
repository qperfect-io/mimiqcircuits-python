#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
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

from __future__ import (
    annotations,
)  # Required for postponed evaluation of type hints in Python 3.7+
import mimiqcircuits as mc
from mimiqcircuits.instruction import Instruction
import copy
from collections.abc import Iterable
from itertools import repeat
import shutil
from typing import List, Union
import random
import numpy as np
from mimiqcircuits.push import push_instruction_container


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
        └── X @ q[0]
        <BLANKLINE>

        Add a Controlled-NOT (CX) gate with control qubit 0 and target qubit 1

        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── X @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>

        Add a Parametric GateRX gate with parameters pi/4

        >>> c.push(GateRX(pi / 4),0)
        2-qubit circuit with 3 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        └── RX((1/4)*pi) @ q[0]
        <BLANKLINE>

        Add a Reset gate on qubit 0

        >>> c.push(Reset(), 0)
        2-qubit circuit with 4 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        └── Reset @ q[0]
        <BLANKLINE>

        Add a Barrier gate on qubits 0 and 1

        >>> c.push(Barrier(2), 0, 1)
        2-qubit circuit with 5 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        └── Barrier @ q[0,1]
        <BLANKLINE>

        Add a Measurement gate on qubit 0, storing the result in bit 0.

        >>> c.push(Measure(), 0, 0)
        2-qubit, 1-bit circuit with 6 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0,1]
        └── M @ q[0], c[0]
        <BLANKLINE>

        Add a Control gate with GateX as the target gate. The first 3
        qubits are the control qubits.

        >>> c.push(Control(3, GateX()), 0, 1, 2, 3)
        4-qubit, 1-bit circuit with 7 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0,1]
        ├── M @ q[0], c[0]
        └── C₃X @ q[0,1,2], q[3]
        <BLANKLINE>

        Add a 3-qubit Parallel gate with GateX

        >>> c.push(Parallel(3,GateX()),0, 1, 2)
        4-qubit, 1-bit circuit with 8 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0,1]
        ├── M @ q[0], c[0]
        ├── C₃X @ q[0,1,2], q[3]
        └── ⨷ ³ X @ q[0], q[1], q[2]
        <BLANKLINE>

        To add operations without constructing them first, use the
        `c.emplace(...)` function.

    Available operations
    --------------------

    **Gates**

    **Single qubit gates**
        :func:`GateX` :func:`GateY` :func:`GateZ` :func:`GateH`
        :func:`GateS` :func:`GateSDG`
        :func:`GateT` :func:`GateTDG`
        :func:`GateSX` :func:`GateSXDG`
        :func:`GateID`

    **Single qubit gates (parametric)**
        :func:`GateU` :func:`GateP`
        :func:`GateRX` :func:`GateRY` :func:`GateRZ` :func:`GateP`

    **Two qubit gates**
        :func:`GateCX` :func:`GateCY` :func:`GateCZ`
        :func:`GateCH`
        :func:`GateSWAP` :func:`GateISWAP`
        :func:`GateCS` :func:`GateCSX`
        :func:`GateECR` :func:`GateDCX`

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

    **Power & Inverse operations**
        :func:`Power`
        :func:`Inverse`

    **Generalized gates**
        :func:`QFT` :func:`PhaseGradient`
    """

    def __init__(self, instructions=None):
        if instructions is None:
            instructions = []

        if not isinstance(instructions, list):
            raise TypeError("Circuit should be initialized with a list of Instruction")

        for instruction in instructions:
            if not isinstance(instruction, Instruction):
                raise TypeError("Non Gate object passed to constructor.")

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

        return n + 1

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

        return n + 1

    def getparams(self):
        params = []
        for inst in self.instructions:
            if hasattr(inst, "getparams"):
                params.extend(inst.getparams())
        return params

    def listvars(self):
        return list({v for inst in self.instructions for v in inst.listvars()})

    def num_zvars(self):
        """
        Returns the number of z-variables in the circuit.
        """
        n = -1
        for instruction in self.instructions:
            zbits = instruction.zvars
            if len(zbits) == 0:
                continue
            m = max(zbits)
            if m > n:
                n = m

        return n + 1

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
            └── H @ q[0]
            <BLANKLINE>
            >>> c.push(GateT(), 0)
            1-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── T @ q[0]
            <BLANKLINE>
            >>> c.push(GateH(), [0,2])
            3-qubit circuit with 4 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            └── H @ q[2]
            <BLANKLINE>
            >>> c.push(GateS(), 0)
            3-qubit circuit with 5 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            └── S @ q[0]
            <BLANKLINE>
            >>> c.push(GateCX(), [2, 0], 1)
            3-qubit circuit with 7 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            └── CX @ q[0], q[1]
            <BLANKLINE>
            >>> c.push(GateH(), 0)
            3-qubit circuit with 8 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            └── H @ q[0]
            <BLANKLINE>
            >>> c.push(Barrier(3), *range(3)) # equivalent to c.push(Barrier(3), 0, 1, 2)
            3-qubit circuit with 9 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            ├── H @ q[0]
            └── Barrier @ q[0,1,2]
            <BLANKLINE>
            >>> c.push(Measure(), range(3), range(3))
            3-qubit, 3-bit circuit with 12 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            ├── H @ q[0]
            ├── Barrier @ q[0,1,2]
            ├── M @ q[0], c[0]
            ├── M @ q[1], c[1]
            └── M @ q[2], c[2]
            <BLANKLINE>
            >>> c
            3-qubit, 3-bit circuit with 12 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            ├── H @ q[0]
            ├── Barrier @ q[0,1,2]
            ├── M @ q[0], c[0]
            ├── M @ q[1], c[1]
            └── M @ q[2], c[2]
            <BLANKLINE>
        """

        return push_instruction_container(self, operation, *args, check_fn=None)

    def _emplace_operation(self, op, regs):
        lr = len(regs)
        lq = op.num_qregs
        lc = op.num_cregs
        lz = op._num_zregs

        if lr != lq + lc + lz:
            raise ValueError(
                f"Wrong number of registers. Expected {lq} quantum + {lc} classical + {lz} z, got {lr}"
            )

        qr = op.qregsizes
        for i in range(lq):
            if len(regs[i]) != qr[i]:
                raise ValueError(
                    f"Wrong size for {i}th quantum register. Expected {qr[i]} but got {len(regs[i])}."
                )

        cr = op.cregsizes
        for i in range(lc):
            if len(regs[lq + i]) != cr[i]:
                raise ValueError(
                    f"Wrong size for {i}th classical register. Expected {cr[i]} but got {len(regs[lq + i])}."
                )

        zr = op.zregsizes
        for i in range(lz):
            if len(regs[lq + lc + i]) != zr[i]:
                raise ValueError(
                    f"Wrong size for {i}th z-register. Expected {zr[i]} but got {len(regs[lq + lc + i])}."
                )

        targets = []
        for reg in regs:
            targets.extend(reg)

        self.push(op, *targets)

        return self

    def emplace(self, op, *regs):
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
            >>> c.emplace(GateX(), [0])
            1-qubit circuit with 1 instructions:
            └── X @ q[0]
            <BLANKLINE>
            >>> c.emplace(GateRX(0.2), [0])
            1-qubit circuit with 2 instructions:
            ├── X @ q[0]
            └── RX(0.2) @ q[0]
            <BLANKLINE>
            >>> c.emplace(QFT(), range(10))
            10-qubit circuit with 3 instructions:
            ├── X @ q[0]
            ├── RX(0.2) @ q[0]
            └── QFT @ q[0,1,2,3,4,5,6,7,8,9]
            <BLANKLINE>

        """

        if isinstance(op, mc.LazyExpr):
            self._emplace_operation(op(*[len(reg) for reg in regs]), regs)
        elif isinstance(op, mc.Parallel):
            if any((isinstance(reg, Iterable) and len(reg) != 1) for reg in regs):
                raise ValueError("Each iterable should contain exactly one qubit.")
            self.push(op, *regs)
        elif isinstance(op, mc.Operation):
            if isinstance(op, mc.AbstractOperator) and not isinstance(op, mc.Gate):
                raise ValueError(
                    "Cannot emplace an abstract operator that is not a Gate into the circuit."
                )
            self._emplace_operation(op, regs)
        elif isinstance(op, type) and issubclass(op, mc.Operation):
            self._emplace_operation(op, regs)
        else:
            raise TypeError("Invalid type passed to emplace")

        return self

    def insert(self, index: int, operation, *args):
        """
        Inserts an operation or another circuit at a specific index in the circuit.

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
            └── X @ q[0]
            <BLANKLINE>
            >>> c.push(GateCX(),0,1)
            2-qubit circuit with 2 instructions:
            ├── X @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>
            >>> c.insert(1, GateH(), 0)
            2-qubit circuit with 3 instructions:
            ├── X @ q[0]
            ├── H @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>
        """
        if isinstance(operation, Circuit):
            for inst in operation.instructions:
                self.instructions.insert(index, inst)
                index += 1
        else:
            N = 0
            M = 0
            Z = 0
            L = len(args)

            if isinstance(operation, Instruction):
                if L != 0:
                    raise ValueError(
                        "No extra arguments allowed when inserting an instruction."
                    )

                self.instructions.insert(index, operation)
                return self

            if operation == mc.Barrier:
                N = L
            else:
                if not isinstance(operation, mc.Operation):
                    raise TypeError("Non Operation object passed to push.")

                # Check if the operation is an abstract operator but not a gate
                if isinstance(operation, mc.AbstractOperator) and not isinstance(
                    operation, mc.Gate
                ):
                    raise ValueError(
                        "Cannot add an abstract operator that is not a Gate to the circuit."
                    )

                N = operation.num_qubits
                M = operation.num_bits
                Z = operation.num_zvars

            if L != N + M + Z:
                raise ValueError(
                    f"Wrong number of target qubits and bits, given {L} for a {N} qubits + {M} bits operation + {Z} Z-variables."
                )

            if operation == mc.Barrier:
                self.instructions.insert(
                    index, Instruction(mc.Barrier(N), (*args,), ())
                )
            else:
                self.instructions.insert(
                    index,
                    Instruction(
                        operation, (*args[:N],), (*args[N : N + M],), (*args[N + M :],)
                    ),
                )

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
                "Only allowed to append a circuit or a list of instructions"
            )

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
        return self

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

        If applied multiple times, will reduce the circuit to a basis set of U
        and CX gates.
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

    def __str__(self, compact=False):
        lines = shutil.get_terminal_size().lines
        n = len(self)

        output = ""

        if compact and n > 0:
            nq = self.num_qubits()
            nb = self.num_bits()
            nz = self.num_zvars()

            parts = []
            if nq > 0:
                parts.append(f"{nq}-qubit")
            if nb > 0:
                parts.append(f"{nb}-bit")
            if nz > 0:
                parts.append(f"{nz}-zvar")

            output += f"{', '.join(parts)} circuit with {n} instructions:\n"

            if lines - 4 <= 0:
                output += "└── ...\n"
            else:
                max_display = min(n, lines - 4)

                for g in self.instructions[: max_display - 1]:
                    output += f"├── {g}\n"

                if n > max_display:
                    output += "⋮   ⋮\n"

                g = self.instructions[-1]
                output += f"└── {g}\n"

        elif not compact and n > 0:
            nq = self.num_qubits()
            output = f"{nq}-qubit circuit with {n} instructions:"

            # iterate from the second gate
            for g in self.instructions[:-1]:
                output += f"\n├── {g}"

            g = self.instructions[-1]
            output += f"\n└── {g}"

        elif n == 0:
            output += "empty circuit"

        return output

    def __repr__(self):
        return self.__str__(compact=True)

    def __eq__(self, other):
        if not isinstance(other, Circuit):
            return False

        return self.instructions == other.instructions

    def __ne__(self, other):
        return not self.__eq__(other)

    def __class__(self):
        return Circuit

    def depth(self):
        """
        Computes the depth of the quantum circuit, including qubits, bits, and z-registers.
        """
        if self.empty() or self.num_qubits() == 0:
            return 0

        d = [0 for _ in range(self.num_qubits() + self.num_bits() + self.num_zvars())]

        for g in self:
            if isinstance(g.operation, mc.Barrier):
                continue
            nq = self.num_qubits()
            nb = self.num_bits()
            optargets = (
                g.qubits
                + tuple(map(lambda x: x + nq, g.bits))
                + tuple(map(lambda x: x + nq + nb, g.zvars))
            )
            dm = max([d[t] for t in optargets])
            for t in g.qubits:
                d[t] = dm + 1
            for t in g.bits:
                d[t + nq] = dm + 1
            for t in g.zvars:
                d[t + nq + nb] = dm + 1

        return max(d)

    def copy(self):
        """Creates a shallow copy of the circuit.
            To create a full copy use deepcopy() instead.

        Returns:
            Circuit: A new Circuit object containing references to the same attributes as the original circuit
        """
        return copy.copy(self)

    def deepcopy(self):
        """Creates a copy of the object and for all its attributes

        Returns:
            Circuit: A new Circuit object fully identical the original circuit
        """
        return copy.deepcopy(self)

    def get_on_qubits(self, target_qubits):
        """
        Get instructions that involve the specified target qubits.

        Arguments:
            target_qubits (list or int): Qubits for which to retrieve instructions.

        Returns:
            Circuit: A new Circuit containing only the instructions that involve the specified qubits.
        """
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]

        selected_instructions = []

        for instruction in self.instructions:
            instruction_qubits = set(instruction.qubits + instruction.bits)
            if any(qubit in instruction_qubits for qubit in target_qubits):
                selected_instructions.append(instruction)

        return Circuit(instructions=selected_instructions)

    def saveproto(self, file):
        """
        Saves the circuit as a protobuf (binary) file.

        Arguments:
            filename (str): The name of the file to save the circuit to.

        Returns:
            int: The number of bytes written to the file.

        Examples:

            >>> from mimiqcircuits import *
            >>> from symengine import *
            >>> import tempfile
            >>> x, y = symbols("x y")
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instructions:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.push(GateXXplusYY(x**2, y),0,1)
            2-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── XXplusYY(x**2, y) @ q[0,1]
            <BLANKLINE>
            >>> c.push(Measure(),0,0)
            2-qubit, 1-bit circuit with 3 instructions:
            ├── H @ q[0]
            ├── XXplusYY(x**2, y) @ q[0,1]
            └── M @ q[0], c[0]
            <BLANKLINE>
            >>> tmpfile = tempfile.NamedTemporaryFile(suffix=".pb", delete=True)
            >>> c.saveproto(tmpfile.name)
            64
            >>> c.loadproto(tmpfile.name)
            2-qubit, 1-bit circuit with 3 instructions:
            ├── H @ q[0]
            ├── XXplusYY(x**2, y) @ q[0,1]
            └── M @ q[0], c[0]
            <BLANKLINE>

            Note:
                This example uses a temporary file to demonstrate the save and load functionality.
                You can save your file with any name at any location using:

                .. code-block:: python

                    c.saveproto("example.pb")
                    c.loadproto("example.pb")
        """
        from mimiqcircuits.proto.circuitproto import toproto_circuit

        if hasattr(file, "write"):
            return file.write(toproto_circuit(self).SerializeToString())
        else:
            # try:
            with open(file, "wb") as f:
                return f.write(toproto_circuit(self).SerializeToString())
        # except TypeError:
        #     raise ValueError(
        #         "Invalid file object. Sould be a filename of a file-like object"
        #     )
        # except Exception as e:
        #     raise e

    @staticmethod
    def loadproto(file):
        """
        Loads a circuit from a protobuf (binary) file.

        Arguments:
            filename (str): The name of the file to load the circuit from.

        Returns:
            Circuit: The circuit loaded from the file.

        Note:

            Look for example in :func:`Circuit.saveproto`

        """

        from mimiqcircuits.proto import circuit_pb2
        from mimiqcircuits.proto.circuitproto import fromproto_circuit

        if isinstance(file, str):
            with open(file, "rb") as f:
                circuit_proto = circuit_pb2.Circuit()
                circuit_proto.ParseFromString(f.read())
                return fromproto_circuit(circuit_proto)
        elif hasattr(file, "read"):
            circuit_proto = circuit_pb2.Circuit()
            circuit_proto.ParseFromString(file.read())
            return fromproto_circuit(circuit_proto)
        else:
            raise ValueError(
                "Invalid file object. Sould be a filename of a file-like object"
            )

    def draw(self):
        """
        Draws the entire quantum circuit on the ASCII canvas and handles the layout of various quantum operations.

        This method iterates through all instructions in the circuit, determines the required width for each operation,
        and delegates the drawing of each operation to the appropriate specialized method based on the operation type.
        If an operation's width exceeds the available space in the current row of the canvas, the canvas is printed and
        reset to continue drawing from a new starting point.

        The method manages different operation types including control, measurement, reset, barrier, parallel, and
        conditional (if) operations using specific drawing methods from the `AsciiCircuit` class.

        Raises:
            TypeError: If any item in the circuit's instructions is not an instance of `Instruction`.
            ValueError: If an operation cannot be drawn because it exceeds the available canvas width even after a reset.

        Prints:
            The current state of the ASCII canvas, either incrementally after each operation if space runs out, or
            entirely at the end of processing all instructions.

        Returns:
            None
        """
        num_qubits = self.num_qubits()
        num_bits = self.num_bits()
        num_zvars = self.num_zvars()

        canvas = mc.AsciiCircuit()
        canvas.draw_wires(range(num_qubits), range(num_bits), range(num_zvars))

        for instruction in self.instructions:
            if not isinstance(instruction, Instruction):
                raise TypeError("Must be an Instruction")

            operation = instruction.get_operation()
            required_width = operation.asciiwidth(
                instruction.qubits,
                instruction.bits if hasattr(instruction, "bits") else [],
                instruction.zvars if hasattr(instruction, "zvars") else [],
            )

            if required_width > canvas.canvas.get_cols() - canvas.get_current_col():
                print(canvas.canvas)
                print("...")
                canvas.reset()
                canvas.draw_wires(range(num_qubits), range(num_bits), range(num_zvars))

            if required_width > canvas.canvas.get_cols() - canvas.get_current_col():
                raise ValueError(
                    "Cannot draw instruction. Insufficient space on screen."
                )

            # Handle drawing based on operation type
            if isinstance(operation, mc.Control):
                canvas.draw_control(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                )

            elif isinstance(operation, mc.PauliString):
                canvas.draw_paulistring(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                    instruction.zvars if hasattr(instruction, "zvars") else [],
                )

            elif isinstance(operation, mc.Reset):
                canvas.draw_reset(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                    instruction.zvars if hasattr(instruction, "zvars") else [],
                )

            elif isinstance(operation, mc.Barrier):
                canvas.draw_barrier(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                    instruction.zvars if hasattr(instruction, "zvars") else [],
                )

            elif isinstance(operation, mc.Parallel):
                canvas.draw_parallel(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                    instruction.zvars if hasattr(instruction, "zvars") else [],
                )

            elif isinstance(operation, mc.IfStatement):
                canvas.draw_ifstatement(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                    instruction.zvars if hasattr(instruction, "zvars") else [],
                )

            elif isinstance(operation, mc.Operation):
                canvas.draw_operation(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                    instruction.zvars if hasattr(instruction, "zvars") else [],
                )
            else:
                # Default drawing method for general operations
                canvas.draw_instruction(instruction)

        print(canvas.canvas)
        return None

    def specify_operations(self):
        """
        Summarizes the types and numbers of operations in the circuit.

        This function inspects each instruction in the circuit and categorizes it by
        the number of qubits, bits, and z-variables involved in the operation. It
        then prints a summary of the total number of operations in the circuit and
        a breakdown of the number of operations grouped by their type.

        Examples:

            >>> from mimiqcircuits import *
            >>> c = Circuit()

            Add a Pauli-X (GateX) gate on qubit 0

            >>> c.push(GateX(), 0)
            1-qubit circuit with 1 instructions:
            └── X @ q[0]
            <BLANKLINE>

            Add a Controlled-NOT (CX) gate with control qubit 0 and target qubit 1

            >>> c.push(GateCX(), 0, 1)
            2-qubit circuit with 2 instructions:
            ├── X @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>

            Add a Measurement operation on qubit 0, storing the result in bit 0

            >>> c.push(Measure(), 0, 0)
            2-qubit, 1-bit circuit with 3 instructions:
            ├── X @ q[0]
            ├── CX @ q[0], q[1]
            └── M @ q[0], c[0]
            <BLANKLINE>

            Add an ExpectationValue operation with GateX on qubit 1, storing the result in z-variable 2.

            >>> c.push(ExpectationValue(GateX()), 1, 2)
            2-qubit, 1-bit, 3-zvar circuit with 4 instructions:
            ├── X @ q[0]
            ├── CX @ q[0], q[1]
            ├── M @ q[0], c[0]
            └── ⟨X⟩ @ q[1], z[2]
            <BLANKLINE>

            Print a summary of the types and numbers of operations

            >>> c.specify_operations()
            Total number of operations: 4
            ├── 1 x 1_qubits
            ├── 1 x 2_qubits
            ├── 1 x 1_qubits & 1_bits
            └── 1 x 1_qubits & 1_zvars
        """
        counts = {}
        for i in self.instructions:
            nq = len(i._qubits)
            nb = len(i._bits)
            nz = len(i._zvars)

            key = ""

            if nq > 0:
                key += f"{nq}_qubits"

            if nb > 0:
                if key:
                    key += " & "
                key += f"{nb}_bits"
            if nz > 0:
                if key:
                    key += " & "
                key += f"{nz}_zvars"

            counts[key] = counts.get(key, 0) + 1

        total_operations = sum(counts.values())
        print(f"Total number of operations: {total_operations}")

        count_items = list(counts.items())
        for idx, (key, count) in enumerate(count_items):
            if idx == len(count_items) - 1:
                print(f"└── {count} x {key}")
            else:
                print(f"├── {count} x {key}")

    def is_symbolic(self):
        """
        Check whether the circuit contains any symbolic (unevaluated) parameters.

        This method examines each instruction in the circuit to determine if any parameter remains
        symbolic (i.e., unevaluated). It recursively checks through each instruction and its nested
        operations, if any.

        Returns:
            bool: True if any parameter is symbolic (unevaluated), False if all parameters are fully evaluated.

        Examples:

            >>> from mimiqcircuits import *
            >>> from symengine import *
            >>> x, y = symbols("x y")
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instructions:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.is_symbolic()
            False
            >>> c.push(GateP(x), 0)
            1-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── P(x) @ q[0]
            <BLANKLINE>
            >>> c.is_symbolic()
            True
            >>> c = c.evaluate({x: 1, y: 2})
            >>> c
            1-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── P(1) @ q[0]
            <BLANKLINE>
            >>> c.is_symbolic()
            False
        """
        return any(
            instruction.operation.is_symbolic() for instruction in self.instructions
        )

    def add_noise_to_gate_single(
        self, g, kraus: Union[mc.krauschannel, mc.Gate], before=False
    ):
        """
        Adds a noise operation `kraus` before or after every instance of a given operation `g`.

        The noise operation `kraus` can be a Kraus channel or a gate and will act on the same qubits
        as the operation `g` to which it is being added.

        Args:
            g (Operation): The operation to which the noise will be added.
            kraus (Operation): The noise operation to be added, which can be a Kraus channel or any valid gate.
            before (bool, optional): If True, the noise is added before the operation `g`. Default is False.

        Raises:
            ValueError: If the noise operation is the same as the operation `g`, to avoid recursion.

        See also:
            :func:`add_noise`: Adds noise to multiple operations at once or in a parallel block.
        """
        if not isinstance(g, mc.Operation):
            raise ValueError(f"{g} must be an Operation")

        if not isinstance(kraus, (mc.krauschannel, mc.Gate)):
            raise ValueError(f"{kraus} is not of type {mc.krauschannel} or {mc.Gate}")

        if g == kraus:
            raise ValueError(
                "Noise can't be the same as gate, otherwise recursion problem."
            )

        rel = 0 if before else 1

        i = 0
        while i < len(self.instructions):
            if self.instructions[i].operation == g:
                qubits = self.instructions[i].qubits
                self.insert(i + rel, kraus, *qubits)
                i += 1
            i += 1

        return self

    def add_noise_to_gate_parallel(
        self, g, kraus: Union[mc.krauschannel, mc.Gate], before=False
    ):
        """
        Adds a block of noise operations `kraus` after/before every block of a given operation `g`.

        The function identifies blocks of consecutive transversal operations of type `g` and
        adds a block of transversal noise operations `kraus` after each such block. The noise operation
        `kraus` can be a Kraus channel or a gate and will act on the same qubits as the operation `g` to which it is being added.

        Args:
            g (Operation): The operation to which the noise will be added.
            kraus (Operation): The noise operation to be added, which can be a Kraus channel or any valid gate.
            before (bool, optional): If True, the noise is added before the operation `g`. Default is False.

        Raises:
            ValueError: If the noise operation is the same as the operation `g`, to avoid recursion.

        See also:
            :func:`add_noise`: Adds noise to multiple operations at once or in a parallel block.
        """

        if not isinstance(g, mc.Operation):
            raise ValueError(f"{g} must be an Operation")

        if not isinstance(kraus, (mc.krauschannel, mc.Gate)):
            raise ValueError(f"{kraus} is not of type {mc.krauschannel} or {mc.Gate}")

        if g == kraus:
            raise ValueError(
                "Noise can't be the same as gate, otherwise recursion problem."
            )

        i = 0
        while i < len(self.instructions):
            inds = [i]
            qubits = set(self.instructions[i].qubits)

            if self.instructions[i].operation == g:
                j = i + 1
                while (
                    j < len(self.instructions)
                    and self.instructions[j].operation == g
                    and not qubits.intersection(set(self.instructions[j].qubits))
                ):
                    inds.append(j)
                    qubits.update(self.instructions[j].qubits)
                    j += 1

                for rel, j in enumerate(inds):
                    if before:
                        self.insert(i + rel, kraus, *self.instructions[j].qubits)
                    else:
                        self.insert(
                            inds[-1] + 1 + rel, kraus, *self.instructions[j].qubits
                        )

                i += len(inds)  # Noise block shifting
            i += len(inds)  # Gate shifting

        return self

    def add_noise(
        self,
        g: Union[mc.Operation, List[mc.Operation]],
        kraus: Union[mc.krauschannel, mc.Gate, List[mc.Gate], List[mc.krauschannel]],
        before: Union[bool, List[bool]] = False,
        parallel: Union[bool, List[bool]] = False,
    ):
        """
        Adds a noise operation `kraus` to every instance of the operation `g` in the circuit.

        The noise operation `kraus` can be a Kraus channel or a gate and will act on the same qubits
        as the operation `g` to which it is being added.

        The operations `g` and `kraus` must act on the same number of qubits.

        Args:
            g (Operation or list of Operation): The operation(s) to which the noise will be added.
            kraus (krauschannel or list of krauschannel): The noise operation(s) to be added.
            before (bool or list of bool, optional): If True, the noise is added before the operation. Default is False.
            parallel (bool or list of bool, optional): If True, noise is added as a block. Default is False.

        Raises:
            ValueError: If `g` and `kraus` are not of the same length, or if their number of qubits differ.
            TypeError: If `before` or `parallel` are not a bool or a list of bool.

        Returns:
            Circuit: The modified circuit with the noise added.

        Examples:

        Adding noise sequentially (not parallel):

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GateH(), [1,2,3])
        4-qubit circuit with 3 instructions:
        ├── H @ q[1]
        ├── H @ q[2]
        └── H @ q[3]
        <BLANKLINE>
        >>> c.add_noise(GateH(), AmplitudeDamping(0.2))
        4-qubit circuit with 6 instructions:
        ├── H @ q[1]
        ├── AmplitudeDamping(0.2) @ q[1]
        ├── H @ q[2]
        ├── AmplitudeDamping(0.2) @ q[2]
        ├── H @ q[3]
        └── AmplitudeDamping(0.2) @ q[3]
        <BLANKLINE>

        Adding noise in parallel:

        >>> c = Circuit()
        >>> c.push(GateH(), [1, 2, 3])
        4-qubit circuit with 3 instructions:
        ├── H @ q[1]
        ├── H @ q[2]
        └── H @ q[3]
        <BLANKLINE>
        >>> c.add_noise(GateH(), AmplitudeDamping(0.2), parallel=True)
        4-qubit circuit with 6 instructions:
        ├── H @ q[1]
        ├── H @ q[2]
        ├── H @ q[3]
        ├── AmplitudeDamping(0.2) @ q[1]
        ├── AmplitudeDamping(0.2) @ q[2]
        └── AmplitudeDamping(0.2) @ q[3]
        <BLANKLINE>

        Parallel will not work if gates aren't transversal.

        >>> c = Circuit()
        >>> c.push(GateCZ(), 1, range(2,5))
        5-qubit circuit with 3 instructions:
        ├── CZ @ q[1], q[2]
        ├── CZ @ q[1], q[3]
        └── CZ @ q[1], q[4]
        <BLANKLINE>
        >>> c.add_noise(GateCZ(), Depolarizing2(0.1), parallel=True)
        5-qubit circuit with 6 instructions:
        ├── CZ @ q[1], q[2]
        ├── Depolarizing(0.1) @ q[1,2]
        ├── CZ @ q[1], q[3]
        ├── Depolarizing(0.1) @ q[1,3]
        ├── CZ @ q[1], q[4]
        └── Depolarizing(0.1) @ q[1,4]
        <BLANKLINE>

        Adding noise before measurement (The `before=True` option is mostly used for `Measure`):

        >>> c = Circuit()
        >>> c.push(Measure(), [1, 2, 3], [1, 2, 3])
        4-qubit, 4-bit circuit with 3 instructions:
        ├── M @ q[1], c[1]
        ├── M @ q[2], c[2]
        └── M @ q[3], c[3]
        <BLANKLINE>
        >>> c.add_noise(Measure(), PauliX(0.1), before=True)
        4-qubit, 4-bit circuit with 6 instructions:
        ├── PauliX(0.1) @ q[1]
        ├── M @ q[1], c[1]
        ├── PauliX(0.1) @ q[2]
        ├── M @ q[2], c[2]
        ├── PauliX(0.1) @ q[3]
        └── M @ q[3], c[3]
        <BLANKLINE>

        Adding unitary gates as noise in the same way:

        >>> c = Circuit()
        >>> c.push(GateH(), [1, 2, 3])
        4-qubit circuit with 3 instructions:
        ├── H @ q[1]
        ├── H @ q[2]
        └── H @ q[3]
        <BLANKLINE>
        >>> c.add_noise(GateH(), GateRX(0.01))
        4-qubit circuit with 6 instructions:
        ├── H @ q[1]
        ├── RX(0.01) @ q[1]
        ├── H @ q[2]
        ├── RX(0.01) @ q[2]
        ├── H @ q[3]
        └── RX(0.01) @ q[3]
        <BLANKLINE>

        """

        # Ensure g and kraus are treated as lists
        g = g if isinstance(g, list) else [g]
        kraus = kraus if isinstance(kraus, list) else [kraus]

        # Check for length mismatch between g and kraus
        if len(g) != len(kraus):
            raise ValueError(
                "Vectors of operations and noise channels must have the same length."
            )

        nops = len(g)

        # Check types of 'before' and 'parallel' and ensure they are the correct length
        if not isinstance(before, (bool, list)):
            raise TypeError("Parameter 'before' has to be a bool or a list of bool.")
        if isinstance(before, list) and len(before) != nops:
            raise ValueError(
                "Vector 'before' must have the same length as the vector of operations."
            )

        if not isinstance(parallel, (bool, list)):
            raise TypeError("Parameter 'parallel' has to be a bool or a list of bool.")
        if isinstance(parallel, list) and len(parallel) != nops:
            raise ValueError(
                "Vector 'parallel' must have the same length as the vector of operations."
            )

        # Vectorize optional parameters if they are not already lists
        if isinstance(before, bool):
            before = [before] * nops
        if isinstance(parallel, bool):
            parallel = [parallel] * nops

        # Validate each operation and noise channel pair
        for operation, noise in zip(g, kraus):
            if not isinstance(operation, mc.Operation):
                raise ValueError(f"{operation} must be an instance of mc.Operation")
            # Check for individual noise type in the list case
            if isinstance(noise, list):
                for n in noise:
                    if not isinstance(n, (mc.krauschannel, mc.Gate)):
                        raise ValueError(
                            f"{n} is not of type {mc.krauschannel} or {mc.Gate}"
                        )
            elif not isinstance(noise, (mc.krauschannel, mc.Gate)):
                raise ValueError(
                    f"{noise} is not of type {mc.krauschannel} or {mc.Gate}"
                )

            if operation.num_qubits != noise.num_qubits:
                raise ValueError(
                    "Noise channel and operation must have the same number of target qubits"
                )

        # Apply noise for each pair
        for operation, noise, add_before, add_parallel in zip(
            g, kraus, before, parallel
        ):
            if add_parallel:
                self.add_noise_to_gate_parallel(operation, noise, before=add_before)
            else:
                self.add_noise_to_gate_single(operation, noise, before=add_before)

        return self

    def sample_mixedunitaries(self, rng=None, ids=False):
        """
            sample_mixedunitaries(rng=None, ids=False)

        Samples one unitary gate for each mixed unitary Kraus channel in the circuit.

        This is possible because for mixed unitary noise channels, the probabilities of each
        Kraus operator are fixed (state-independent).

        Note: This function is internally called (before applying any gate) when executing
        a circuit with noise using trajectories. It can also be used to generate samples
        of circuits without running them.

        See also:
            - :func:`krauschannel.ismixedunitary`
            - :class:`MixedUnitary`

        Args:
            rng (optional): Random number generator. If not provided, Python's default
                random number generator is used.
            ids (optional): Boolean, default=False. Determines whether to include identity
                Kraus operators in the sampled circuit. If True, identity gates are added
                to the circuit; otherwise, they are omitted. Usually, most selected Kraus
                operators will be identity gates.

        Returns:
            Circuit: A copy of the circuit with every mixed unitary Kraus channel replaced
            by one of the unitary gates of the channel. Identity gates are omitted unless
            `ids=True`.

        Examples:
            Gates and non-mixed-unitary Kraus channels remain unchanged:

            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateH(), [1, 2, 3])
            4-qubit circuit with 3 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            └── H @ q[3]
            <BLANKLINE>
            >>> c.push(Depolarizing1(0.5), [1, 2, 3])
            4-qubit circuit with 6 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── Depolarizing(0.5) @ q[1]
            ├── Depolarizing(0.5) @ q[2]
            └── Depolarizing(0.5) @ q[3]
            <BLANKLINE>
            >>> c.push(AmplitudeDamping(0.5), [1, 2, 3])
            4-qubit circuit with 9 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── Depolarizing(0.5) @ q[1]
            ├── Depolarizing(0.5) @ q[2]
            ├── Depolarizing(0.5) @ q[3]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]
            <BLANKLINE>

            >>> rng = random.Random(42)

            >>> new_circuit = c.sample_mixedunitaries(rng=rng, ids=True)
            >>> print(new_circuit)
            4-qubit circuit with 9 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── X @ q[1]
            ├── I @ q[2]
            ├── I @ q[3]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]

            By default, identities are not included:

            >>> new_circuit = c.sample_mixedunitaries(rng=rng)
            >>> print(new_circuit)
            4-qubit circuit with 8 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── Y @ q[2]
            ├── Y @ q[3]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]

            Different calls to the function generate different results:

            >>> new_circuit = c.sample_mixedunitaries(rng=rng)
            >>> print(new_circuit)
            4-qubit circuit with 7 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── Z @ q[1]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]

            >>> new_circuit = c.sample_mixedunitaries(rng=rng)
            >>> print(new_circuit)
            4-qubit circuit with 7 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── X @ q[3]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]
        """

        if rng is None:
            rng = random()

        scirc = Circuit()

        for inst in self.instructions:
            op = inst.get_operation()

            if isinstance(op, mc.krauschannel) and op.ismixedunitary():
                cumulative_probs = op.unwrappedcumprobabilities()

                r = rng.random()

                # Use `next` to find the index of the first cumulative probability greater than `r`
                index = next(i for i, p in enumerate(cumulative_probs) if p > r)

                gate = op.unitarygates()[index]

                if ids or not gate.isidentity():
                    scirc.push(gate, *inst.qubits)
            else:
                scirc.push(inst)

        return scirc


# export the cirucit classes
__all__ = ["Circuit"]
