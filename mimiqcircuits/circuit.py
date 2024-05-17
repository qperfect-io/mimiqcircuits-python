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

import mimiqcircuits as mc
import copy
from collections.abc import Iterable
from itertools import repeat
from mimiqcircuits.proto.circuitproto import *
from mimiqcircuits.proto import circuit_pb
import shutil


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
        2-qubit circuit with 6 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0,1]
        └── Measure @ q[0], c[0]
        <BLANKLINE>

        Add a Control gate with GateX as the target gate. The first 3
        qubits are the control qubits.

        >>> c.push(Control(3, GateX()), 0, 1, 2, 3)
        4-qubit circuit with 7 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0,1]
        ├── Measure @ q[0], c[0]
        └── C₃X @ q[0,1,2], q[3]
        <BLANKLINE>

        Add a 3-qubit Parallel gate with GateX

        >>> c.push(Parallel(3,GateX()),0, 1, 2)
        4-qubit circuit with 8 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0,1]
        ├── Measure @ q[0], c[0]
        ├── C₃X @ q[0,1,2], q[3]
        └── Parallel(3, X) @ q[0], q[1], q[2]
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
            if not isinstance(instruction, mc.Instruction):
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
            3-qubit circuit with 12 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            ├── H @ q[0]
            ├── Barrier @ q[0,1,2]
            ├── Measure @ q[0], c[0]
            ├── Measure @ q[1], c[1]
            └── Measure @ q[2], c[2]
            <BLANKLINE>
            >>> c
            3-qubit circuit with 12 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            ├── H @ q[0]
            ├── Barrier @ q[0,1,2]
            ├── Measure @ q[0], c[0]
            ├── Measure @ q[1], c[1]
            └── Measure @ q[2], c[2]
            <BLANKLINE>
        """
        N = 0
        M = 0
        L = len(args)

        if isinstance(operation, mc.Instruction):
            if L != 0:
                raise (
                    ValueError(
                        "No extra arguments allowed when pushing an instruction."
                    )
                )

            self.instructions.append(operation)
            return self

        if not isinstance(operation, mc.Operation):
            raise (TypeError("Non Operation object passed to push."))

        N = operation.num_qubits
        M = operation.num_bits

        if L != N + M:
            raise (
                ValueError(
                    f"Wrong number of target qubits and bits, given {L} for a {N} qubits + {M} bits operation."
                )
            )

        targets = []
        hasiterable = False
        for arg in args[:-1]:
            if isinstance(arg, int):
                targets.append(repeat(arg))
            elif isinstance(arg, Iterable):
                targets.append(arg)
                hasiterable = True
            else:
                raise TypeError(f"Invalid target type for {arg} of type {type(arg)}")

        larg = args[-1]
        if isinstance(larg, int):
            if hasiterable:
                targets.append(repeat(larg))
            else:
                targets.append([larg])
        elif isinstance(larg, Iterable):
            targets.append(larg)
        else:
            raise TypeError(f"Invalid target type for {arg} of type {type(larg)}")

        for tg in zip(*targets):
            if operation == mc.Barrier:
                self.instructions.append(mc.Instruction(mc.Barrier(N), (*tg,), ()))
            else:
                self.instructions.append(
                    mc.Instruction(operation, (*tg[:N],), (*tg[N:],))
                )

        return self

    def _emplace_operation(self, op, regs):
        lr = len(regs)
        lq = op.num_qregs
        lc = op.num_cregs

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
            L = len(args)

            if isinstance(operation, mc.Instruction):
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

                N = operation.num_qubits
                M = operation.num_bits

            if L != N + M:
                raise ValueError(
                    f"Wrong number of target qubits and bits, given {L} for a {N} qubits + {M} bits operation."
                )

            if operation == mc.Barrier:
                self.instructions.insert(
                    index, mc.Instruction(mc.Barrier(N), (*args,), ())
                )
            else:
                self.instructions.insert(
                    index, mc.Instruction(operation, (*args[:N],), (*args[N:],))
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
            output += f"{nq}-qubit circuit with {n} instructions:\n"

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
        Computes the depth of the quantum circuit.
        """
        if self.empty() or self.num_qubits() == 0:
            return 0

        d = [0 for _ in range(self.num_qubits() + self.num_bits())]

        for g in self:
            if isinstance(g.operation, mc.Barrier):
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
            TypeError: If any item in the circuit's instructions is not an instance of `mc.Instruction`.
            ValueError: If an operation cannot be drawn because it exceeds the available canvas width even after a reset.

        Prints:
            The current state of the ASCII canvas, either incrementally after each operation if space runs out, or
            entirely at the end of processing all instructions.

        Returns:
            None
        """
        num_qubits = self.num_qubits()
        num_bits = self.num_bits()

        canvas = mc.AsciiCircuit()
        print(canvas.canvas)
        canvas.draw_wires(range(num_qubits), range(num_bits))

        for instruction in self.instructions:
            if not isinstance(instruction, mc.Instruction):
                raise TypeError("Must be an Instruction")

            operation = instruction.get_operation()
            required_width = operation.asciiwidth(
                instruction.qubits,
                instruction.bits if hasattr(instruction, "bits") else [],
            )

            if required_width > canvas.canvas.get_cols() - canvas.get_current_col():
                print(canvas.canvas)
                print("...")
                canvas.reset()
                canvas.draw_wires(range(num_qubits), range(num_bits))

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

            elif isinstance(operation, mc.Measure):
                canvas.draw_measure(
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                )

            elif isinstance(operation, mc.Reset):
                canvas.draw_reset(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                )

            elif isinstance(operation, mc.Barrier):
                canvas.draw_barrier(operation, instruction.qubits)

            elif isinstance(operation, mc.Parallel):
                canvas.draw_parallel(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                )

            elif isinstance(operation, mc.IfStatement):
                canvas.draw_ifstatement(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                )

            elif isinstance(operation, mc.Operation):
                canvas.draw_operation(
                    operation,
                    instruction.qubits,
                    instruction.bits if hasattr(instruction, "bits") else [],
                )
            else:
                # Default drawing method for general operations
                canvas.draw_instruction(instruction)

        print(canvas.canvas)
        return None

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


# export the cirucit classes
__all__ = ["Instruction", "Circuit"]
