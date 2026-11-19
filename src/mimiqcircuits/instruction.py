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
"""Instruction class and utilities."""

from mimiqcircuits.operations.operation import Operation
import copy
import numpy as np
from mimiqcircuits.matrices import reorder_qubits_matrix
from mimiqcircuits.instruction_extras import (
    _partition,
    _string_with_square,
    _find_unit_range,
)


# Operation subclasses already seen by Instruction. Membership in this set is
# a plain hash lookup, where isinstance against the Operation ABC costs a call
# through ABCMeta; a type is only added once isinstance has accepted it.
_OPERATION_TYPES = set()


class Instruction:
    """An operation together with the qubits, bits and z-variables it acts on.

    Instructions are the elements of a :class:`Circuit`. They are usually
    created by :meth:`Circuit.push`; build one directly to push it later or to
    insert it at a given position. An instruction cannot be modified after it
    is created.

    Args:
        operation (Operation): The operation to apply.
        qubits (tuple of int): The target qubits, as many as
            ``operation.num_qubits``. Defaults to ``()``.
        bits (tuple of int): The target classical bits, as many as
            ``operation.num_bits``. Defaults to ``()``.
        zvars (tuple of int): The target z-variables, as many as
            ``operation.num_zvars``. Defaults to ``()``.

    Raises:
        TypeError: If ``operation`` is not an :class:`Operation`, or a target
            group is not a tuple.
        ValueError: If a target group has the wrong length, an index is
            negative, or a qubit is repeated. Repeated bits or z-variables are
            refused too, unless the operation allows them.

    Examples:
        >>> from mimiqcircuits import *
        >>> Instruction(GateX(), (0,))
        X @ q[0]
        >>> # a measurement targets one qubit and one classical bit
        >>> Instruction(Measure(), (2,), (0,))
        M @ q[2], c[0]
        >>> # consecutive targets are shown as a range
        >>> Instruction(Barrier(4), (0, 1, 2, 3))
        Barrier @ q[0:3]
    """

    _operation = None
    _qubits = None
    _bits = None
    _zvars = None

    def __init__(self, operation, qubits=None, bits=None, zvars=None):
        # Every check below runs for each instruction a circuit builds, so
        # they avoid ABCMeta isinstance calls and short-circuit on the common
        # one- and two-target cases. The checks, their order and their
        # messages are part of the interface.
        if qubits is None:
            qubits = ()
        elif not isinstance(qubits, tuple):
            raise TypeError(
                f"Target qubits should be given in a tuple of integers. Given {qubits} of type {type(qubits)}."
            )

        if bits is None:
            bits = ()
        elif not isinstance(bits, tuple):
            raise TypeError(
                f"Target bits should be given in a tuple of integers. Given {bits} of type {type(bits)}."
            )

        if zvars is None:
            zvars = ()
        elif not isinstance(zvars, tuple):
            raise TypeError(
                f"Target z-variables should be given in a tuple of integers. Given {zvars} of type {type(zvars)}."
            )

        if type(operation) not in _OPERATION_TYPES:
            if not isinstance(operation, Operation):
                raise TypeError(
                    f"Operation must be a subclass of Operation. Given {operation} of type f{type(operation)}"
                )
            _OPERATION_TYPES.add(type(operation))

        nq = len(qubits)
        nb = len(bits)
        nz = len(zvars)

        if nq > 1 and len(set(qubits)) != nq:
            raise ValueError("Duplicated qubit target in instruction")

        if nb > 1 and not operation.allow_bit_aliasing() and len(set(bits)) != nb:
            raise ValueError("Duplicated classical bit target in instruction")

        if nz > 1 and not operation.allow_zvar_aliasing() and len(set(zvars)) != nz:
            raise ValueError("Duplicated z-variables target in instruction")

        for qi in qubits:
            if qi < 0:
                raise ValueError("Qubit target index cannot be negative")

        for bi in bits:
            if bi < 0:
                raise ValueError("Bit target index cannot be negative")

        for z in zvars:
            if z < 0:
                raise ValueError("Z-variable target index cannot be negative")

        if nq != operation.num_qubits:
            raise ValueError(
                f"Wrong number of target qubits for operation {operation} wanted  {operation.num_qubits}, given {nq}"
            )

        if nb != operation.num_bits:
            raise ValueError(
                f"Wrong number of target bits for operation {operation} wanted  {operation.num_bits}, given {nb}"
            )

        if nz != operation.num_zvars:
            raise ValueError(
                f"Wrong number of z-targets for operation {operation} wanted  {operation.num_zvars}, given {nz}"
            )

        self._operation = operation
        self._qubits = qubits
        self._bits = bits
        self._zvars = zvars

    @property
    def operation(self):
        """The operation applied by the instruction (read-only)."""
        return self._operation

    @operation.setter
    def operation(self, _):
        raise AttributeError("operation is a read-only attribute")

    @property
    def qubits(self):
        """The target qubits, as a tuple of int (read-only)."""
        return self._qubits

    @qubits.setter
    def qubits(self, _):
        raise AttributeError("qubits is a read-only attribute")

    @property
    def bits(self):
        """The target classical bits, as a tuple of int (read-only)."""
        return self._bits

    @bits.setter
    def bits(self, _):
        raise AttributeError("bits is a read-only attribute")

    @property
    def zvars(self):
        """The target z-variables, as a tuple of int (read-only)."""
        return self._zvars

    @zvars.setter
    def zvars(self, _):
        raise AttributeError("zvars is a read-only attribute")

    def __repr__(self):
        return str(self)

    def get_operation(self):
        """Return the operation of the instruction. Same as :attr:`operation`."""
        return self._operation

    def get_qubits(self):
        """Return the target qubits. Same as :attr:`qubits`."""
        return self._qubits

    def get_bits(self):
        """Return the target classical bits. Same as :attr:`bits`."""
        return self._bits

    def get_zvars(self):
        """Return the target z-variables. Same as :attr:`zvars`."""
        return self._zvars

    def num_qubits(self):
        """Return the number of target qubits."""
        return len(self._qubits)

    def num_bits(self):
        """Return the number of target classical bits."""
        return len(self._bits)

    def num_zvars(self):
        """Return the number of target z-variables."""
        return len(self._zvars)

    def getparams(self):
        """Return the parameters of the operation, as a list.

        See :meth:`Operation.getparams`.
        """
        return self.operation.getparams()

    def listvars(self):
        """List the symbolic variables of the operation of the instruction.

        See :meth:`Operation.listvars`.
        """
        return self.operation.listvars()

    def asciiwidth(self):
        return self.operation.asciiwidth(self._qubits, self._bits)

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return False
        return (
            (self.operation == other.operation)
            and (self.qubits == other.qubits)
            and (self.bits == other.bits)
            and (self.zvars == other.zvars)
        )

    def inverse(self):
        """Return the instruction applying the inverse operation on the same targets.

        Raises:
            TypeError: If the operation has no inverse, such as a measurement.
            NotImplementedError: If the operation is a Kraus channel.

        Examples:
            >>> from mimiqcircuits import *
            >>> Instruction(GateS(), (1,)).inverse()
            S† @ q[1]
        """
        return Instruction(self.operation.inverse(), self.qubits, self.bits)

    def isunitary(self):
        """Return ``True`` if the operation of the instruction is unitary."""
        return self.operation.isunitary()

    def copy(self):
        """Return a shallow copy of the instruction.

        The copy shares its operation with the original. Use :meth:`deepcopy`
        to copy the operation too.
        """
        return copy.copy(self)

    def deepcopy(self):
        """Return a deep copy of the instruction, operation included."""
        return copy.deepcopy(self)

    def _decompose(self, circ):
        return self.operation._decompose(circ, self.qubits, self.bits, self.zvars)

    def decompose(self):
        """Return a circuit with one decomposition step of the operation.

        The operation is replaced by the instructions of its own
        decomposition, placed on the instruction's targets. Operations that
        have no simpler form, such as ``GateU`` or ``GateCX``, come back
        unchanged. Use :meth:`Circuit.decompose` to decompose to a basis.

        Returns:
            Circuit: The decomposed instruction.

        Examples:
            >>> from mimiqcircuits import *
            >>> Instruction(GateCZ(), (0, 1)).decompose()
            2-qubit circuit with 3 instructions:
            ├── H @ q[1]
            ├── CX @ q[0], q[1]
            └── H @ q[1]
            <BLANKLINE>
        """
        from mimiqcircuits.circuit import Circuit
        return self._decompose(Circuit())

    def evaluate(self, d):
        """Return the instruction with symbols replaced by values from ``d``.

        Symbols not in ``d`` stay symbolic. When nothing changes, the
        instruction itself is returned.

        Args:
            d (dict): Map from symbols to values.

        Returns:
            Instruction: The evaluated instruction.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import Symbol
            >>> theta = Symbol("theta")
            >>> Instruction(GateRX(theta), (0,)).evaluate({theta: 0.5})
            RX(0.5) @ q[0]
        """
        op = self.operation.evaluate(d)
        if op is self.operation:
            return self
        return Instruction(op, self.qubits, self.bits, self.zvars)

    def matrix(self, nq: int = None):
        """Return the matrix of the instruction on the full register of ``nq`` qubits.

        The matrix of the operation is placed on the target qubits and
        extended with the identity on the other ones.

        Args:
            nq: The number of qubits of the register. Defaults to one more
                than the largest target qubit.

        Returns:
            The ``2**nq`` by ``2**nq`` matrix, as a symengine matrix.

        Raises:
            ValueError: If ``nq`` is not given and the instruction has no
                target qubits.

        Examples:
            >>> from mimiqcircuits import *
            >>> # X on qubit 0 of a 2-qubit register: X ⊗ I
            >>> Instruction(GateX(), (0,)).matrix(2)
            [0, 0, 1.0, 0]
            [0, 0, 0, 1.0]
            [1.0, 0, 0, 0]
            [0, 1.0, 0, 0]
            <BLANKLINE>
        """
        op = self.get_operation()
        qubits = self.get_qubits()

        if nq is None:
            if len(qubits) == 0:
                raise ValueError("Cannot infer nq: instruction has no qubit targets.")
            nq = max(qubits) + 1

        return reorder_qubits_matrix(op.matrix(), qubits, nq)

    def __str__(self):
        compact = False
        if hasattr(self.operation, "format_with_targets"):
            # Custom rendering — do not append @ targets
            return self.operation.format_with_targets(
                self.qubits, self.bits, self.zvars
            )

        op = str(self.operation)
        nq = len(self.qubits)
        nb = len(self.bits)
        nz = len(self.zvars)
        space = "" if compact else " "
        targets = ""
        if nq != 0 or nb != 0 or nz != 0:
            targets = f"{space}@{space}"

            if nq != 0:
                q_partition = _partition(
                    self.get_qubits(), np.cumsum(self.operation.qregsizes)
                )
                q_targets = f",{space}".join(
                    f"q{_string_with_square(_find_unit_range(x), ',')}"
                    for x in q_partition
                )
                targets += f",".join(q_targets.split(","))

            if nb != 0:
                if nq != 0:  # If qubits exist, add a separator before classical bits
                    targets += f", "
                c_partition = _partition(
                    self.get_bits(), np.cumsum(self.operation.cregsizes)
                )
                c_targets = "".join(  # Concatenate classical bits without extra commas
                    f"c{_string_with_square(_find_unit_range(x), ',')}"
                    for x in c_partition
                )
                targets += c_targets

            if nz != 0:
                if (
                    nq != 0 or nb != 0
                ):  # Add a separator before z-vars only if qubits or bits exist
                    targets += f", "
                z_partition = _partition(
                    self.get_zvars(), np.cumsum(self.operation.zregsizes)
                )
                z_targets = "".join(  # Concatenate z-vars without extra commas
                    f"z{_string_with_square(_find_unit_range(x), ',')}"
                    for x in z_partition
                )
                targets += z_targets

        return f"{op}{targets}"


__all__ = ["Instruction"]
