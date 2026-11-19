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
"""Parallel gate operation."""

import numpy as np
from functools import reduce
from mimiqcircuits.operations.gates.gate import Gate
import mimiqcircuits.lazy as lz
from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.power import Power


def to_superscript(number):
    superscript_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(number).translate(superscript_map)


class Parallel(Gate):
    """Copies of a gate applied in parallel on separate qubits.

    ``Parallel(n, gate)`` applies ``gate`` ``n`` times, each copy on its own
    group of qubits: the first ``gate.num_qubits`` targets go to the first
    copy, the next ones to the second, and so on. Its matrix is the tensor
    product of ``n`` copies of the gate's matrix.

    The ``parallel`` method of a gate is the usual way to build one.

    Args:
        num_repeats: Number of copies, at least 2.
        op: The gate to repeat.

    Raises:
        TypeError: If ``op`` is not a gate.
        ValueError: If ``num_repeats`` is less than 2.

    Examples:
        >>> from mimiqcircuits import *
        >>> # three X gates on qubits 1, 2 and 3 as a single instruction
        >>> c = Circuit()
        >>> c.push(Parallel(3, GateX()), 1, 2, 3)
        4-qubit circuit with 1 instruction:
        └── ⨷ ³ X @ q[1], q[2], q[3]
        <BLANKLINE>
        >>> # a two-qubit gate in parallel: CX on (0, 1) and on (2, 3)
        >>> Parallel(2, GateCX()).decompose()
        4-qubit circuit with 2 instructions:
        ├── CX @ q[0], q[1]
        └── CX @ q[2], q[3]
        <BLANKLINE>
    """

    _name = "Parallel"
    _num_qubits = None
    _num_bits = 0
    _num_repeats = None
    _op = None

    def __init__(self, num_repeats, op: Gate):
        if not isinstance(op, (Gate)):
            raise TypeError(f"{op.__class__.__name__} cannot be Paralleled.")

        if num_repeats < 2:
            raise ValueError(
                f"Parallel operations must have at least two repeats. If one, just use {op}."
            )

        super().__init__()
        self._num_qubits = op.num_qubits * num_repeats
        self._num_repeats = num_repeats
        self._op = op
        self._qregsizes = [1] * self._num_qubits
        self._num_qregs = self._num_qubits

    def _matrix(self):
        op_matrix = self.op.matrix()
        return reduce(np.kron, [op_matrix] * (self._num_repeats))

    def _matrix_numeric(self, *params):
        op_matrix = self.op.unwrappedmatrix()
        return reduce(np.kron, [op_matrix] * self._num_repeats)

    @property
    def num_repeats(self):
        """Number of parallel copies. Read-only."""
        return self._num_repeats

    @num_repeats.setter
    def num_repeats(self, value):
        raise ValueError("Cannot set num_repeats. Read only parameter.")

    @property
    def op(self):
        """The repeated gate. Read-only."""
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    def iswrapper(self):
        return True

    def inverse(self):
        """Return the parallel copies of the inverse gate."""
        return Parallel(self.num_repeats, self.op.inverse())

    def power(self, *args):
        """Raise each parallel copy to a power.

        Args:
            *args: The exponent. When omitted, a lazy expression is returned
                that takes the exponent later.

        Returns:
            Parallel: ``Parallel(num_repeats, Power(op, p))``, or a lazy
            expression when called without arguments.

        Raises:
            ValueError: If more than one argument is given.
        """
        if len(args) == 0:
            return lz.power(self)
        elif len(args) == 1:
            pwr = args[0]
            return Power(self.op, pwr).parallel(self.num_repeats)
        else:
            raise ValueError("Invalid number of arguments.")

    def control(self, *args):
        if len(args) == 0:
            return lz.control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return Control(num_controls, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def parallel(self, *args):
        """Repeat in parallel again, multiplying the number of copies.

        Args:
            *args: The extra repetition factor ``n``. When omitted, a lazy
                expression is returned that takes it later.

        Returns:
            Parallel: ``Parallel(num_repeats * n, op)``, or a lazy
            expression when called without arguments.

        Raises:
            ValueError: If more than one argument is given.

        Examples:
            >>> from mimiqcircuits import *
            >>> Parallel(2, GateH()).parallel(3)
            ⨷ ⁶ H
        """
        if len(args) == 0:
            return lz.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return Parallel(self.num_repeats * num_repeats, self.op)
        else:
            raise ValueError("Invalid number of arguments.")

    def _decompose(self, circ, qubits, bits, zvars):
        nq = self.op.num_qubits
        for i in range(self.num_repeats):
            q = [qubits[j] for j in range(i * nq, (i + 1) * nq)]
            circ.push(self.op, *q)
        return circ

    def __str__(self):
        tail = f"⨷ {to_superscript(self.num_repeats)}"
        return f"{tail} {self.op}"

    def evaluate(self, d):
        if not self.is_symbolic():
            return self
        return self.op.evaluate(d).parallel(self.num_repeats)

    def gettypekey(self):
        return (Parallel, self.num_repeats, self.op.gettypekey())

    def get_operation(self):
        return self.op


# export operations
__all__ = ["Parallel"]
