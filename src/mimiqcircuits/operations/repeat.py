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
"""Repeat operation."""

import copy
from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.lazy import LazyArg, LazyExpr


class Repeat(Operation):
    r"""Operation applied several times in a row on the same targets.

    ``Repeat(n, op)`` applies ``op`` ``n`` times on the same qubits, bits and
    z-variables. Unlike :class:`Power`, it works for any operation, not only
    gates, and keeps the repetitions explicit: decomposing it gives ``n``
    copies of ``op``.

    The ``repeat`` method of a gate and the :func:`repeat` function are
    shorthands for it.

    Args:
        repeats (int): Number of repetitions, zero or more.
        op (Operation): The operation to repeat.

    Raises:
        ValueError: If ``repeats`` is not a non-negative ``int``.
        TypeError: If ``op`` is not an operation.

    Examples:

        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> # five X gates in a row
        >>> Repeat(5, GateX())
        ∏⁵ X

        >>> Repeat(3, GateRX(Symbol("x")))
        ∏³ RX(x)

        >>> # nested repeats are kept as they are
        >>> Repeat(2, Repeat(3, GateX()))
        ∏² ∏³ X

        >>> # without a count, repeat() returns a lazy expression
        >>> Parallel(2, GateH()).repeat()
        lazy repeat(?, ⨷ ² H)

        >>> Parallel(2, GateH()).repeat()(10)
        ∏¹⁰ ⨷ ² H

        >>> c = Circuit().push(Repeat(2, GateX()), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── ∏² X @ q[0]
        <BLANKLINE>

        >>> # decomposing gives the individual repetitions
        >>> Repeat(2, GateX()).decompose()
        1-qubit circuit with 2 instructions:
        ├── X @ q[0]
        └── X @ q[0]
        <BLANKLINE>

        >>> Repeat(3, GateSWAP()).decompose()
        2-qubit circuit with 3 instructions:
        ├── SWAP @ q[0:1]
        ├── SWAP @ q[0:1]
        └── SWAP @ q[0:1]
        <BLANKLINE>
    """

    _name = "Repeat"

    def __init__(self, repeats, op):
        if not isinstance(repeats, int) or repeats < 0:
            raise ValueError(
                "Invalid number of repetitions, must be a non-negative integer."
            )
        if not isinstance(op, Operation):
            raise TypeError("Repeat requires a valid Operation to wrap.")

        super().__init__()
        self.repeats = repeats
        self.op = op

        self._num_qubits = op.num_qubits
        self._num_bits = op.num_bits
        self._num_zvars = op.num_zvars
        self._num_qregs = op.num_qregs
        self._num_cregs = op.num_cregs
        self._num_zvars = op.num_zvars
        self._qregsizes = op.qregsizes
        self._cregsizes = op.cregsizes
        self._zregsizes = op.zregsizes
        self._parnames = op.parnames

    def get_operation(self):
        """Return the repeated operation, :attr:`op`."""
        return self.op

    def getparam(self, name):
        """Return the parameter ``name`` of the repeated operation."""
        return self.op.getparam(name)

    def getparams(self):
        """Return the parameters of the repeated operation."""
        return self.op.getparams()

    def power(self, pow):
        """Return the repetitions of ``op`` raised to ``pow``.

        Args:
            pow: The exponent.

        Returns:
            Repeat: ``Repeat(repeats, op.power(pow))``.
        """
        return Repeat(self.repeats, self.op.power(pow))

    def parallel(self, repeat):
        """Return the repetitions of ``repeat`` parallel copies of ``op``.

        Args:
            repeat: Number of parallel copies.

        Returns:
            Repeat: ``Repeat(repeats, op.parallel(repeat))``.
        """
        return Repeat(self.repeats, self.op.parallel(repeat))

    def control(self, repeat):
        """Return the repetitions of ``op`` controlled by ``repeat`` qubits.

        Args:
            repeat: Number of control qubits.

        Returns:
            Repeat: ``Repeat(repeats, op.control(repeat))``.
        """
        return Repeat(self.repeats, self.op.control(repeat))

    def inverse(self):
        """Return the repetitions of the inverse of ``op``.

        Returns:
            Repeat: ``Repeat(repeats, op.inverse())``. Fails when ``op``
            has no inverse.
        """
        return Repeat(self.repeats, self.op.inverse())

    def iswrapper(self):
        return True

    def supports_canonical_rewrite(self):
        return True

    def evaluate(self, d):
        if not self.is_symbolic():
            return self
        return Repeat(self.repeats, self.op.evaluate(d))

    def _decompose(self, circ, qubits, bits, zvars):
        for _ in range(self.repeats):
            circ.push(self.op, *qubits, *bits, *zvars)
        return circ

    def __str__(self):
        superscript = "⁰¹²³⁴⁵⁶⁷⁸⁹"
        repeat_str = "".join(superscript[int(d)] for d in str(self.repeats))
        return f"∏{repeat_str} {self.op}"

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def format_with_targets(self, qubits, bits, zvars):
        opname = str(self)
        if hasattr(self.get_operation(), "format_with_targets"):
            return f"{opname} @ {self.get_operation().format_with_targets(qubits, bits, zvars).split('@')[1]}"
        parts = []
        if qubits:
            parts.append(f"q[{','.join(map(str, qubits))}]")
        if bits:
            parts.append(f"c[{','.join(map(str, bits))}]")
        if zvars:
            parts.append(f"z[{','.join(map(str, zvars))}]")
        return f"{opname} @ {', '.join(parts)}"


def repeat(*args):
    """Build a ``Repeat`` operation, or a lazy one that waits for its count.

    ``repeat(n, op)`` is ``Repeat(n, op)``. ``repeat(op)`` returns a lazy
    expression; calling it with ``n`` builds ``Repeat(n, op)``.

    Args:
        *args: ``(n, op)`` or ``(op,)``.

    Returns:
        Repeat or LazyExpr: The operation, or the lazy expression.

    Raises:
        TypeError: If called with no arguments or with more than two.

    Examples:
        >>> from mimiqcircuits import *
        >>> repeat(3, GateX())
        ∏³ X
        >>> repeat(GateX())(3)
        ∏³ X
    """
    if len(args) == 1:
        op = args[0]
        return LazyExpr(repeat, LazyArg(), op)
    elif len(args) == 2:
        num_repeats, op = args
        if isinstance(op, LazyExpr):
            return LazyExpr(repeat, num_repeats, op)
        return Repeat(num_repeats, op)
    else:
        raise TypeError("repeat expects 1 or 2 arguments")


__all__ = ["Repeat", "repeat"]
