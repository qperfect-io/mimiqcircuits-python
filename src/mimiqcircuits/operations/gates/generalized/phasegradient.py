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
"""Phase gradient gate."""

import mimiqcircuits.operations.gates.gate as mcg
from mimiqcircuits.operations.gates.standard.phase import GateP
import mimiqcircuits.lazy as lz
from symengine import pi
from mimiqcircuits.operations.power import _scale_by_exponent


class PhaseGradient(mcg.Gate):
    r"""Phase gradient gate on ``num_qubits`` qubits.

    Each computational basis state :math:`|k\rangle` gets a phase
    proportional to its integer value :math:`k`, with the first qubit as the
    most significant bit:

    .. math::
        \operatorname{PhaseGradient} =
        \sum_{k=0}^{2^n-1} \mathrm{e}^{i \frac{2 \pi}{2^n} k} |k\rangle\langle k|

    The gate decomposes into one :class:`GateP` per qubit. ``PhaseGradient()``
    with no arguments returns a lazy expression that takes the number of
    qubits later. The gate has no matrix representation: :meth:`matrix`
    raises ``NotImplementedError``.

    Args:
        num_qubits (int): Number of qubits.

    Raises:
        ValueError: If more than one argument is given.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(PhaseGradient(2), 9, 8)
        10-qubit circuit with 1 instruction:
        └── PhaseGradient @ q[9,8]
        <BLANKLINE>
        >>> PhaseGradient(3).decompose()
        3-qubit circuit with 3 instructions:
        ├── P((1/4)*pi) @ q[2]
        ├── P((1/2)*pi) @ q[1]
        └── P(pi) @ q[0]
        <BLANKLINE>
    """

    _name = "PhaseGradient"
    _num_qregs = 1

    def __init__(self, num_qubits):
        super().__init__()

        self._num_qubits = num_qubits
        self._qregsizes = [
            num_qubits,
        ]
        self._params = [num_qubits]

        self._num_bits = 0

    def __new__(cls, *args):
        if len(args) == 0:
            return lz.LazyExpr(PhaseGradient, lz.LazyArg())
        elif len(args) == 1:
            return object.__new__(cls)
        else:
            raise ValueError("Invalid number of arguments.")

    def _decompose(self, circ, qubits, bits, _):
        return self._decompose_power(circ, qubits, 1)

    def _decompose_power(self, circ, qubits, exponent):
        for i in range(len(qubits) - 1, -1, -1):
            phase = _scale_by_exponent(pi, exponent) / 2**i
            circ.push(GateP(phase), qubits[i])
        return circ

    def _matrix(self):
        raise NotImplementedError(
            "Matrix representation for PhaseGradient is not implemented."
        )


__all__ = ["PhaseGradient"]
