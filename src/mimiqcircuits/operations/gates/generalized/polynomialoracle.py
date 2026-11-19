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
"""Polynomial Oracle gate."""

from symengine import pi

from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.gates.standard.id import GateID
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateZ
from mimiqcircuits.operations.gates.standard.u import GateU
import mimiqcircuits.lazy as lz


class PolynomialOracle(Gate):
    r"""Quantum oracle for a polynomial of two registers.

    Applies a :math:`\pi` phase to every basis state :math:`|x\rangle|y\rangle`
    with :math:`a xy + b x + c y + d = 0`, where :math:`x` and :math:`y` are
    the integer values of the two registers. Each register is little-endian:
    its first qubit is the least significant bit. The gate is its own inverse.

    ``PolynomialOracle(a, b, c, d)`` with four arguments returns a lazy
    expression that takes the two register sizes later.

    The oracle has no matrix representation. It decomposes into one
    multi-controlled Z per solution, so the decomposition is only practical
    when there are few solutions. A decomposition without solutions is a
    single ``GateID`` on the last qubit, which keeps the oracle's width.

    Args:
        nx (int): Number of qubits in the :math:`x` register.
        ny (int): Number of qubits in the :math:`y` register.
        a (int): Coefficient of :math:`xy`.
        b (int): Coefficient of :math:`x`.
        c (int): Coefficient of :math:`y`.
        d (int): Constant term.

    Raises:
        ValueError: If a coefficient is not an integer, or the number of
            arguments is neither 4 nor 6.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(PolynomialOracle(5, 5, 1, 2, 3, 4), *range(1, 11))
        11-qubit circuit with 1 instruction:
        └── PolynomialOracle(1, 2, 3, 4) @ q[1:5], q[6:10]
        <BLANKLINE>
        >>> # lazy form: the register sizes are given later
        >>> PolynomialOracle(1, 2, 3, 4)(5, 5)
        PolynomialOracle(1, 2, 3, 4)
    """

    _name = "PolynomialOracle"
    _num_qregs = 2
    _num_qubits = None

    def __init__(self, nx, ny, a, b, c, d):
        if not all(isinstance(x, int) for x in [a, b, c, d]):
            raise ValueError("Coefficients a, b, c, d must be integers.")

        super().__init__()

        self._num_qubits = nx + ny
        self._qregsizes = [nx, ny]
        self._params = [nx, ny, a, b, c, d]
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __new__(cls, *args):
        if len(args) == 4:
            return lz.LazyExpr(PolynomialOracle, lz.LazyArg(), lz.LazyArg(), *args)
        elif len(args) == 6:
            return object.__new__(cls)
        else:
            raise ValueError("Invalid number of arguments.")

    def inverse(self):
        return self

    def _matrix(self):
        raise NotImplementedError(
            "Matrix representation for PolynomialOracle is not implemented."
        )

    def supports_canonical_rewrite(self):
        """Return ``True``: the oracle decomposes into multi-controlled Z gates."""
        return True

    def _decompose(self, circ, qubits, bits, _):
        # For each value u of the smaller register the polynomial is s*v + t in
        # the other one, so the solutions come without looping over all 2^n states.
        nx, ny = self._qregsizes
        small, large = list(qubits[:nx]), list(qubits[nx:])
        a, b, c, d = self.a, self.b, self.c, self.d
        if nx > ny:  # f is symmetric under swapping x <-> y together with b <-> c
            small, large, b, c = large, small, c, b

        for u in range(2 ** len(small)):
            s, t = a * u + c, b * u + d
            if s == 0:
                if t == 0:  # every v is a solution
                    _flipstate(circ, small, u, qubits)
            elif t % s == 0 and 0 <= -t // s < 2 ** len(large):
                v = -t // s
                _flipstate(circ, small + large, u + (v << len(small)), qubits)
        if circ.num_qubits() <= max(qubits):
            circ.push(GateID(), max(qubits))
        return circ

    def __str__(self):
        return f"PolynomialOracle{self.a, self.b, self.c, self.d}"


def _flipstate(circ, qs, val, qubits):
    """Phase -1 on the basis states where qubits ``qs`` hold the little-endian value ``val``."""
    if not qs:  # every state flips
        return circ.push(GateU(0, 0, 0, pi), qubits[0])
    zeros = [q for i, q in enumerate(qs) if not (val >> i) & 1]
    for q in zeros:
        circ.push(GateX(), q)
    circ.push(GateZ() if len(qs) == 1 else Control(len(qs) - 1, GateZ()), *qs)
    for q in zeros:
        circ.push(GateX(), q)
    return circ


__all__ = ["PolynomialOracle"]
