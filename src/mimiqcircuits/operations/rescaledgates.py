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
"""Rescaled gate operation."""

import symengine as se
from mimiqcircuits._sympycheck import _is_basic
from typing import TypeVar, Generic
from mimiqcircuits.operations.operator import AbstractOperator
from mimiqcircuits.operations.gates.gate import Gate

T = TypeVar("T", bound=Gate)  # T is a type variable bound to Gate


class RescaledGate(AbstractOperator, Generic[T]):
    r"""A gate multiplied by a scalar factor ``p``.

    Represents the operator :math:`p\,U`, where :math:`U` is the matrix of
    ``gate``. For ``p < 1`` the result is not unitary, so a ``RescaledGate``
    is an operator, not a gate: it cannot be pushed to a circuit directly or
    inverted. It is used where an operator is expected, for example in
    :class:`ExpectationValue`, or as a Kraus operator of
    :class:`MixedUnitary`, where ``RescaledGate(U, s)`` stands for :math:`U`
    applied with probability :math:`s^2`.

    Args:
        gate (Gate): The gate to rescale.
        p: The factor, a number in :math:`[0, 1]` or a symbolic expression.
            Symbolic factors are not range checked.

    Raises:
        TypeError: If ``gate`` is not a :class:`Gate`.
        ValueError: If a numeric ``p`` is outside :math:`[0, 1]`.

    Examples:
        >>> from mimiqcircuits import *
        >>> rg = RescaledGate(GateH(), 0.5)
        >>> rg
        0.5*H
        >>> rg.get_scale()
        0.5
        >>> rg.matrix()
        [0.353553390593274, 0.353553390593274]
        [0.353553390593274, -0.353553390593274]
        <BLANKLINE>
        >>> # rescaling again multiplies the factors
        >>> rg.rescale(0.8)
        0.4*H
        >>> # use it as the observable of an expectation value
        >>> Circuit().push(ExpectationValue(RescaledGate(GateZ(), 0.5)), 0, 0)
        1-qubit, 1-zvar circuit with 1 instruction:
        └── ⟨0.5*Z⟩ @ q[0], z[0]
        <BLANKLINE>
    """

    _name = "RescaledGate"

    def __init__(self, gate: T, p):
        if not isinstance(gate, Gate):
            raise TypeError(
                f"Expected gate to be an instance of Gate, got {type(gate).__name__}"
            )

        super().__init__()
        self._num_bits = gate.num_bits
        self._num_qubits = gate.num_qubits
        self._num_zvars = gate.num_zvars
        self._num_cregs = gate._num_cregs
        self._num_qregs = gate._num_qregs
        self._num_zregs = gate._num_zregs
        self._cregsizes = gate._cregsizes
        self._qregsizes = gate._qregsizes
        self._zregsizes = gate._zregsizes

        if not _is_basic(p) and (float(p) < 0 or float(p) > 1):
            raise ValueError("Value of p must be between 0 and 1.")
        self.gate = gate
        self.p = p

    def __repr__(self):
        return self.__str__()

    def iswrapper(self):
        """Return ``True``: a rescaled gate wraps another gate."""
        return True

    def getparams(self):
        """Return ``[p, *gate.getparams()]``."""
        return [self.p, *self.gate.getparams()]

    def get_operation(self):
        """Return the wrapped gate."""
        return self.gate

    def get_param(self, name):
        """Return the factor for ``name == "p"``, otherwise look ``name`` up on the wrapped gate."""
        if name == "p":
            return self.p
        return self.gate.getparam(name)

    def _matrix(self, *args):
        return self.p * self.gate._matrix(*args)

    def _matrix_numeric(self, *params):
        return complex(self.p) * self.gate.unwrappedmatrix()

    def opsquared(self):
        r"""Return :math:`(pU)^\dagger (pU) = |p|^2 I` as a rescaled identity."""
        from mimiqcircuits.operations.gates.standard.id import GateID
        from mimiqcircuits.operations.parallel import Parallel

        n = self.num_qubits
        ident = GateID() if n == 1 else Parallel(n, GateID())
        return RescaledGate(ident, abs(self.p) ** 2)

    def get_scale(self):
        """Return the factor ``p``."""
        return self.p

    def rescale(self, p):
        """Return a new ``RescaledGate`` of the same gate with factor ``self.p * p``."""
        return RescaledGate(self.gate, self.p * p)

    def rescale_inplace(self, p):
        """Multiply the factor by ``p`` in place and return ``self``.

        The product is not range checked.
        """
        self.p *= p
        return self

    def evaluate(self, d):
        """Substitute symbols in the factor and in the wrapped gate.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            RescaledGate: ``self`` if nothing is symbolic, otherwise a new
            rescaled gate.
        """
        if not self.is_symbolic():
            return self
        return RescaledGate(self.gate.evaluate(d), evaluate(self.p, d))

    def __str__(self):
        mult = "*" if getattr(self, "compact", True) else " * "
        return f"{self.p}{mult}{self.gate}"


def evaluate(param, d):
    """Return ``param`` with the symbols in ``d`` substituted; non-symbolic values are returned unchanged."""
    return param.subs(d) if _is_basic(param) else param
