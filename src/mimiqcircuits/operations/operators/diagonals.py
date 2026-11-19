#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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

from mimiqcircuits.operations.operator import AbstractOperator
import symengine as se
from numbers import Real
from mimiqcircuits._sympycheck import _is_basic


class DiagonalOp(AbstractOperator):
    r"""One-qubit diagonal operator.

    Matrix Representation:
        .. math::
            \begin{pmatrix}
                a & 0 \\
                0 & b
            \end{pmatrix}

    The operator does not have to be unitary. Use it in an
    :class:`ExpectationValue` or as a Kraus operator.

    Args:
        a: The top-left entry, a real number or a symbolic expression.
        b: The bottom-right entry, a real number or a symbolic expression.

    Raises:
        ValueError: If ``a`` or ``b`` is neither a real number nor a
            symbolic expression.

    See Also:
        :class:`Operator`, :class:`Projector0`, :class:`Projector1`

    Examples:
        >>> from mimiqcircuits import *
        >>> DiagonalOp(1, 0.5)
        D(1, 0.5)

        >>> # expectation value on qubit 1, stored in z-variable 2
        >>> c = Circuit()
        >>> c.push(ExpectationValue(DiagonalOp(1, 0.5)), 1, 2)
        2-qubit, 3-zvar circuit with 1 instruction:
        └── ⟨D(1, 0.5)⟩ @ q[1], z[2]
        <BLANKLINE>
    """

    _name = "D"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a, b):
        if not all(isinstance(v, Real) or _is_basic(v) for v in (a, b)):
            raise ValueError("a and b must be real numbers or symbolic expressions")
        self.a = a
        self.b = b

        super().__init__()
        self._parnames = ("a", "b")

    def _matrix(self):
        mat = se.Matrix([[self.a, 0], [0, self.b]])
        return mat

    def __str__(self):
        return f"{self._name}{self.a, self.b}"

    @property
    def parnames(self):
        return self._parnames

    def opsquared(self):
        r"""Return :math:`O^\dagger O`, the diagonal operator with entries :math:`|a|^2` and :math:`|b|^2`."""
        return DiagonalOp(abs(self.a) ** 2, abs(self.b) ** 2)


__all__ = ["DiagonalOp"]
