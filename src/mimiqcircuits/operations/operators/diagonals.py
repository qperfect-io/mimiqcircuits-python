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

import mimiqcircuits as mc
import symengine as se


class DiagonalOp(mc.AbstractOperator):
    r"""One-qubit diagonal operator.

    The corresponding matrix is given by:

    **Matrix Representation**

    .. math::
        \begin{pmatrix}
            a & 0 \\
            0 & b
        \end{pmatrix}

    where `a` and `b` are complex numbers.

    See Also:
        :class:`Operator`, :class:`Projector0`, :class:`Projector1`

    Parameters:
        a (complex): The top-left entry of the diagonal matrix.
        b (complex): The bottom-right entry of the diagonal matrix.

    Examples:
        >>> from mimiqcircuits import *
        >>> op = DiagonalOp(1, 0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(DiagonalOp(1, 0.5)), 1, 2)
        2-qubit, 3-zvar circuit with 1 instructions:
        └── ⟨D(1, 0.5)⟩ @ q[1], z[2]
        <BLANKLINE>
    """

    _name = "D"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a, b):
        if not isinstance(a, (float, int)) or not isinstance(b, (float, int)):
            raise ValueError("a and b must be float or int")
        self.a = a
        self.b = b

        super().__init__()
        self._parnames = ("a", "b")

    def _matrix(self):
        mat = se.Matrix([[self.a, 0], [0, self.b]])
        return mat

    def __str__(self):
        return f"{self._name}{self.a,self.b}"

    @property
    def parnames(self):
        return self._parnames

    def rescale(self, scale):
        """Return a new rescaled DiagonalOp operator."""
        return DiagonalOp(self.a * scale, self.b * scale)

    def rescale_inplace(self, scale):
        """In-place rescaling of the DiagonalOp operator."""
        self.a *= scale
        self.b *= scale
        return self

    def opsquared(self):
        """Return the operator with each parameter squared."""
        return DiagonalOp(abs(self.a) ** 2, abs(self.b) ** 2)


__all__ = ["DiagonalOp"]
