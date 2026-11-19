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
from mimiqcircuits.operations.operators.projectors import Projector0, Projector1


class SigmaMinus(AbstractOperator):
    r"""One-qubit operator corresponding to :math:`|0 \rangle\langle 1|`.

    Matrix Representation:
        .. math::
            \begin{pmatrix}
                0 & a\\
                0 & 0
            \end{pmatrix}

    The factor ``a`` scales the operator, for example to add a phase. The
    operator is not unitary, so it cannot be applied to a circuit directly;
    use it in an :class:`ExpectationValue` or as a Kraus operator.

    Args:
        a (complex): Scaling factor. Defaults to 1.

    See Also:
        :class:`SigmaPlus`

    Examples:
        >>> from mimiqcircuits import *
        >>> SigmaMinus()
        SigmaMinus(1)

        >>> SigmaMinus(0.5)
        SigmaMinus(0.5)

        >>> # expectation value on qubit 1, stored in z-variable 1
        >>> c = Circuit()
        >>> c.push(ExpectationValue(SigmaMinus()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instruction:
        └── ⟨SigmaMinus(1)⟩ @ q[1], z[1]
        <BLANKLINE>
    """

    _name = "SigmaMinus"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    @property
    def parnames(self):
        return self._parnames

    @property
    def num_qubits(self):
        return self._num_qubits

    def _matrix(self):
        return se.Matrix([[0, self.a], [0, 0]])

    def opsquared(self):
        r"""Return :math:`O^\dagger O = |a|^2 |1\rangle\langle 1|` as a :class:`Projector1`."""
        return Projector1(abs(self.a) ** 2)

    def __str__(self):
        return f"{self.opname()}({self.a})"


class SigmaPlus(AbstractOperator):
    r"""One-qubit operator corresponding to :math:`|1 \rangle\langle 0|`.

    Matrix Representation:
        .. math::
            \begin{pmatrix}
                0 & 0\\
                a & 0
            \end{pmatrix}

    The factor ``a`` scales the operator, for example to add a phase. The
    operator is not unitary, so it cannot be applied to a circuit directly;
    use it in an :class:`ExpectationValue` or as a Kraus operator.

    Args:
        a (complex): Scaling factor. Defaults to 1.

    See Also:
        :class:`SigmaMinus`

    Examples:
        >>> from mimiqcircuits import *
        >>> SigmaPlus()
        SigmaPlus(1)

        >>> SigmaPlus(0.5)
        SigmaPlus(0.5)

        >>> # expectation value on qubit 1, stored in z-variable 1
        >>> c = Circuit()
        >>> c.push(ExpectationValue(SigmaPlus()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instruction:
        └── ⟨SigmaPlus(1)⟩ @ q[1], z[1]
        <BLANKLINE>
    """

    _name = "SigmaPlus"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    @property
    def parnames(self):
        return self._parnames

    @property
    def num_qubits(self):
        return self._num_qubits

    def _matrix(self):
        return se.Matrix([[0, 0], [self.a, 0]])

    def opsquared(self):
        r"""Return :math:`O^\dagger O = |a|^2 |0\rangle\langle 0|` as a :class:`Projector0`."""
        return Projector0(abs(self.a) ** 2)

    def __str__(self):
        return f"{self.opname()}({self.a})"


__all__ = [
    "SigmaMinus",
    "SigmaPlus",
]
