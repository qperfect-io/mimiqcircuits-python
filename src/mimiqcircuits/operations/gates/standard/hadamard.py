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
"""Hadamard (H) and related gates (HXY, HYZ, HXZ)."""

from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.power import Power
from mimiqcircuits.operations.utils import power_idempotent
from mimiqcircuits.operations.utils import control_one_defined
from symengine import pi, sqrt, Matrix, I
from mimiqcircuits.operations.gates.standard._angles import PI_2, NEG_PI_4


class GateH(Gate):
    r"""Single qubit Hadamard gate, :math:`(X + Z)/\sqrt{2}`.

    Matrix Representation:
        .. math::
            \operatorname{H} = \frac{1}{\sqrt{2}} \begin{pmatrix}
                1 & 1 \\
                1 & -1
            \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateH()
        H
        >>> GateH().matrix()
        [0.707106781186548, 0.707106781186548]
        [0.707106781186548, -0.707106781186548]
        <BLANKLINE>
        >>> c = Circuit().push(GateH(), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> GateH().power(2), GateH().inverse()
        (ID, H)
        >>> GateH().decompose()
        1-qubit circuit with 1 instruction:
        └── U((1/2)*pi, 0, pi, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "H"

    _num_qubits = 1
    _qregsizes = [1]

    def inverse(self):
        return self

    def _power(self, p):
        # H^(2n) = ID
        # H^(2n+1) = H
        return power_idempotent(self, p)

    def _control(self, n):
        from mimiqcircuits.operations.gates.standard.chadamard import GateCH
        return control_one_defined(n, self, GateCH(), Control(2, GateH()))

    def _matrix(self):
        return Matrix([[1, 1], [1, -1]]) / sqrt(2)

    def _decompose(self, circ, qubits, bits, zvars):
        from mimiqcircuits.operations.gates.standard.u import GateU
        q = qubits[0]
        circ.push(GateU(PI_2, 0, pi), q)
        return circ


class GateHXY(Gate):
    r"""Single qubit HXY gate, :math:`(X + Y)/\sqrt{2}`.

    A Hadamard-like gate that exchanges the :math:`X` and :math:`Y` axes.

    Matrix Representation:
        .. math::
            \operatorname{HXY} = \frac{1}{\sqrt{2}} \begin{pmatrix}
                0 & 1 - i \\
                1 + i & 0
            \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateHXY()
        HXY
        >>> GateHXY().matrix()
        [0, 0.707106781186548 - 0.707106781186548*I]
        [0.707106781186548 + 0.707106781186548*I, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateHXY(), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── HXY @ q[0]
        <BLANKLINE>
        >>> GateHXY().power(2), GateHXY().inverse()
        (HXY**2, HXY)
        >>> GateHXY().decompose()
        1-qubit circuit with 5 instructions:
        ├── H @ q[0]
        ├── Z @ q[0]
        ├── H @ q[0]
        ├── S @ q[0]
        └── U(0, 0, 0, (-1/4)*pi) @ q[0]
        <BLANKLINE>
    """

    _name = "HXY"

    _num_qubits = 1
    _qregsizes = [1]

    def inverse(self):
        return self

    def _power(self, p):
        return Power(GateHXY(), p)

    def _control(self, n):
        return control_one_defined(n, self, Control(1, GateHXY()))

    def _matrix(self):
        return Matrix([[0, 1 - I], [1 + I, 0]]) / sqrt(2)

    def _decompose(self, circ, qubits, bits, zvars):
        from mimiqcircuits.operations.gates.standard.u import GateU
        from mimiqcircuits.operations.gates.standard.s import GateS
        from mimiqcircuits.operations.gates.standard.pauli import GateZ
        q = qubits[0]
        circ.push(GateH(), q)
        circ.push(GateZ(), q)
        circ.push(GateH(), q)
        circ.push(GateS(), q)
        circ.push(GateU(0, 0, 0, NEG_PI_4), q)
        return circ


class GateHYZ(Gate):
    r"""Single qubit HYZ gate, :math:`(Y + Z)/\sqrt{2}`.

    A Hadamard-like gate that exchanges the :math:`Y` and :math:`Z` axes.

    Matrix Representation:
        .. math::
            \operatorname{HYZ} = \frac{1}{\sqrt{2}} \begin{pmatrix}
                1 & -i \\
                i & -1
            \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateHYZ()
        HYZ
        >>> GateHYZ().matrix()
        [0.707106781186548, 0.0 - 0.707106781186548*I]
        [0.0 + 0.707106781186548*I, -0.707106781186548]
        <BLANKLINE>
        >>> c = Circuit().push(GateHYZ(), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── HYZ @ q[0]
        <BLANKLINE>
        >>> GateHYZ().power(2), GateHYZ().inverse()
        (HYZ**2, HYZ)
        >>> GateHYZ().decompose()
        1-qubit circuit with 5 instructions:
        ├── H @ q[0]
        ├── S @ q[0]
        ├── H @ q[0]
        ├── Z @ q[0]
        └── U(0, 0, 0, (-1/4)*pi) @ q[0]
        <BLANKLINE>
    """

    _name = "HYZ"

    _num_qubits = 1
    _qregsizes = [1]

    def inverse(self):
        return self

    def _power(self, p):
        return Power(GateHYZ(), p)

    def _control(self, n):
        return control_one_defined(
            n, self, Control(1, GateHYZ()), Control(2, GateHYZ())
        )

    def _matrix(self):
        return Matrix([[1, -I], [I, -1]]) / sqrt(2)

    def _decompose(self, circ, qubits, bits, zvars):
        from mimiqcircuits.operations.gates.standard.u import GateU
        from mimiqcircuits.operations.gates.standard.s import GateS
        from mimiqcircuits.operations.gates.standard.pauli import GateZ
        q = qubits[0]
        circ.push(GateH(), q)
        circ.push(GateS(), q)
        circ.push(GateH(), q)
        circ.push(GateZ(), q)
        circ.push(GateU(0, 0, 0, NEG_PI_4), q)
        return circ


class GateHXZ(Gate):
    r"""Single qubit HXZ gate, :math:`(X + Z)/\sqrt{2}`, the same as :class:`GateH`.

    ``GateHXZ()`` returns a :class:`GateH` instance.

    Matrix Representation:
        .. math::
            \operatorname{HXZ} = \frac{1}{\sqrt{2}} \begin{pmatrix}
                1 & 1 \\
                1 & -1
            \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateHXZ()
        H
        >>> GateHXZ().matrix()
        [0.707106781186548, 0.707106781186548]
        [0.707106781186548, -0.707106781186548]
        <BLANKLINE>
        >>> c = Circuit().push(GateHXZ(), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> GateHXZ().power(2), GateHXZ().inverse()
        (ID, H)
        >>> GateHXZ().decompose()
        1-qubit circuit with 1 instruction:
        └── U((1/2)*pi, 0, pi, 0.0) @ q[0]
        <BLANKLINE>
    """

    def __new__(self):
        return GateH()
