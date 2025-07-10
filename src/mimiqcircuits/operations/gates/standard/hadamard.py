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

import mimiqcircuits as mc
from mimiqcircuits.operations.utils import power_idempotent
from mimiqcircuits.operations.utils import control_one_defined
from symengine import pi, sqrt, Matrix, I


class GateH(mc.Gate):
    r"""Single qubit Hadamard gate.

    **Matrix representation:**

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
        1-qubit circuit with 1 instructions:
        └── H @ q[0]
        <BLANKLINE>
        >>> GateH().power(2), GateH().inverse()
        (ID, H)
        >>> GateH().decompose()
        1-qubit circuit with 1 instructions:
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
        return control_one_defined(n, self, mc.GateCH(), mc.Control(2, GateH()))

    def _matrix(self):
        return Matrix([[1, 1], [1, -1]]) / sqrt(2)

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(mc.GateU(pi / 2, 0, pi), q)
        return circ


class GateHXY(mc.Gate):
    r"""Single qubit HXY gate.

    **Matrix representation:**

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
        1-qubit circuit with 1 instructions:
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
        return mc.Power(GateHXY(), p)

    def _control(self, n):
        return control_one_defined(n, self, mc.Control(1, GateHXY()))

    def _matrix(self):
        return Matrix([[0, 1 - I], [1 + I, 0]]) / sqrt(2)

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(mc.GateH(), q)
        circ.push(mc.GateZ(), q)
        circ.push(mc.GateH(), q)
        circ.push(mc.GateS(), q)
        circ.push(mc.GateU(0, 0, 0, -pi / 4), q)
        return circ


class GateHYZ(mc.Gate):
    r"""Single qubit HYZ gate.

    **Matrix representation:**

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
        [0.707106781186548, -0.0 - 0.707106781186548*I]
        [0.0 + 0.707106781186548*I, -0.707106781186548]
        <BLANKLINE>
        >>> c = Circuit().push(GateHYZ(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
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
        return mc.Power(GateHYZ(), p)

    def _control(self, n):
        return control_one_defined(
            n, self, mc.Control(1, GateHYZ()), mc.Control(2, GateHYZ())
        )

    def _matrix(self):
        return Matrix([[1, -I], [I, -1]]) / sqrt(2)

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(mc.GateH(), q)
        circ.push(mc.GateS(), q)
        circ.push(mc.GateH(), q)
        circ.push(mc.GateZ(), q)
        circ.push(mc.GateU(0, 0, 0, -pi / 4), q)
        return circ


class GateHXZ(mc.Gate):
    r"""Single qubit HXZ gate (alias for :class:`GateH`).

    **Matrix representation:**

    .. math::
        \operatorname{H} = \frac{1}{\sqrt{2}} \begin{pmatrix}
            1 & 1 \\
            1 & -1
        \end{pmatrix}

    Examples:

        The HXZ gate behaves exactly like the Hadamard gate:

        >>> from mimiqcircuits import *
        >>> GateHXZ()
        H
        >>> GateHXZ().matrix()
        [0.707106781186548, 0.707106781186548]
        [0.707106781186548, -0.707106781186548]
        <BLANKLINE>

        Adding GateHXZ to a circuit:

        >>> c = Circuit().push(GateHXZ(), 0)

        Power and inverse of the gate:

        >>> GateHXZ().power(2), GateHXZ().inverse()
        (ID, H)

        Decomposition of the gate:

        >>> GateHXZ().decompose()
        1-qubit circuit with 1 instructions:
        └── U((1/2)*pi, 0, pi, 0.0) @ q[0]
        <BLANKLINE>
    """

    def __new__(self):
        return GateH()
