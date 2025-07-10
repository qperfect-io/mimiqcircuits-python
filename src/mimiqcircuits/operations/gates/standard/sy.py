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

from mimiqcircuits.operations.utils import control_one_defined
import mimiqcircuits as mc
from symengine import pi


class GateSY(mc.Power):
    r"""Single qubit :math:`\sqrt{Y}` gate.

    See Also:
        :class:`GateSYDG`, :class:`GateY`, :class:`Power`

    **Matrix Representation**

    .. math::
        \operatorname{SY} =
        \sqrt{\operatorname{Y}} =
        \frac{1}{2}
        \begin{pmatrix}
            1+i & -1-i \\
            1+i & 1+i
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSY()
        SY

        >>> GateSY().matrix()
        [0.5 + 0.5*I, -0.5 - 0.5*I]
        [0.5 + 0.5*I, 0.5 + 0.5*I]
        <BLANKLINE>

        >>> c = Circuit()
        >>> c.push(GateSY(), 1)
        2-qubit circuit with 1 instructions:
        └── SY @ q[1]
        <BLANKLINE>

        >>> c.push(GateSY(), 2)
        3-qubit circuit with 2 instructions:
        ├── SY @ q[1]
        └── SY @ q[2]
        <BLANKLINE>

        >>> power(GateSY(), 2)
        Y**1.0

    **Decomposition**

        >>> GateSY().decompose()
        1-qubit circuit with 4 instructions:
        ├── S @ q[0]
        ├── S @ q[0]
        ├── H @ q[0]
        └── U(0, 0, 0, (1/4)*pi) @ q[0]
        <BLANKLINE>
      
    """

    _name = "SY"

    def __init__(self):
        super().__init__(mc.GateY(), 1 / 2)

    def inverse(self):
        return GateSYDG()

    def isopalias(self):
        return True

    def _control(self, n):
        return control_one_defined(n, self, mc.Control(1, GateSY()))

    def _power(self, p):
        return mc.Power(self, p)

    def __str__(self):
        return f"{self.name}"

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(mc.GateS(), q)
        circ.push(mc.GateS(), q)
        circ.push(mc.GateH(), q)
        circ.push(mc.GateU(0, 0, 0, pi / 4), q)
        return circ


class GateSYDG(mc.Inverse):
    r"""Single qubit :math:`\sqrt{Y}^\dagger` gate (conjugate transpose of the :math:`\sqrt{Y}` gate).

    See Also:
        :class:`GateSY`, :class:`GateY`, :class:`Power`, :class:`Inverse`

    **Matrix Representation**

    .. math::
        \operatorname{SYDG} =
        \sqrt{\operatorname{Y}}^\dagger =
        \frac{1}{2}
        \begin{pmatrix}
            1-i & 1-i \\
            -1+i & 1-i
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSYDG()
        SY†

        >>> GateSYDG().matrix()
        [0.5 - 0.5*I, 0.5 - 0.5*I]
        [-0.5 + 0.5*I, 0.5 - 0.5*I]
        <BLANKLINE>

        >>> c = Circuit()
        >>> c.push(GateSYDG(), 1)
        2-qubit circuit with 1 instructions:
        └── SY† @ q[1]
        <BLANKLINE>

        >>> c.push(GateSYDG(), 2)
        3-qubit circuit with 2 instructions:
        ├── SY† @ q[1]
        └── SY† @ q[2]
        <BLANKLINE>

        >>> power(GateSYDG(), 2)
        SY†**2

        >>> inverse(GateSYDG())
        SY

    """

    def __init__(self):
        super().__init__(GateSY())

    def inverse(self):
        return GateSY()

    def isopalias(self):
        return True

    def _power(self, p):
        return mc.Power(self, p)

    def _control(self, n):
        return control_one_defined(n, self, mc.Control(1, GateSYDG()))

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(mc.GateU(0, 0, 0, -pi / 4), q)
        circ.push(mc.GateH(), q)
        circ.push(mc.GateSDG(), q)
        circ.push(mc.GateSDG(), q)
        return circ
