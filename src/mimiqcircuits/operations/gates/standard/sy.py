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
"""SY (Sqrt(Y)) and SYDG gates."""

from mimiqcircuits.operations.utils import control_one_defined
from mimiqcircuits.operations.gates.standard.pauli import GateY
from fractions import Fraction
from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.inverse import Inverse, canonical_inverse
from mimiqcircuits.operations.power import Power, canonical_power
from mimiqcircuits.operations.gates.standard._angles import PI_4, NEG_PI_4


# Fractions are immutable, so all instances share this one: building a
# Fraction costs more than the rest of the constructor.
_EXPONENT = Fraction(1, 2)


@canonical_power(GateY, _EXPONENT)
class GateSY(Power):
    r"""Single qubit :math:`\sqrt{Y}` gate.

    Equal to ``GateY().power(1/2)``.

    Matrix Representation:
        .. math::
            \operatorname{SY} = \sqrt{\operatorname{Y}} = \frac{1}{2} \begin{pmatrix}
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
        >>> c = Circuit().push(GateSY(), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── SY @ q[0]
        <BLANKLINE>
        >>> GateSY().power(2), GateSY().inverse()
        (Y**1, SY†)
        >>> GateSY().decompose()
        1-qubit circuit with 4 instructions:
        ├── S @ q[0]
        ├── S @ q[0]
        ├── H @ q[0]
        └── U(0, 0, 0, (1/4)*pi) @ q[0]
        <BLANKLINE>

    See Also:
        :class:`GateSYDG`, :class:`GateY`, :class:`Power`
    """

    _name = "SY"

    def __init__(self, operation=None, exponent=None):
        """Initialize a SY gate. The arguments are unused."""
        super().__init__(GateY(), _EXPONENT)

    def inverse(self):
        return GateSYDG()

    def isopalias(self):
        return True

    def _control(self, n):
        return control_one_defined(n, self, Control(1, GateSY()))

    def _power(self, p):
        return Power(self, p)

    def __str__(self):
        return f"{self.name}"

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(GateS(), q)
        circ.push(GateS(), q)
        circ.push(GateH(), q)
        circ.push(GateU(0, 0, 0, PI_4), q)
        return circ


@canonical_inverse(GateSY)
class GateSYDG(Inverse):
    r"""Single qubit :math:`\sqrt{Y}^\dagger` gate, the inverse of :class:`GateSY`.

    Matrix Representation:
        .. math::
            \operatorname{SYDG} = \sqrt{\operatorname{Y}}^\dagger = \frac{1}{2} \begin{pmatrix}
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
        >>> c = Circuit().push(GateSYDG(), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── SY† @ q[0]
        <BLANKLINE>
        >>> GateSYDG().power(2), GateSYDG().inverse()
        (SY†**2, SY)
        >>> GateSYDG().decompose()
        1-qubit circuit with 4 instructions:
        ├── U(0, 0, 0, (-1/4)*pi) @ q[0]
        ├── H @ q[0]
        ├── S† @ q[0]
        └── S† @ q[0]
        <BLANKLINE>

    See Also:
        :class:`GateSY`, :class:`GateY`, :class:`Inverse`
    """

    def __init__(self, operation=None):
        """Initialize a SYDG gate. The argument is unused."""
        super().__init__(GateSY())

    def inverse(self):
        return GateSY()

    def isopalias(self):
        return True

    def _power(self, p):
        return Power(self, p)

    def _control(self, n):
        return control_one_defined(n, self, Control(1, GateSYDG()))

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(GateU(0, 0, 0, NEG_PI_4), q)
        circ.push(GateH(), q)
        circ.push(GateSDG(), q)
        circ.push(GateSDG(), q)
        return circ
