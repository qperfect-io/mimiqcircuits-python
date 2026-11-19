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
"""SX (Sqrt(X)) and SXDG gates."""

from fractions import Fraction

from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.pauli import GateX
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.inverse import (
    Inverse,
    canonical_inverse,
    register_inverse_decomposition,
)
from mimiqcircuits.operations.power import (
    Power,
    canonical_power,
    register_power_alias,
    register_power_decomposition,
)
from mimiqcircuits.operations.gates.standard._angles import PI_4, NEG_PI_4


# Fractions are immutable, so all instances share this one: building a
# Fraction costs more than the rest of the constructor.
_EXPONENT = Fraction(1, 2)


@canonical_power(GateX, _EXPONENT)
class GateSX(Power):
    r"""Single qubit :math:`\sqrt{X}` gate.

    Equal to ``GateX().power(1/2)``.

    Matrix Representation:
        .. math::
            \sqrt{\operatorname{X}} = \frac{1}{2} \begin{pmatrix}
                1+i & 1-i \\
                1-i & 1+i\\
            \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSX()
        SX
        >>> GateSX().matrix()
        [0.5 + 0.5*I, 0.5 - 0.5*I]
        [0.5 - 0.5*I, 0.5 + 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateSX(), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── SX @ q[0]
        <BLANKLINE>
        >>> GateSX().power(2), GateSX().inverse()
        (X, SX†)
        >>> GateSX().decompose()
        1-qubit circuit with 4 instructions:
        ├── S† @ q[0]
        ├── H @ q[0]
        ├── S† @ q[0]
        └── U(0, 0, 0, (1/4)*pi) @ q[0]
        <BLANKLINE>
    """

    def __init__(self, operation=None, exponent=None):
        """Initialize an SX gate. The arguments are unused."""
        super().__init__(GateX(), _EXPONENT)


register_power_alias(GateX, 1 / 2, "SX")


@register_power_decomposition(GateX, 1 / 2)
def _decompose_gatesx(self, circ, qubits, bits, zvars):
    q = qubits[0]
    circ.push(GateSDG(), q)
    circ.push(GateH(), q)
    circ.push(GateSDG(), q)
    circ.push(GateU(0, 0, 0, PI_4), q)
    return circ


@canonical_inverse(GateSX)
class GateSXDG(Inverse):
    r"""Single qubit :math:`\sqrt{X}^\dagger` gate, the inverse of :class:`GateSX`.

    Matrix Representation:
        .. math::
            \sqrt{\operatorname{X}}^\dagger = \frac{1}{2} \begin{pmatrix}
                1-i & 1+i \\
                1+i & 1-i\\
            \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSXDG()
        SX†
        >>> GateSXDG().matrix()
        [0.5 - 0.5*I, 0.5 + 0.5*I]
        [0.5 + 0.5*I, 0.5 - 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateSXDG(), 0)
        >>> c
        1-qubit circuit with 1 instruction:
        └── SX† @ q[0]
        <BLANKLINE>
        >>> GateSXDG().power(2), GateSXDG().inverse()
        ((SX†)**2, SX)
        >>> GateSXDG().decompose()
        1-qubit circuit with 4 instructions:
        ├── S @ q[0]
        ├── H @ q[0]
        ├── S @ q[0]
        └── U(0, 0, 0, (-1/4)*pi) @ q[0]
        <BLANKLINE>
    """

    def __init__(self, operation=None):
        """Initialize an SXDG gate. The argument is unused."""
        super().__init__(GateSX())


@register_inverse_decomposition((Power, GateX, 1 / 2))
def _decompose_gatesxdg(self, circ, qubits, bits, zvars):
    q = qubits[0]
    circ.push(GateS(), q)
    circ.push(GateH(), q)
    circ.push(GateS(), q)
    circ.push(GateU(0, 0, 0, NEG_PI_4), q)
    return circ
