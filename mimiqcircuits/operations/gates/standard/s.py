#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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

from mimiqcircuits.operations.power import Power
from mimiqcircuits.operations.inverse import Inverse
from mimiqcircuits.operations.gates.standard.pauli import GateZ
from mimiqcircuits.operations.gates.standard.u import GateU
from symengine import pi


class GateS(Power):
    """Single qubit gate S.

    It induces a :math:`\\frac{\\pi}{2}` phase gate.

    **Matrix representation:**

    .. math::
        \\operatorname{S} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & i
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateS()
        S
        >>> GateS().matrix()
        >>> c = Circuit().push(GateS(), 0)
        >>> GateS().power(2), GateS().inverse()
        (Z^(1.0), S†)
        >>> GateS().decompose()
        1-qubit circuit with 1 instructions:
        └── S @ q0
        1-qubit circuit with 1 instructions:
        └── S @ q0
        >>> GateS().matrix()
        >>> c = Circuit().push(GateS(), 0)
        >>> GateS().power(2), GateS().inverse()
        (Z^(1.0), S†)
        >>> GateS().decompose()
        1-qubit circuit with 1 instructions:
        └── S @ q0
    """
    _name = 'S'

    def __init__(self):
        super().__init__(GateZ(), 1/2)

    def isopalias(self):
        return True

    def inverse(self):
        return GateSDG()

    def __str__(self):
        return f"{self.name}"

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, pi/2), q)
        return circ


class GateSDG(Inverse):
    """Single qubit S-dagger gate (conjugate transpose of the S gate).

    **Matrix representation:**

    .. math::
        \\operatorname{S}^\\dagger = \\begin{pmatrix}
            1 & 0 \\\\
            0 & -i
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSDG()
        Inverse
        >>> GateSDG().matrix()
        >>> c = Circuit().push(GateSDG(), 0)
        >>> GateSDG().power(2), GateSDG().inverse()
        (Inverse^(2), S)
        >>> GateSDG().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/2)*pi) @ q0
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/2)*pi) @ q0
        >>> GateSDG().matrix()
        >>> c = Circuit().push(GateSDG(), 0)
        >>> GateSDG().power(2), GateSDG().inverse()
        (Inverse^(2), S)
        >>> GateSDG().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/2)*pi) @ q0
    """

    def __init__(self):
        super().__init__(GateS())

    def isopalias(self):
        return True

    def inverse(self):
        return GateS()

    def __str__(self):
        return f"{self.name}"

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, -pi/2), q)
        return circ
