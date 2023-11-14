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
from mimiqcircuits.operations.gates.standard.s import GateS
from mimiqcircuits.operations.gates.standard.u import GateU
from symengine import pi


class GateT(Power):
    """ Single qubit T gate.

    **Matrix representation:**

    .. math::
        \\operatorname{T} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & \\exp\\left(\\frac{i\\pi}{4}\\right)
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateT()
        T
        >>> GateT().matrix()
        >>> c = Circuit().push(GateT(), 0)
        >>> GateT().power(2), GateT().inverse()
        (S^(1.0), T†)
        >>> GateT().decompose()
        1-qubit circuit with 1 instructions:
        └── T @ q0
        1-qubit circuit with 1 instructions:
        └── T @ q0
        >>> GateT().matrix()
        >>> c = Circuit().push(GateT(), 0)
        >>> GateT().power(2), GateT().inverse()
        (S^(1.0), T†)
        >>> GateT().decompose()
        1-qubit circuit with 1 instructions:
        └── T @ q0
    """
    _name = 'T'

    def __init__(self):
        super().__init__(GateS(), 1/2)

    def isopalias(self):
        return True

    def inverse(self):
        return GateTDG()

    def __str__(self):
        return f"{self.name}"

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, pi/4), q)
        return circ


class GateTDG(Inverse):
    """Single qubit T-dagger gate (conjugate transpose of the T gate).

    **Matrix representation:**

    .. math::
        \\operatorname{T}^\\dagger = \\begin{pmatrix}
            1 & 0 \\\\
            0 & \\exp\\left(\\frac{-i\\pi}{4}\\right)
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateTDG()
        Inverse
        >>> GateTDG().matrix()
        >>> c = Circuit().push(GateTDG(), 0)
        >>> GateTDG().power(2), GateTDG().inverse()
        (Inverse^(2), T)
        >>> GateTDG().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/4)*pi) @ q0
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/4)*pi) @ q0
        >>> GateTDG().matrix()
        >>> c = Circuit().push(GateTDG(), 0)
        >>> GateTDG().power(2), GateTDG().inverse()
        (Inverse^(2), T)
        >>> GateTDG().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/4)*pi) @ q0
    """

    def __init__(self):
        super().__init__(GateT())

    def isopalias(self):
        return True

    def inverse(self):
        return GateT()

    def __str__(self):
        return f"{self.name}"

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, -pi/4), q)
        return circ
