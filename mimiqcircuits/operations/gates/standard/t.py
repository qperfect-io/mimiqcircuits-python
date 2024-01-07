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
import mimiqcircuits as mc


class GateT(Power):
    r""" Single qubit T gate.

    **Matrix representation:**

    .. math::
        \operatorname{T} = \begin{pmatrix}
            1 & 0 \\
            0 & \exp\left(\frac{i\pi}{4}\right)
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateT()
        T
        >>> GateT().matrix()
        [1.0, 0]
        [0, 0.707106781186548 + 0.707106781186548*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateT(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── T @ q[0]
        <BLANKLINE>
        >>> GateT().power(2), GateT().inverse()
        (S, T†)
        >>> GateT().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (1/4)*pi) @ q[0]
        <BLANKLINE>
    """
    _name = 'T'

    def __init__(self):
        super().__init__(GateS(), 1/2)

    def isopalias(self):
        return True

    def inverse(self):
        return GateTDG()

    def _power(self, p):
        if p == 1:
            return self

        elif p == 2:
            return mc.GateS()

        elif p == 4:
            return mc.GateZ()

        elif p == 6:
            return mc.GateSDG()

        else:
            return mc.Power(self, p)

    def __str__(self):
        return f"{self.name}"

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, pi/4), q)
        return circ


class GateTDG(Inverse):
    r"""Single qubit T-dagger gate (conjugate transpose of the T gate).

    **Matrix representation:**

    .. math::
        \operatorname{T}^\dagger = \begin{pmatrix}
            1 & 0 \\
            0 & \exp\left(\frac{-i\pi}{4}\right)
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateTDG()
        T†
        >>> GateTDG().matrix()
        [1.0, 0]
        [0, 0.707106781186547 - 0.707106781186547*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateTDG(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── T† @ q[0]
        <BLANKLINE>
        >>> GateTDG().power(2), GateTDG().inverse()
        (T†^(2), T)
        >>> GateTDG().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/4)*pi) @ q[0]
        <BLANKLINE>
    """

    def __init__(self):
        super().__init__(GateT())

    def isopalias(self):
        return True

    def inverse(self):
        return GateT()

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, -pi/4), q)
        return circ
