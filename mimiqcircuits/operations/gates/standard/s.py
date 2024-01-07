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
from mimiqcircuits.operations.utils import control_one_defined
import mimiqcircuits as mc
from symengine import pi


class GateS(Power):
    r"""Single qubit gate S.

    It induces a :math:`\frac{\pi}{2}` phase gate.

    **Matrix representation:**

    .. math::
        \operatorname{S} = \begin{pmatrix}
            1 & 0 \\
            0 & i
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateS()
        S
        >>> GateS().matrix()
        [1.0, 0]
        [0, 0.0 + 1.0*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateS(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── S @ q[0]
        <BLANKLINE>
        >>> GateS().power(2), GateS().inverse()
        (Z, S†)
        >>> GateS().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (1/2)*pi) @ q[0]
        <BLANKLINE>
    """
    _name = 'S'

    def __init__(self):
        super().__init__(GateZ(), 1/2)

    def isopalias(self):
        return True

    def inverse(self):
        return GateSDG()

    def _power(self, p):
        if p == 1:
            return self
        elif p == 2:
            return GateZ()

        elif p == 3:
            return GateSDG()

        elif p == 4:
            return mc.GateID()

        elif p == 5:
            return self

        elif p == 6:
            return GateZ()

        else:
            return mc.Power(self, p)

    def _control(self, n):
        return control_one_defined(n, self, mc.GateCS())

    def __str__(self):
        return f"{self.name}"

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, pi/2), q)
        return circ


class GateSDG(Inverse):
    r"""Single qubit S-dagger gate (conjugate transpose of the S gate).

    **Matrix representation:**

    .. math::
        \operatorname{S}^\dagger = \begin{pmatrix}
            1 & 0 \\
            0 & -i
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSDG()
        S†
        >>> GateSDG().matrix()
        [1.0, 0]
        [0, 6.12323399573677e-17 - 1.0*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateSDG(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── S† @ q[0]
        <BLANKLINE>
        >>> GateSDG().power(2), GateSDG().inverse()
        (S†^(2), S)
        >>> GateSDG().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/2)*pi) @ q[0]
        <BLANKLINE>
    """

    def __init__(self):
        super().__init__(GateS())

    def isopalias(self):
        return True

    def inverse(self):
        return GateS()

    def _control(self, n):
        return control_one_defined(n, self, mc.GateCSDG())

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, -pi/2), q)
        return circ
