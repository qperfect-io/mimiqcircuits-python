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

from mimiqcircuits.operations.utils import control_one_defined
import mimiqcircuits as mc
from symengine import pi


class GateSX(mc.Power):
    r"""Single qubit :math:`\sqrt{X}` gate.

    **Matrix representation:**

    .. math::
        \sqrt{\operatorname{X}} = \frac{1}{2} \begin{pmatrix}
            1+i & 1-i \\
            1-i & 1+i\\
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSX()
        X^(1/2)
        >>> GateSX().matrix()
        [0.5 + 0.5*I, 0.5 - 0.5*I]
        [0.5 - 0.5*I, 0.5 + 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateSX(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── X^(1/2) @ q[0]
        <BLANKLINE>
        >>> GateSX().power(2), GateSX().inverse()
        (X, (X^(1/2))†)
        >>> GateSX().decompose()
        1-qubit circuit with 4 instructions:
        ├── S† @ q[0]
        ├── H @ q[0]
        ├── S† @ q[0]
        └── GPhase((1/4)*pi) @ q[0]
        <BLANKLINE>
    """

    def __init__(self):
        super().__init__(mc.GateX(), 1/2)

    def inverse(self):
        return GateSXDG()

    def _control(self, n):
        return control_one_defined(n, self, mc.GateCSX())

    def _power(self, p):
        pmod = p % 2

        if pmod == 0:
            return mc.GateX()
        else:
            if p in {3, 7, 11}:
                return mc.GateSXDG()
            elif p in {5, 9, 13}:
                return self
            else:
                return mc.Power(self, p)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(mc.GateSDG(), q)
        circ.push(mc.GateH(), q)
        circ.push(mc.GateSDG(), q)
        circ.push(mc.GPhase(1, pi/4), q)
        return circ


class GateSXDG(mc.Inverse):
    r"""Single qubit :math:`\sqrt{X}^\dagger` gate (conjugate transpose of the :math:`\sqrt{X}` gate).

    **Matrix representation:**

    .. math::
        \sqrt{\operatorname{X}}^\dagger = \frac{1}{2} \begin{pmatrix}
            1-i & 1+i \\
            1+i & 1-i\\
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSXDG()
        (X^(1/2))†
        >>> GateSXDG().matrix()
        [0.5 - 0.5*I, 0.5 + 0.5*I]
        [0.5 + 0.5*I, 0.5 - 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateSXDG(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── (X^(1/2))† @ q[0]
        <BLANKLINE>
        >>> GateSXDG().power(2), GateSXDG().inverse()
        (((X^(1/2))†)^(2), X^(1/2))
        >>> GateSXDG().decompose()
        1-qubit circuit with 4 instructions:
        ├── GPhase((-1/4)*pi) @ q[0]
        ├── S @ q[0]
        ├── H @ q[0]
        └── S @ q[0]
        <BLANKLINE>
    """

    def __init__(self):
        super().__init__(GateSX())

    def inverse(self):
        return GateSX()

    def _control(self, n):
        return control_one_defined(n, self, mc.GateCSXDG())

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(mc.GPhase(1, -pi/4), q)
        circ.push(mc.GateS(), q)
        circ.push(mc.GateH(), q)
        circ.push(mc.GateS(), q)
        return circ
