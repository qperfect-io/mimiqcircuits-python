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

from symengine import pi

import mimiqcircuits as mc


def GateSX():
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
        SX
        >>> GateSX().matrix()
        [0.5 + 0.5*I, 0.5 - 0.5*I]
        [0.5 - 0.5*I, 0.5 + 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateSX(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
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
    return mc.Power(mc.GateX(), 1 / 2)


mc.register_power_alias(mc.GateX, 1 / 2, "SX")


@mc.register_power_decomposition(mc.GateX, 1 / 2)
def _decompose_gatesx(self, circ, qubits, bits, zvars):
    q = qubits[0]
    circ.push(mc.GateSDG(), q)
    circ.push(mc.GateH(), q)
    circ.push(mc.GateSDG(), q)
    circ.push(mc.GateU(0, 0, 0, pi / 4), q)
    return circ


def GateSXDG():
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
        SX†
        >>> GateSXDG().matrix()
        [0.5 - 0.5*I, 0.5 + 0.5*I]
        [0.5 + 0.5*I, 0.5 - 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateSXDG(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
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
    return mc.Inverse(GateSX())


@mc.register_inverse_decomposition((mc.Power, mc.GateX, 1 / 2))
def _decompose_gatesxdg(self, circ, qubits, bits, zvars):
    q = qubits[0]
    circ.push(mc.GateS(), q)
    circ.push(mc.GateH(), q)
    circ.push(mc.GateS(), q)
    circ.push(mc.GateU(0, 0, 0, -pi / 4), q)
    return circ
