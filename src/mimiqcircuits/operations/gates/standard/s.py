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


def GateS():
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
        └── U(0, 0, (1/2)*pi, 0.0) @ q[0]
        <BLANKLINE>
    """
    return mc.Power(mc.GateZ(), 1 / 2)


mc.register_power_alias(mc.GateZ, 1 / 2, "S")


@mc.register_power_decomposition(mc.GateZ, 1 / 2)
def _decompose_gates(self, circ, qubits, bits, zvars):
    q = qubits[0]
    circ.push(mc.GateU(0, 0, pi / 2), q)
    return circ


def GateSDG():
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
        ((S†)**2, S)
        >>> GateSDG().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/2)*pi, 0.0) @ q[0]
        <BLANKLINE>
    """
    return mc.Inverse(GateS())


@mc.register_inverse_decomposition((mc.Power, mc.GateZ, 1 / 2))
def _decompose_gatesdg(self, circ, qubits, bits, zvars):
    q = qubits[0]
    circ.push(mc.GateU(0, 0, -pi / 2), q)
    return circ
