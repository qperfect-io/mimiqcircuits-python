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


def GateT():
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
        └── U(0, 0, (1/4)*pi, 0.0) @ q[0]
        <BLANKLINE>
    """
    return mc.Power(mc.GateS(), 1 / 2)


mc.register_power_alias(mc.GateZ, 1 / 4, "T")


@mc.register_power_decomposition(mc.GateZ, 1 / 4)
def _decompose_gatet(self, circ, qubits, bits, zvars):
    q = qubits[0]
    circ.push(mc.GateU(0, 0, pi / 4), q)
    return circ


def GateTDG():
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
        ((T†)**2, T)
        >>> GateTDG().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, (-1/4)*pi, 0.0) @ q[0]
        <BLANKLINE>
    """
    return mc.Inverse(GateT())


@mc.register_inverse_decomposition((mc.Power, mc.GateZ, 1 / 4))
def _decompose_gatetdg(gate, circ, qubits, bits, zvars):
    q = qubits[0]
    circ.push(mc.GateU(0, 0, -pi / 4), q)
    return circ
