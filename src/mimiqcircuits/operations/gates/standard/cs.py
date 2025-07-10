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

from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.cphase import GateCP
import mimiqcircuits as mc
from symengine import pi


def GateCS():
    r"""Two qubit Controlled-S gate.

    By convention, the first qubit is the control and the second is
    the target

    See Also :func:`GateS`

    **Matrix representation:**:

    .. math::
        \operatorname{CS} =\begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & i
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCS(), GateCS().num_controls, GateCS().num_targets, GateCS().num_qubits
        (CS, 1, 1, 2)
        >>> GateCS().matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 1.0, 0]
        [0, 0, 0, 0.0 + 1.0*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateCS(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── CS @ q[0], q[1]
        <BLANKLINE>
        >>> GateCS().power(2), GateCS().inverse()
        (CZ, C(S†))
        >>> GateCS().decompose()
        2-qubit circuit with 1 instructions:
        └── CU(0, 0, (1/2)*pi, 0.0) @ q[0], q[1]
        <BLANKLINE>
    """
    return mc.Control(1, GateS())


@mc.register_control_decomposition(1, mc.GateS)
def _decompose_gatecs(gate, circ, qubits, bits, zvars):
    a, b = qubits
    circ.push(GateCP(pi / 2), a, b)
    return circ


def GateCSDG():
    r"""Adjoint of two qubit Controlled-S gate.

    By convention, the first qubit is the control and the second is
    the target

    **Matrix representation:**

    .. math::
        \operatorname{CS}^{\dagger} = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & -i
            \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCSDG(), GateCSDG().num_controls, GateCSDG().num_targets, GateCSDG().num_qubits
        (C(S†), 1, 1, 2)
        >>> GateCSDG().matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 1.0, 0]
        [0, 0, 0, 6.12323399573677e-17 - 1.0*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateCSDG(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── C(S†) @ q[0], q[1]
        <BLANKLINE>
        >>> GateCSDG().power(2), GateCSDG().inverse()
        (C((S†)**2), CS)
        >>> GateCSDG().decompose()
        2-qubit circuit with 1 instructions:
        └── CU(0, 0, (-1/2)*pi, 0.0) @ q[0], q[1]
        <BLANKLINE>
    """
    return mc.Control(1, mc.GateSDG())


@mc.register_control_decomposition(1, mc.GateSDG)
def _decompose_gatecsdg(gate, circ, qubits, bits, zvars):
    a, b = qubits
    circ.push(GateCP(-pi / 2), a, b)
    return circ
