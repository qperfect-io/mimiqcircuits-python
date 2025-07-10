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

from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.deprecated import GateU1
import mimiqcircuits as mc
from symengine import pi


def GateCSX():
    r"""Two qubit Controled-SX gate.

    By convention, the first qubit is the control and second one is the
    targets.


    **Matrix representation:**

    .. math::

        \operatorname{CSX} =\begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1+i}{2} & \frac{1-i}{2} \\
            0 & 0 & \frac{1-i}{2} & \frac{1+i}{2}
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCSX(), GateCSX().num_controls, GateCSX().num_targets, GateCSX().num_qubits
        (CSX, 1, 1, 2)
        >>> GateCSX().matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 0.5 + 0.5*I, 0.5 - 0.5*I]
        [0, 0, 0.5 - 0.5*I, 0.5 + 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateCSX(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── CSX @ q[0], q[1]
        <BLANKLINE>
        >>> GateCSX().power(2), GateCSX().inverse()
        (CX, C(SX†))
        >>> GateCSX().decompose()
        2-qubit circuit with 4 instructions:
        ├── C(S†) @ q[0], q[1]
        ├── CH @ q[0], q[1]
        ├── C(S†) @ q[0], q[1]
        └── CU(0, 0, 0, (1/4)*pi) @ q[0], q[1]
        <BLANKLINE>
    """
    return mc.Control(1, GateSX())


@mc.register_control_decomposition(1, mc.GateSX)
def _decompose_gatecsx(gate, circ, qubits, bits, zvars):
    a, b = qubits

    circ.push(GateH(), b)
    circ.push(GateU1(pi / 2).control(1), a, b)
    circ.push(GateH(), b)

    return circ


def GateCSXDG():
    r"""Two qubit :math:`{CSX}^\dagger` gate.

    **Matrix representation:**

    .. math::
        \operatorname{CSX}^{\dagger} =\begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1-i}{2} & \frac{1+i}{2} \\
            0 & 0 & \frac{1+i}{2} & \frac{1-i}{2}
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCSXDG(), GateCSXDG().num_controls, GateCSXDG().num_targets, GateCSXDG().num_qubits
        (C(SX†), 1, 1, 2)
        >>> GateCSXDG().matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 0.5 - 0.5*I, 0.5 + 0.5*I]
        [0, 0, 0.5 + 0.5*I, 0.5 - 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateCSXDG(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── C(SX†) @ q[0], q[1]
        <BLANKLINE>
        >>> GateCSXDG().power(2), GateCSXDG().inverse()
        (C((SX†)**2), CSX)
        >>> GateCSXDG().decompose()
        2-qubit circuit with 4 instructions:
        ├── CS @ q[0], q[1]
        ├── CH @ q[0], q[1]
        ├── CS @ q[0], q[1]
        └── CU(0, 0, 0, (-1/4)*pi) @ q[0], q[1]
        <BLANKLINE>
    """
    return mc.Control(1, GateSXDG())


@mc.register_control_decomposition(1, mc.GateSXDG)
def _decompose_gatecsxdg(gate, circ, qubits, bits, zvars):
    a, b = qubits

    circ.push(GateH(), b)
    circ.push(GateU1(-pi / 2).control(1), a, b)
    circ.push(GateH(), b)

    return circ
