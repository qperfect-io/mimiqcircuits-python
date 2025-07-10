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

import mimiqcircuits.operations.control as mctrl
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.hadamard import GateH
import mimiqcircuits as mc


def GateCX():
    r"""Two qubit Controlled-X gate (or CNOT).

    By convention, the first qubit is the control and the second is
    the target

    **Matrix representation:**

    .. math::
        \operatorname{CX} = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCX(), GateCX().num_controls, GateCX().num_targets
        (CX, 1, 1)
        >>> GateCX().matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 0, 1.0]
        [0, 0, 1.0, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateCX(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── CX @ q[0], q[1]
        <BLANKLINE>
        >>> GateCX().power(2), GateCX().inverse()
        (CID, CX)
        >>> GateCX().decompose()
        2-qubit circuit with 1 instructions:
        └── CX @ q[0], q[1]
        <BLANKLINE>
    """
    return mc.Control(1, mc.GateX())


@mc.register_control_decomposition(1, mc.GateX)
def _decompose_gatecx(gate, circ, qubits, bits, zvars):
    c, t = qubits
    circ.push(gate, c, t)
    return circ


def GateCY():
    r"""Two qubit Controlled-Y gate.

    By convention, the first qubit is the control and the second is
    the target

    **Matrix representation:**

    .. math::
        \operatorname{CY} = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & -i \\
            0 & 0 & i & 0
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCY(), GateCY().num_controls, GateCY().num_targets
        (CY, 1, 1)
        >>> GateCY().matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 0, -0.0 - 1.0*I]
        [0, 0, 0.0 + 1.0*I, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateCY(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── CY @ q[0], q[1]
        <BLANKLINE>
        >>> GateCY().power(2), GateCY().inverse()
        (CID, CY)
        >>> GateCY().decompose()
        2-qubit circuit with 3 instructions:
        ├── S† @ q[1]
        ├── CX @ q[0], q[1]
        └── S @ q[1]
        <BLANKLINE>
    """
    return mc.Control(1, GateY())


@mc.register_control_decomposition(1, mc.GateY)
def _decompose_gatecy(gate, circ, qubits, bits, zvars):
    c, t = qubits
    circ.push(GateSDG(), t)
    circ.push(GateCX(), c, t)
    circ.push(GateS(), t)
    return circ


def GateCZ():
    r"""Two qubit Controlled-Z gate.

    By convention, the first qubit is the control and the second is
    the target

    **Matrix representation:**

    .. math::
        \operatorname{CZ} = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & -1
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCZ(), GateCZ().num_controls, GateCZ().num_targets
        (CZ, 1, 1)
        >>> GateCZ().matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 1.0, 0]
        [0, 0, 0, -1.0]
        <BLANKLINE>
        >>> c = Circuit().push(GateCZ(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── CZ @ q[0], q[1]
        <BLANKLINE>
        >>> GateCZ().power(2), GateCZ().inverse()
        (CID, CZ)
        >>> GateCZ().decompose()
        2-qubit circuit with 3 instructions:
        ├── H @ q[1]
        ├── CX @ q[0], q[1]
        └── H @ q[1]
        <BLANKLINE>
    """
    return mc.Control(1, GateZ())


@mc.register_control_decomposition(1, mc.GateZ)
def _decompose_gatecz(gate, circ, qubits, bits, zvars):
    c, t = qubits
    circ.push(GateH(), t)
    circ.push(GateCX(), c, t)
    circ.push(GateH(), t)
    return circ
