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
"""Controlled-Hadamard gate."""

from mimiqcircuits.operations.control import (
    Control,
    canonical_control,
    register_control_decomposition,
)
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG
from mimiqcircuits.operations.gates.standard.hadamard import GateH


@canonical_control(1, GateH)
class GateCH(Control):
    r"""Two qubit Controlled-Hadamard gate.

    By convention, the first qubit is the control and the second is
    the target.

    Matrix Representation:
        .. math::
            \operatorname{CH} = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \frac{1}{\sqrt{2}}  & \frac{1}{\sqrt{2}}  \\
                0 & 0 & \frac{1}{\sqrt{2}}  & -\frac{1}{\sqrt{2}}
            \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCH(), GateCH().num_controls, GateCH().num_targets, GateCH().num_qubits
        (CH, 1, 1, 2)
        >>> GateCH().matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 0.707106781186548, 0.707106781186548]
        [0, 0, 0.707106781186548, -0.707106781186548]
        <BLANKLINE>
        >>> c = Circuit().push(GateCH(), 0, 1)
        >>> c
        2-qubit circuit with 1 instruction:
        └── CH @ q[0], q[1]
        <BLANKLINE>
        >>> GateCH().power(2), GateCH().inverse()
        (CID, CH)
        >>> GateCH().decompose()
        2-qubit circuit with 7 instructions:
        ├── S @ q[1]
        ├── H @ q[1]
        ├── T @ q[1]
        ├── CX @ q[0], q[1]
        ├── T† @ q[1]
        ├── H @ q[1]
        └── S† @ q[1]
        <BLANKLINE>
    """

    def __init__(self, num_controls=1, operation=None):
        """Initialize a CH gate. The arguments are unused."""
        super().__init__(1, GateH())


@register_control_decomposition(1, GateH)
def _decompose_gatech(gate, circ, qubits, bits, zvars):
    c, t = qubits

    circ.push(GateS(), t)
    circ.push(GateH(), t)
    circ.push(GateT(), t)
    circ.push(GateCX(), c, t)
    circ.push(GateTDG(), t)
    circ.push(GateH(), t)
    circ.push(GateSDG(), t)

    return circ
