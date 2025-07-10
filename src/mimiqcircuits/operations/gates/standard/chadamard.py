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

import mimiqcircuits as mc


def GateCH():
    r"""Two qubit Controlled-Hadamard gate.

    By convention, the first qubit is the control and the second is
    the target

    **Matrix representation:**

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
        2-qubit circuit with 1 instructions:
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
    return mc.Control(1, mc.GateH())


@mc.register_control_decomposition(1, mc.GateH)
def _decompose_gatech(gate, circ, qubits, bits, zvars):
    c, t = qubits

    circ.push(mc.GateS(), t)
    circ.push(mc.GateH(), t)
    circ.push(mc.GateT(), t)
    circ.push(mc.GateCX(), c, t)
    circ.push(mc.GateTDG(), t)
    circ.push(mc.GateH(), t)
    circ.push(mc.GateSDG(), t)

    return circ
