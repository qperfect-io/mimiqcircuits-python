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


def GateCCX():
    """Three qubit Controlled-Controlled-X gate.

    By convention, the first two qubits are the controls and the third is the
    target.

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCCX(), GateCCX().num_controls, GateCCX().num_targets, GateCCX().num_qubits
        (C₂X, 2, 1, 3)
        >>> GateCCX().matrix()
        [1.0, 0, 0, 0, 0, 0, 0, 0]
        [0, 1.0, 0, 0, 0, 0, 0, 0]
        [0, 0, 1.0, 0, 0, 0, 0, 0]
        [0, 0, 0, 1.0, 0, 0, 0, 0]
        [0, 0, 0, 0, 1.0, 0, 0, 0]
        [0, 0, 0, 0, 0, 1.0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 1.0]
        [0, 0, 0, 0, 0, 0, 1.0, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateCCX(), 0, 1, 2)
        >>> c
        3-qubit circuit with 1 instructions:
        └── C₂X @ q[0,1], q[2]
        <BLANKLINE>
        >>> GateCCX().power(2), GateCCX().inverse()
        (C₂ID, C₂X)
        >>> GateCCX().decompose()
        3-qubit circuit with 15 instructions:
        ├── H @ q[2]
        ├── CX @ q[1], q[2]
        ├── T† @ q[2]
        ├── CX @ q[0], q[2]
        ├── T @ q[2]
        ├── CX @ q[1], q[2]
        ├── T† @ q[2]
        ├── CX @ q[0], q[2]
        ├── T @ q[1]
        ├── T @ q[2]
        ├── H @ q[2]
        ├── CX @ q[0], q[1]
        ├── T @ q[0]
        ├── T† @ q[1]
        └── CX @ q[0], q[1]
        <BLANKLINE>
    """
    return mc.Control(2, mc.GateX())


@mc.register_control_decomposition(2, mc.GateX)
def _decompose_gateccx(self, circ, qubits, bits, zvars):
    c1, c2, t = qubits
    circ.push(mc.GateH(), t)
    circ.push(mc.GateCX(), c2, t)
    circ.push(mc.GateT().inverse(), t)
    circ.push(mc.GateCX(), c1, t)
    circ.push(mc.GateT(), t)
    circ.push(mc.GateCX(), c2, t)
    circ.push(mc.GateT().inverse(), t)
    circ.push(mc.GateCX(), c1, t)
    circ.push(mc.GateT(), c2)
    circ.push(mc.GateT(), t)
    circ.push(mc.GateH(), t)
    circ.push(mc.GateCX(), c1, c2)
    circ.push(mc.GateT(), c1)
    circ.push(mc.GateT().inverse(), c2)
    circ.push(mc.GateCX(), c1, c2)
    return circ


def GateC3X():
    r"""Four qubit Controlled-Controlled-Controlled-X gate.

    By convention, the first three qubits are the controls and the fourth is
    the target

    Examples:
        >>> from mimiqcircuits import *
        >>> GateC3X(), GateC3X().num_controls, GateC3X().num_targets, GateC3X().num_qubits
        (C₃X, 3, 1, 4)
        >>> GateC3X().matrix()
        [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateC3X(), 0, 1, 2, 3)
        >>> c
        4-qubit circuit with 1 instructions:
        └── C₃X @ q[0,1,2], q[3]
        <BLANKLINE>
        >>> GateC3X().power(2), GateC3X().inverse()
        (C₃ID, C₃X)
        >>> GateC3X().decompose()
        4-qubit circuit with 31 instructions:
        ├── H @ q[3]
        ├── P((1/8)*pi) @ q[0]
        ├── P((1/8)*pi) @ q[1]
        ├── P((1/8)*pi) @ q[2]
        ├── P((1/8)*pi) @ q[3]
        ├── CX @ q[0], q[1]
        ├── P((-1/8)*pi) @ q[1]
        ├── CX @ q[0], q[1]
        ├── CX @ q[1], q[2]
        ├── P((-1/8)*pi) @ q[2]
        ├── CX @ q[0], q[2]
        ├── P((1/8)*pi) @ q[2]
        ├── CX @ q[1], q[2]
        ├── P((-1/8)*pi) @ q[2]
        ├── CX @ q[0], q[2]
        ├── CX @ q[2], q[3]
        ├── P((-1/8)*pi) @ q[3]
        ├── CX @ q[1], q[3]
        ├── P((1/8)*pi) @ q[3]
        ⋮   ⋮
        └── H @ q[3]
        <BLANKLINE>
    """
    return mc.Control(3, mc.GateX())


@mc.register_control_decomposition(3, mc.GateX)
def _decompose_gatec3x(self, circ, qubits, bits, zvars):
    a, b, c, d = qubits
    circ.push(mc.GateH(), d)
    circ.push(mc.GateP(pi / 8), qubits)
    circ.push(mc.GateCX(), a, b)
    circ.push(mc.GateP(-pi / 8), b)
    circ.push(mc.GateCX(), a, b)
    circ.push(mc.GateCX(), b, c)
    circ.push(mc.GateP(-pi / 8), c)
    circ.push(mc.GateCX(), a, c)
    circ.push(mc.GateP(pi / 8), c)
    circ.push(mc.GateCX(), b, c)
    circ.push(mc.GateP(-pi / 8), c)
    circ.push(mc.GateCX(), a, c)
    circ.push(mc.GateCX(), c, d)
    circ.push(mc.GateP(-pi / 8), d)
    circ.push(mc.GateCX(), b, d)
    circ.push(mc.GateP(pi / 8), d)
    circ.push(mc.GateCX(), c, d)
    circ.push(mc.GateP(-pi / 8), d)
    circ.push(mc.GateCX(), a, d)
    circ.push(mc.GateP(pi / 8), d)
    circ.push(mc.GateCX(), c, d)
    circ.push(mc.GateP(-pi / 8), d)
    circ.push(mc.GateCX(), b, d)
    circ.push(mc.GateP(pi / 8), d)
    circ.push(mc.GateCX(), c, d)
    circ.push(mc.GateP(-pi / 8), d)
    circ.push(mc.GateCX(), a, d)
    circ.push(mc.GateH(), d)
    return circ
