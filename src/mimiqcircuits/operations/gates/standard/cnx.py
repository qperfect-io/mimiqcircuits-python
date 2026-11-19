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
"""Controlled-Controlled-X (Toffoli) and C3X gates."""

from mimiqcircuits.operations.gates.standard.pauli import GateX
from mimiqcircuits.operations.control import (
    Control,
    canonical_control,
    register_control_decomposition,
)
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.gates.standard.t import GateT
from mimiqcircuits.operations.gates.standard._angles import PI_8, NEG_PI_8


@canonical_control(2, GateX)
class GateCCX(Control):
    r"""Three qubit Controlled-Controlled-X (Toffoli) gate.

    By convention, the first two qubits are the controls and the third is the
    target. The target is flipped when both controls are :math:`|1\rangle`.

    Matrix Representation:
        .. math::
            \operatorname{CCX} = \begin{pmatrix}
                I_6 & 0 \\
                0 & \operatorname{X}
            \end{pmatrix}

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
        3-qubit circuit with 1 instruction:
        └── C₂X @ q[0:1], q[2]
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

    def __init__(self, num_controls=2, operation=None):
        """Initialize a CCX (Toffoli) gate. The arguments are unused."""
        super().__init__(2, GateX())


@register_control_decomposition(2, GateX)
def _decompose_gateccx(self, circ, qubits, bits, zvars):
    c1, c2, t = qubits
    circ.push(GateH(), t)
    circ.push(GateCX(), c2, t)
    circ.push(GateT().inverse(), t)
    circ.push(GateCX(), c1, t)
    circ.push(GateT(), t)
    circ.push(GateCX(), c2, t)
    circ.push(GateT().inverse(), t)
    circ.push(GateCX(), c1, t)
    circ.push(GateT(), c2)
    circ.push(GateT(), t)
    circ.push(GateH(), t)
    circ.push(GateCX(), c1, c2)
    circ.push(GateT(), c1)
    circ.push(GateT().inverse(), c2)
    circ.push(GateCX(), c1, c2)
    return circ


@canonical_control(3, GateX)
class GateC3X(Control):
    r"""Four qubit Controlled-Controlled-Controlled-X gate.

    By convention, the first three qubits are the controls and the fourth is
    the target. The target is flipped when all controls are
    :math:`|1\rangle`.

    Matrix Representation:
        .. math::
            \operatorname{C_3X} = \begin{pmatrix}
                I_{14} & 0 \\
                0 & \operatorname{X}
            \end{pmatrix}

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
        4-qubit circuit with 1 instruction:
        └── C₃X @ q[0:2], q[3]
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

    def __init__(self, num_controls=3, operation=None):
        """Initialize a C3X gate. The arguments are unused."""
        super().__init__(3, GateX())


@register_control_decomposition(3, GateX)
def _decompose_gatec3x(self, circ, qubits, bits, zvars):
    a, b, c, d = qubits
    circ.push(GateH(), d)
    circ.push(GateP(PI_8), qubits)
    circ.push(GateCX(), a, b)
    circ.push(GateP(NEG_PI_8), b)
    circ.push(GateCX(), a, b)
    circ.push(GateCX(), b, c)
    circ.push(GateP(NEG_PI_8), c)
    circ.push(GateCX(), a, c)
    circ.push(GateP(PI_8), c)
    circ.push(GateCX(), b, c)
    circ.push(GateP(NEG_PI_8), c)
    circ.push(GateCX(), a, c)
    circ.push(GateCX(), c, d)
    circ.push(GateP(NEG_PI_8), d)
    circ.push(GateCX(), b, d)
    circ.push(GateP(PI_8), d)
    circ.push(GateCX(), c, d)
    circ.push(GateP(NEG_PI_8), d)
    circ.push(GateCX(), a, d)
    circ.push(GateP(PI_8), d)
    circ.push(GateCX(), c, d)
    circ.push(GateP(NEG_PI_8), d)
    circ.push(GateCX(), b, d)
    circ.push(GateP(PI_8), d)
    circ.push(GateCX(), c, d)
    circ.push(GateP(NEG_PI_8), d)
    circ.push(GateCX(), a, d)
    circ.push(GateH(), d)
    return circ
