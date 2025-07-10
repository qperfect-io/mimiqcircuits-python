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


def GateCCP(lmbda):
    r"""Three qubit Controlled-Controlled-Phase gate.

    By convention, the first two qubits are the controls and the third is the
    target

    Arguments:
        lmbda: Phase angle.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lmbda')
        >>> GateCCP(lmbda), GateCCP(lmbda).num_controls, GateCCP(lmbda).num_targets, GateCCP(lmbda).num_qubits
        (C₂P(lmbda), 2, 1, 3)
        >>> GateCCP(lmbda).matrix()
        [1.0, 0, 0, 0, 0, 0, 0, 0]
        [0, 1.0, 0, 0, 0, 0, 0, 0]
        [0, 0, 1.0, 0, 0, 0, 0, 0]
        [0, 0, 0, 1.0, 0, 0, 0, 0]
        [0, 0, 0, 0, 1.0, 0, 0, 0]
        [0, 0, 0, 0, 0, 1.0, 0, 0]
        [0, 0, 0, 0, 0, 0, 1.0, 0]
        [0, 0, 0, 0, 0, 0, 0, exp(I*lmbda)]
        <BLANKLINE>
        >>> c = Circuit().push(GateCCP(lmbda), 0, 1, 2)
        >>> c
        3-qubit circuit with 1 instructions:
        └── C₂P(lmbda) @ q[0,1], q[2]
        <BLANKLINE>
        >>> GateCCP(lmbda).power(2), GateCCP(lmbda).inverse()
        (C₂P(2*lmbda), C₂P(-lmbda))
        >>> GateCCP(lmbda).decompose()
        3-qubit circuit with 5 instructions:
        ├── CP((1/2)*lmbda) @ q[1], q[2]
        ├── CX @ q[0], q[1]
        ├── CP((-1/2)*lmbda) @ q[1], q[2]
        ├── CX @ q[0], q[1]
        └── CP((1/2)*lmbda) @ q[0], q[2]
        <BLANKLINE>
    """
    return mc.Control(2, mc.GateP(lmbda))


@mc.register_control_decomposition(2, mc.GateP)
def _decompose_gateccp(gate, circ, qubits, bits, zvars):
    c1, c2, t = qubits
    lmbda = gate.op.lmbda
    circ.push(mc.GateCP(lmbda / 2), c2, t)
    circ.push(mc.GateCX(), c1, c2)
    circ.push(mc.GateCP(-lmbda / 2), c2, t)
    circ.push(mc.GateCX(), c1, c2)
    circ.push(mc.GateCP(lmbda / 2), c1, t)
    return circ
