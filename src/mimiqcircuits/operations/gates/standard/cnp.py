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
"""Controlled-Controlled-Phase (CCP) gate."""

from mimiqcircuits.operations.gates.standard.phase import GateP
import mimiqcircuits as mc


@mc.canonical_control(2, GateP)
class GateCCP(mc.Control):
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
        3-qubit circuit with 1 instruction:
        └── C₂P(lmbda) @ q[0:1], q[2]
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

    def __init__(
        self, lmbda_or_num_controls, num_controls_or_operation=None, operation=None
    ):
        """Initialize a CCP gate.

        Args:
            lmbda_or_num_controls: Phase angle in radians when called directly,
                or num_controls when called from Control's canonical creation.
            num_controls_or_operation: Ignored when called directly, or the GateP
                operation when called from Control's canonical creation.
            operation: Ignored (for compatibility).
        """
        # Detect if called from Control's canonical creation: Control(2, GateP(lmbda))
        if isinstance(lmbda_or_num_controls, int) and isinstance(
            num_controls_or_operation, GateP
        ):
            lmbda = num_controls_or_operation.lmbda
        else:
            lmbda = lmbda_or_num_controls
        super().__init__(2, GateP(lmbda))


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
