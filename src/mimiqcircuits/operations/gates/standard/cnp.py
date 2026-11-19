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
from mimiqcircuits.operations.control import (
    Control,
    canonical_control,
    register_control_decomposition,
)
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.cphase import GateCP


@canonical_control(2, GateP)
class GateCCP(Control):
    r"""Three qubit Controlled-Controlled-Phase gate.

    By convention, the first two qubits are the controls and the third is the
    target.

    Matrix Representation:
        .. math::
            \operatorname{CCP}(\lambda) =
            \operatorname{diag}(1, 1, 1, 1, 1, 1, 1, e^{i\lambda})

    Args:
        lmbda: Phase angle :math:`\lambda` in radians, numeric or symbolic.

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

        Called directly as ``GateCCP(lmbda)``. The signature also accepts
        ``GateCCP(2, GateP(lmbda))`` and ``GateCCP(1, GateCP(lmbda))``, the
        forms :class:`Control` uses to build its canonical subclass.

        Args:
            lmbda_or_num_controls: Phase angle in radians, or the number of
                controls in the ``Control`` form.
            num_controls_or_operation: Unused when called directly, or the
                ``GateP`` (or ``GateCP``) whose angle is taken in the
                ``Control`` form.
            operation: Unused.
        """
        # Detect if called from Control's canonical creation: Control(2, GateP(lmbda)),
        # or Control(1, GateCP(lmbda)) which flattens to the same gate.
        inner = num_controls_or_operation
        if isinstance(inner, Control):
            inner = inner.op
        if isinstance(lmbda_or_num_controls, int) and isinstance(inner, GateP):
            lmbda = inner.lmbda
        else:
            lmbda = lmbda_or_num_controls
        super().__init__(2, GateP(lmbda))


@register_control_decomposition(2, GateP)
def _decompose_gateccp(gate, circ, qubits, bits, zvars):
    c1, c2, t = qubits
    lmbda = gate.op.lmbda
    circ.push(GateCP(lmbda / 2), c2, t)
    circ.push(GateCX(), c1, c2)
    circ.push(GateCP(-lmbda / 2), c2, t)
    circ.push(GateCX(), c1, c2)
    circ.push(GateCP(lmbda / 2), c1, t)
    return circ
