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
"""Controlled-Phase gate."""

from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.control import (
    Control,
    canonical_control,
    register_control_decomposition,
)


@canonical_control(1, GateP)
class GateCP(Control):
    r"""Two qubit Controlled-Phase gate.

    By convention, the first qubit is the control and the second is
    the target.

    Matrix Representation:
        .. math::
            \operatorname{CP}(\lambda) = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\lambda}
            \end{pmatrix}

    Args:
        lmbda: Phase angle :math:`\lambda` in radians, numeric or symbolic.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateCP(lmbda), GateCP(lmbda).num_controls, GateCP(lmbda).num_targets, GateCP(lmbda).num_qubits
        (CP(lambda), 1, 1, 2)
        >>> GateCP(lmbda).matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 1.0, 0]
        [0, 0, 0, exp(I*lambda)]
        <BLANKLINE>
        >>> c = Circuit().push(GateCP(lmbda), 10, 11)
        >>> c
        12-qubit circuit with 1 instruction:
        └── CP(lambda) @ q[10], q[11]
        <BLANKLINE>
        >>> GateCP(lmbda).power(2), GateCP(lmbda).inverse()
        (CP(2*lambda), CP(-lambda))
        >>> GateCP(lmbda).decompose()
        2-qubit circuit with 5 instructions:
        ├── P((1/2)*lambda) @ q[0]
        ├── CX @ q[0], q[1]
        ├── P((-1/2)*lambda) @ q[1]
        ├── CX @ q[0], q[1]
        └── P((1/2)*lambda) @ q[1]
        <BLANKLINE>

    See Also:
        :class:`GateP`, :class:`GateCCP`
    """

    def __init__(self, lmbda_or_num_controls, num_controls_or_operation=None, operation=None):
        """Initialize a CP gate.

        Called directly as ``GateCP(lmbda)``. The signature also accepts
        ``GateCP(1, GateP(lmbda))``, the form :class:`Control` uses to build
        its canonical subclass.

        Args:
            lmbda_or_num_controls: Phase angle in radians, or the number of
                controls (always 1) in the ``Control`` form.
            num_controls_or_operation: Unused when called directly, or the
                ``GateP`` whose angle is taken in the ``Control`` form.
            operation: Unused.
        """
        # Detect if called from Control's canonical creation: Control(1, GateP(lmbda))
        if isinstance(lmbda_or_num_controls, int) and isinstance(num_controls_or_operation, GateP):
            lmbda = num_controls_or_operation.lmbda
        else:
            lmbda = lmbda_or_num_controls
        super().__init__(1, GateP(lmbda))


@register_control_decomposition(1, GateP)
def _decompose_gatecp(gate, circ, qubits, bits, zvars):
    c, t = qubits
    lmbda2 = gate.op.lmbda / 2

    circ.push(GateP(lmbda2), c)
    circ.push(GateCX(), c, t)
    circ.push(GateP(-lmbda2), t)
    circ.push(GateCX(), c, t)
    circ.push(GateP(lmbda2), t)
    return circ
