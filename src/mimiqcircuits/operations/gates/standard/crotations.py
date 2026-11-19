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
"""Controlled-Rotation (CRX, CRY, CRZ) gates."""

from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.standard.s import GateS
from mimiqcircuits.operations.gates.standard.rotations import GateRX, GateRY, GateRZ
from mimiqcircuits.operations.control import (
    Control,
    canonical_control,
    register_control_decomposition,
)
from mimiqcircuits.operations.gates.standard._angles import NEG_PI_2


@canonical_control(1, GateRX)
class GateCRX(Control):
    r"""Two qubit Controlled-RX gate.

    By convention, the first qubit is the control and the second is
    the target.

    Matrix Representation:
        .. math::
            \operatorname{CRX}(\theta) = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{pmatrix}

    Args:
        theta: Rotation angle in radians, numeric or symbolic.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')
        >>> GateCRX(theta), GateCRX(theta).num_controls, GateCRX(theta).num_targets, GateCRX(theta).num_qubits
        (CRX(theta), 1, 1, 2)
        >>> GateCRX(theta).matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, cos((1/2)*theta), -I*sin((1/2)*theta)]
        [0, 0, -I*sin((1/2)*theta), cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRX(theta), 0, 1)
        >>> c
        2-qubit circuit with 1 instruction:
        └── CRX(theta) @ q[0], q[1]
        <BLANKLINE>
        >>> GateCRX(theta).power(2), GateCRX(theta).inverse()
        (CRX(2*theta), CRX(-theta))
        >>> GateCRX(theta).decompose()
        2-qubit circuit with 5 instructions:
        ├── S @ q[1]
        ├── CX @ q[0], q[1]
        ├── U((-1/2)*theta, 0, 0, 0.0) @ q[1]
        ├── CX @ q[0], q[1]
        └── U((1/2)*theta, (-1/2)*pi, 0, 0.0) @ q[1]
        <BLANKLINE>

    See Also:
        :class:`GateRX`, :class:`GateCRY`, :class:`GateCRZ`
    """

    def __init__(
        self, theta_or_num_controls, num_controls_or_operation=None, operation=None
    ):
        """Initialize a CRX gate.

        Called directly as ``GateCRX(theta)``. The signature also accepts
        ``GateCRX(1, GateRX(theta))``, the form :class:`Control` uses to build
        its canonical subclass.

        Args:
            theta_or_num_controls: Rotation angle in radians, or the number of
                controls (always 1) in the ``Control`` form.
            num_controls_or_operation: Unused when called directly, or the
                ``GateRX`` whose angle is taken in the ``Control`` form.
            operation: Unused.
        """
        # Detect if called from Control's canonical creation: Control(1, GateRX(theta))
        # In that case, theta_or_num_controls is int and num_controls_or_operation is GateRX
        if isinstance(theta_or_num_controls, int) and isinstance(
            num_controls_or_operation, GateRX
        ):
            theta = num_controls_or_operation.theta
        else:
            theta = theta_or_num_controls
        super().__init__(1, GateRX(theta))


@register_control_decomposition(1, GateRX)
def _decompose_gatecrx(gate, circ, qubits, bits, zvars):
    c, t = qubits
    theta = gate.op.theta
    circ.push(GateS(), t)
    circ.push(GateCX(), c, t)
    circ.push(GateU(-theta / 2, 0, 0), t)
    circ.push(GateCX(), c, t)
    circ.push(GateU(theta / 2, NEG_PI_2, 0), t)
    return circ


@canonical_control(1, GateRY)
class GateCRY(Control):
    r"""Two qubit Controlled-RY gate.

    By convention, the first qubit is the control and the second is
    the target.

    Matrix Representation:
        .. math::
            \operatorname{CRY}(\theta) = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                0 & 0 &  \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{pmatrix}

    Args:
        theta: Rotation angle in radians, numeric or symbolic.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')
        >>> GateCRY(theta), GateCRY(theta).num_controls, GateCRY(theta).num_targets, GateCRY(theta).num_qubits
        (CRY(theta), 1, 1, 2)
        >>> GateCRY(theta).matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, cos((1/2)*theta), -sin((1/2)*theta)]
        [0, 0, sin((1/2)*theta), cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRY(theta), 0, 1)
        >>> c
        2-qubit circuit with 1 instruction:
        └── CRY(theta) @ q[0], q[1]
        <BLANKLINE>
        >>> GateCRY(theta).power(2), GateCRY(theta).inverse()
        (CRY(2*theta), CRY(-theta))
        >>> GateCRY(theta).decompose()
        2-qubit circuit with 4 instructions:
        ├── RY((1/2)*theta) @ q[1]
        ├── CX @ q[0], q[1]
        ├── RY((-1/2)*theta) @ q[1]
        └── CX @ q[0], q[1]
        <BLANKLINE>

    See Also:
        :class:`GateRY`, :class:`GateCRX`, :class:`GateCRZ`
    """

    def __init__(
        self, theta_or_num_controls, num_controls_or_operation=None, operation=None
    ):
        """Initialize a CRY gate.

        Called directly as ``GateCRY(theta)``. The signature also accepts
        ``GateCRY(1, GateRY(theta))``, the form :class:`Control` uses to build
        its canonical subclass.

        Args:
            theta_or_num_controls: Rotation angle in radians, or the number of
                controls (always 1) in the ``Control`` form.
            num_controls_or_operation: Unused when called directly, or the
                ``GateRY`` whose angle is taken in the ``Control`` form.
            operation: Unused.
        """
        # Detect if called from Control's canonical creation: Control(1, GateRY(theta))
        if isinstance(theta_or_num_controls, int) and isinstance(
            num_controls_or_operation, GateRY
        ):
            theta = num_controls_or_operation.theta
        else:
            theta = theta_or_num_controls
        super().__init__(1, GateRY(theta))


@register_control_decomposition(1, GateRY)
def _decompose_gatecry(gate, circ, qubits, bits, zvars):
    c, t = qubits
    theta = gate.op.theta
    circ.push(GateRY(theta / 2), t)
    circ.push(GateCX(), c, t)
    circ.push(GateRY(-theta / 2), t)
    circ.push(GateCX(), c, t)
    return circ


@canonical_control(1, GateRZ)
class GateCRZ(Control):
    r"""Two qubit Controlled-RZ gate.

    By convention, the first qubit is the control and the second is
    the target.

    Matrix Representation:
        .. math::
            \operatorname{CRZ}(\lambda) = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{-i\frac{\lambda}{2}} & 0 \\
                0 & 0 & 0 & e^{i\frac{\lambda}{2}}
            \end{pmatrix}

    Args:
        lmbda: Rotation angle :math:`\lambda` in radians, numeric or symbolic.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateCRZ(lmbda), GateCRZ(lmbda).num_controls, GateCRZ(lmbda).num_targets, GateCRZ(lmbda).num_qubits
        (CRZ(lambda), 1, 1, 2)
        >>> GateCRZ(lmbda).matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, exp(-1/2*I*lambda), 0]
        [0, 0, 0, exp(1/2*I*lambda)]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRZ(lmbda), 0, 1)
        >>> c
        2-qubit circuit with 1 instruction:
        └── CRZ(lambda) @ q[0], q[1]
        <BLANKLINE>
        >>> GateCRZ(lmbda).power(2), GateCRZ(lmbda).inverse()
        (CRZ(2*lambda), CRZ(-lambda))
        >>> GateCRZ(lmbda).decompose()
        2-qubit circuit with 4 instructions:
        ├── RZ((1/2)*lambda) @ q[1]
        ├── CX @ q[0], q[1]
        ├── RZ((-1/2)*lambda) @ q[1]
        └── CX @ q[0], q[1]
        <BLANKLINE>

    See Also:
        :class:`GateRZ`, :class:`GateCRX`, :class:`GateCRY`
    """

    def __init__(
        self, lmbda_or_num_controls, num_controls_or_operation=None, operation=None
    ):
        """Initialize a CRZ gate.

        Called directly as ``GateCRZ(lmbda)``. The signature also accepts
        ``GateCRZ(1, GateRZ(lmbda))``, the form :class:`Control` uses to build
        its canonical subclass.

        Args:
            lmbda_or_num_controls: Rotation angle in radians, or the number of
                controls (always 1) in the ``Control`` form.
            num_controls_or_operation: Unused when called directly, or the
                ``GateRZ`` whose angle is taken in the ``Control`` form.
            operation: Unused.
        """
        # Detect if called from Control's canonical creation: Control(1, GateRZ(lmbda))
        if isinstance(lmbda_or_num_controls, int) and isinstance(
            num_controls_or_operation, GateRZ
        ):
            lmbda = num_controls_or_operation.lmbda
        else:
            lmbda = lmbda_or_num_controls
        super().__init__(1, GateRZ(lmbda))


@register_control_decomposition(1, GateRZ)
def _decompose_gatecrz(gate, circ, qubits, bits, zvars):
    c, t = qubits
    lmbda = gate.op.lmbda
    circ.push(GateRZ(lmbda / 2), t)
    circ.push(GateCX(), c, t)
    circ.push(GateRZ(-lmbda / 2), t)
    circ.push(GateCX(), c, t)
    return circ
