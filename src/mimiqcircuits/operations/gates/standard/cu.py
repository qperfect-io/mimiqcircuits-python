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

from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.standard.phase import GateP
import mimiqcircuits as mc


def GateCU(theta, phi, lmbda, gamma):
    r"""Two qubit controlled unitary gate.

    **Matrix representation:**

    .. math::
        \operatorname{CU}(\theta, \phi, \lambda, \gamma) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & e^{i\gamma} \cos\left(\frac{\theta}{2}\right) & -e^{i\gamma} e^{i\lambda}\sin\left(\frac{\theta}{2}\right) \\
            0 & 0 & e^{i\gamma} \mathrm{e}^{i\phi}\sin\left(\frac{\theta}{2}\right) & e^{i\gamma} \mathrm{e}^{i(\phi+\lambda)}\cos\left(\frac{\theta}{2}\right)
        \end{pmatrix}

    Parameters:
        theta (float): Euler angle 1 in radians.
        phi (float): Euler angle 2 in radians.
        lmbda (float): Euler angle 3 in radians.
        gamma (float): Global phase of the CU gate.

    Examples:

        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, phi, lmbda, gamma = symbols('theta phi lambda gamma')
        >>> GateCU(theta, phi, lmbda, gamma), GateCU(theta, phi, lmbda, gamma).num_controls, GateCU(theta, phi, lmbda, gamma).num_targets, GateCU(theta, phi, lmbda, gamma).num_qubits
        (CU(theta, phi, lambda, gamma), 1, 1, 2)
        >>> GateCU(theta, phi, lmbda, gamma).matrix()
        [1.0, 0, 0, 0]
        [0, 1.0, 0, 0]
        [0, 0, exp(I*gamma)*cos((1/2)*theta), -exp(I*(gamma + lambda))*sin((1/2)*theta)]
        [0, 0, exp(I*(gamma + phi))*sin((1/2)*theta), exp(I*(gamma + lambda + phi))*cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateCU(theta, phi, lmbda, gamma), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── CU(theta, phi, lambda, gamma) @ q[0], q[1]
        <BLANKLINE>
        >>> GateCU(theta, phi, lmbda, gamma).power(2), GateCU(theta, phi, lmbda, gamma).inverse()
        (C(U(theta, phi, lambda, gamma)**2), CU(-theta, -lambda, -phi, -gamma))
        >>> GateCU(theta, phi, lmbda, gamma).decompose()
        2-qubit circuit with 7 instructions:
        ├── P(gamma) @ q[0]
        ├── P((1/2)*(lambda + phi)) @ q[0]
        ├── P((1/2)*(lambda - phi)) @ q[1]
        ├── CX @ q[0], q[1]
        ├── U((-1/2)*theta, 0, (-1/2)*(lambda + phi), 0.0) @ q[1]
        ├── CX @ q[0], q[1]
        └── U((1/2)*theta, phi, 0, 0.0) @ q[1]
        <BLANKLINE>
    """
    return mc.Control(1, GateU(theta, phi, lmbda, gamma))


@mc.register_control_decomposition(1, mc.GateU)
def _decompose_gatecu(gate, circ, qubits, bits, zvars):
    c, t = qubits
    theta = gate.op.theta
    phi = gate.op.phi
    lmbda = gate.op.lmbda
    gamma = gate.op.gamma

    circ.push(GateP(gamma), c)
    circ.push(GateP((lmbda + phi) / 2), c)
    circ.push(GateP((lmbda - phi) / 2), t)
    circ.push(GateCX(), c, t)
    circ.push(GateU(-theta / 2, 0, -(lmbda + phi) / 2), t)
    circ.push(GateCX(), c, t)
    circ.push(GateU(theta / 2, phi, 0), t)

    return circ
