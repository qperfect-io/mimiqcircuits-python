#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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

import mimiqcircuits.operations.control as mctrl
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.u import GateU, GateUPhase
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.gates.generalized.gphase import GPhase
from symengine import pi


class GateCU(mctrl.Control):
    """Two qubit generic unitary gate.

    equivalent to the [qiskit CU-Gate](https://qiskit.org/documentation/stubs/qiskit.circuit.library.CUGate.html)


    **Matrix representation:**

    .. math::
        \\operatorname{CU}(\\theta, \\phi, \\lambda, \\gamma) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & e^{i\\gamma}\\cos\\frac{\\theta}{2} & -e^{i(\\gamma+\\lambda)}\\sin\\frac{\\theta}{2} \\\\
            0 & 0 & e^{i(\\gamma+\\phi)}\\sin\\frac{\\theta}{2} & e^{i(\\gamma+\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): Euler angle 1 in radians.
        phi (float): Euler angle 2 in radians.
        lambda (float): Euler angle 3 in radians.
        gamma (float): Global phase of the CU gate.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, phi, lmbda, gamma = symbols('theta phi lambda gamma')
        >>> GateCU(theta, phi, lmbda, gamma), GateCU(theta, phi, lmbda, gamma).num_controls, GateCU(theta, phi, lmbda, gamma).num_targets, GateCU(theta, phi, lmbda, gamma).num_qubits
        (CUPhase(theta, phi, lambda, gamma), 1, 1, 2)
        >>> GateCU(theta, phi, lmbda, gamma).matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, (1/2)*exp(I*gamma)*(1 + exp(I*theta)), 1/2*I*exp(I*(gamma + lambda))*(-1 + exp(I*theta))]
        [0, 0, 1/2*I*exp(I*(gamma + phi))*(1 - exp(I*theta)), (1/2)*exp(I*(gamma + lambda + phi))*(1 + exp(I*theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateCU(theta, phi, lmbda, gamma), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── CUPhase(theta, phi, lambda, gamma) @ q0, q1
        >>> GateCU(theta, phi, lmbda, gamma).power(2), GateCU(theta, phi, lmbda, gamma).inverse()
        (C(UPhase(theta, phi, lambda, gamma)^(2)), CUPhase(-theta, -lambda, -phi, -gamma))
        >>> GateCU(theta, phi, lmbda, gamma).decompose()
        2-qubit circuit with 8 instructions:
        ├── CGPhase(lmbda=theta) @ q0, q1
        ├── U((1/2)*theta, 0, lambda) @ q1
        ├── CX @ q0, q1
        ├── U((1/2)*theta, (-1/2)*(lambda + phi - 2*pi), pi) @ q1
        ├── CX @ q0, q1
        ├── P(-(lambda - phi)) @ q1
        ├── P((1/2)*(lambda + phi + theta)) @ q0
        └── GPhase(lmbda=(-1/2)*theta) @ q0, q1
    """

    def __init__(self, *args, **kwargs):
        super().__init__(1, GateUPhase(*args, **kwargs))

        self.theta = args[0]
        self.phi = args[1]
        self.lmbda = args[2]
        self.gamma = args[3]

    def _decompose(self, circ, qubits, bits):
        a, b = qubits
        circ.push(mctrl.Control(1, GPhase(1, self.theta)), a, b)
        circ.push(GateU((self.theta)/2, 0, self.lmbda), b)
        circ.push(GateCX(), a, b)
        circ.push(GateU(self.theta/2, -(self.phi +
                  self.lmbda - 2*pi)/2, pi), b)
        circ.push(GateCX(), a, b)
        circ.push(GateP(-(self.lmbda - self.phi)), b)
        circ.push(GateP((self.lmbda + self.phi + self.theta)/2), a)
        circ.push(GPhase(2, -self.theta/2), a, b)

        return circ
