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

import mimiqcircuits.operations.gates.gate as mcg
import mimiqcircuits.operations.gates.generalized.gphase as mcp
from mimiqcircuits.matrices import umatrix
import symengine as se
import sympy as sp


class GateUPhase(mcg.Gate):
    """Single qubit generic unitary phase gate.

    **Matrix representation:**

    .. math::
        \\operatorname{U}(\\theta, \\phi, \\lambda, \\gamma) = \\frac{1}{2} e^{i\\gamma} \\begin{pmatrix}
        1 + e^{i\\theta} & -i e^{i\\lambda}(1 - e^{i\\theta}) \\\\
        i e^{i\\phi}(1 - e^{i\\theta}) & e^{i(\\phi + \\lambda)}(1 + e^{i\\theta})
        \\end{pmatrix}

    Parameters:
        theta (float): Euler angle 1 in radians.
        phi (float): Euler angle 2 in radians.
        lambda (float): Euler angle 3 in radians.
        gamma (float, optional): Euler angle 4 in radians (default is 0).

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, phi, lmbda, gamma = symbols('theta phi lambda gamma')
        >>> GateUPhase(theta, phi, lmbda, gamma)
        UPhase(theta, phi, lambda, gamma)
        >>> GateUPhase(theta, phi, lmbda, gamma).matrix()
        [(1/2)*exp(I*gamma)*(1 + exp(I*theta)), 1/2*I*exp(I*gamma + I*lambda)*(-1 + exp(I*theta))]
        [1/2*I*exp(I*gamma + I*phi)*(1 - exp(I*theta)), (1/2)*exp(I*gamma + I*(lambda + phi))*(1 + exp(I*theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateUPhase(theta, phi, lmbda, gamma), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── UPhase(theta, phi, lambda, gamma) @ q0
        >>> GateUPhase(theta, phi, lmbda, gamma).power(2), GateUPhase(theta, phi, lmbda, gamma).inverse()
        (UPhase(theta, phi, lambda, gamma)^(2), UPhase(-theta, -lambda, -phi, -gamma))
        >>> GateUPhase(theta, phi, lmbda, gamma).decompose()
        1-qubit circuit with 2 instructions:
        ├── GPhase(lmbda=gamma) @ q0
        └── U(theta, phi, lambda) @ q0
    """
    _name = 'UPhase'

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ('theta', 'phi', 'lmbda', 'gamma')

    def __init__(self, theta, phi, lmbda, gamma):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda
        self.gamma = gamma

    def matrix(self):
        return se.Matrix(sp.simplify(umatrix(self.theta, self.phi, self.lmbda, self.gamma)))

    def inverse(self):
        return GateUPhase(-self.theta, -self.lmbda, -self.phi, -self.gamma)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(mcp.GPhase(1, self.gamma), q)
        circ.push(GateU(self.theta, self.phi, self.lmbda), q)
        return circ


class GateU(mcg.Gate):
    """Single qubit generic unitary gate.

    **Matrix representation:**

    .. math::
        \\operatorname{U}(\\theta, \\phi, \\lambda) = 
        \\frac{1}{2}
        \\begin{pmatrix}
        1 + e^{i\\theta} & -i e^{i\\lambda}(1 - e^{i\\theta}) \\\\
        i e^{i\\phi}(1 - e^{i\\theta}) & e^{i(\\phi + \\lambda)}(1 + e^{i\\theta})
        \\end{pmatrix}

    Parameters:
        theta (float): Euler angle 1 in radians.
        phi (float): Euler angle 2 in radians.
        lambda (float): Euler angle 3 in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, phi, lmbda = symbols('theta phi lambda')
        >>> GateU(theta, phi, lmbda)
        U(theta, phi, lambda)
        >>> GateU(theta, phi, lmbda).matrix()
        [1/2 + (1/2)*exp(I*theta), 1/2*I*exp(I*lambda)*(-1 + exp(I*theta))]
        [1/2*I*exp(I*phi)*(1 - exp(I*theta)), (1/2)*exp(I*(lambda + phi))*(1 + exp(I*theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateU(theta, phi, lmbda), 10)
        >>> c
        11-qubit circuit with 1 instructions:
        └── U(theta, phi, lambda) @ q10
        >>> GateU(theta, phi, lmbda).power(2), GateU(theta, phi, lmbda).inverse()
        (U(theta, phi, lambda)^(2), U(-theta, -lambda, -phi))
        >>> GateU(theta, phi, lmbda).decompose()
        1-qubit circuit with 1 instructions:
        └── U(theta, phi, lambda) @ q0
        >>> c.decompose()
        11-qubit circuit with 1 instructions:
        └── U(theta, phi, lambda) @ q10
    """
    _name = 'U'

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ('theta', 'phi', 'lmbda')

    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

    def matrix(self):
        return se.Matrix(sp.simplify(umatrix(self.theta, self.phi, self.lmbda)))

    def inverse(self):
        return GateU(-self.theta, -self.lmbda, -self.phi)
    
    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(self.theta, self.phi, self.lmbda), q)
        return circ
