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
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.generalized.gphase import GPhase
from mimiqcircuits.matrices import pmatrix, umatrix, gphase
from symengine import pi


class GateU1(mcg.Gate):
    """Single qubit generic unitary gate :math:`{U_1}`.

    Equivalent to :func:`GateP`

    **Matrix representation:**

    .. math::
        \\operatorname{U1}(\\lambda) = \\begin{pmatrix}
            1 & 0 \\\\
            0 & e^{i\\lambda}
        \\end{pmatrix}

    Parameters:
        lambda (float): Euler angle 3 in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> lmbda = Symbol('lambda')
        >>> GateU1(lmbda)
        >>> GateU1(lmbda).matrix()
        >>> c = Circuit().push(GateU1(lmbda), 0)
        >>> GateU1(lmbda).power(2), GateU1(lmbda).inverse()
        >>> GateU1(lmbda).decompose()
    """
    _name = 'U1'

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return pmatrix(self.lmbda)

    def inverse(self):
        return GateU1(-self.lmbda)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, self.lmbda), q)
        return circ


class GateU2(mcg.Gate):
    """Single qubit generic unitary gate :math:`{U_2}`.

    Equivalent to :func:`GateU2DG`

    **Matrix representation:**

    .. math::
        \\operatorname{U2}(\\phi,\\lambda) = \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
            1 & -e^{i\\lambda} \\\\
            e^{i\\phi} & e^{i(\\phi+\\lambda)}
        \\end{pmatrix}

    Parameters:
        phi: Euler angle in radians.
        lambda: Euler angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> phi, lmbda = symbols('phi lambda')
        >>> GateU2(phi, lmbda)
        U2(phi, lambda)
        >>> GateU2(phi, lmbda).matrix()
        [(1/2 + 1/2*I)*(-I*sin((1/2)*(lambda + phi + (1/2)*pi)) + cos((1/2)*(lambda + phi + (1/2)*pi))), (-1/2 - 1/2*I)*(-I*sin((1/2)*(lambda + phi + (1/2)*pi)) + cos((1/2)*(lambda + phi + (1/2)*pi)))*(I*sin(lambda) + cos(lambda))]
        [(1/2 + 1/2*I)*(-I*sin((1/2)*(lambda + phi + (1/2)*pi)) + cos((1/2)*(lambda + phi + (1/2)*pi)))*(I*sin(phi) + cos(phi)), (1/2 + 1/2*I)*(I*sin(lambda + phi) + cos(lambda + phi))*(-I*sin((1/2)*(lambda + phi + (1/2)*pi)) + cos((1/2)*(lambda + phi + (1/2)*pi)))]
        <BLANKLINE>
        >>> c = Circuit().push(GateU2(phi, lmbda), 0)
        >>> GateU2(phi, lmbda).power(2), GateU2(phi, lmbda).inverse()
        (U2(phi, lambda)^(2), U2(-lambda - pi, -phi + pi))
        >>> GateU2(phi, lmbda).decompose()
        1-qubit circuit with 2 instructions:
        ├── GPhase(lmbda=(-1/2)*(lambda + phi + (1/2)*pi)) @ q0
        └── U((1/2)*pi, phi, lambda) @ q0
        1-qubit circuit with 2 instructions:
        ├── GPhase(lmbda=(-1/2)*(lambda + phi + (1/2)*pi)) @ q0
        └── U((1/2)*pi, phi, lambda) @ q0
        >>> GateU2(phi, lmbda).matrix()
        [(1/2 + 1/2*I)*(-I*sin((1/2)*(lambda + phi + (1/2)*pi)) + cos((1/2)*(lambda + phi + (1/2)*pi))), (-1/2 - 1/2*I)*(-I*sin((1/2)*(lambda + phi + (1/2)*pi)) + cos((1/2)*(lambda + phi + (1/2)*pi)))*(I*sin(lambda) + cos(lambda))]
        [(1/2 + 1/2*I)*(-I*sin((1/2)*(lambda + phi + (1/2)*pi)) + cos((1/2)*(lambda + phi + (1/2)*pi)))*(I*sin(phi) + cos(phi)), (1/2 + 1/2*I)*(I*sin(lambda + phi) + cos(lambda + phi))*(-I*sin((1/2)*(lambda + phi + (1/2)*pi)) + cos((1/2)*(lambda + phi + (1/2)*pi)))]
        <BLANKLINE>
        >>> c = Circuit().push(GateU2(phi, lmbda), 0)
        >>> GateU2(phi, lmbda).power(2), GateU2(phi, lmbda).inverse()
        (U2(phi, lambda)^(2), U2(-lambda - pi, -phi + pi))
        >>> GateU2(phi, lmbda).decompose()
        1-qubit circuit with 2 instructions:
        ├── GPhase(lmbda=(-1/2)*(lambda + phi + (1/2)*pi)) @ q0
        └── U((1/2)*pi, phi, lambda) @ q0
    """
    _name = 'U2'

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ('phi', 'lmbda')

    def __init__(self, phi, lmbda):
        self.phi = phi
        self.lmbda = lmbda

    def matrix(self):
        return gphase(-(self.phi + self.lmbda + pi/2)/2) * umatrix(pi/2, self.phi, self.lmbda)

    def inverse(self):
        return GateU2(-self.lmbda - pi, -self.phi + pi)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GPhase(1, -(self.lmbda + self.phi + pi/2)/2), q)
        circ.push(GateU(pi/2, self.phi, self.lmbda), q)
        return circ


class GateU3(mcg.Gate):
    """Single qubit generic unitary gate :math:`{U_3}`.

    **Matrix representation:**

    .. math::
        \\operatorname{U3}(\\theta,\\phi,\\lambda) = \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -e^{i\\lambda}\\sin\\frac{\\theta}{2} \\\\
            e^{i\\phi}\\sin\\frac{\\theta}{2} & e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta: Euler angle 1 in radians.
        phi: Euler angle 2 in radians.
        lambda: Euler angle 3 in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, phi, lmbda = symbols('theta phi lambda')
        >>> GateU3(theta, phi, lmbda)
        U3(theta, phi, lambda)
        >>> GateU3(theta, phi, lmbda).matrix()
        [(1/2)*(-I*sin((1/2)*(lambda + phi + theta)) + cos((1/2)*(lambda + phi + theta)))*(1 + I*sin(theta) + cos(theta)), -1/2*I*(-I*sin((1/2)*(lambda + phi + theta)) + cos((1/2)*(lambda + phi + theta)))*(I*sin(lambda) + cos(lambda))*(1 - (I*sin(theta) + cos(theta)))]
        [1/2*I*(-I*sin((1/2)*(lambda + phi + theta)) + cos((1/2)*(lambda + phi + theta)))*(I*sin(phi) + cos(phi))*(1 - (I*sin(theta) + cos(theta))), (1/2)*(I*sin(lambda + phi) + cos(lambda + phi))*(-I*sin((1/2)*(lambda + phi + theta)) + cos((1/2)*(lambda + phi + theta)))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateU3(theta, phi, lmbda), 0)
        >>> GateU3(theta, phi, lmbda).power(2), GateU3(theta, phi, lmbda).inverse()
        (U3(theta, phi, lambda)^(2), U3(-theta, -lambda, -phi))
        >>> GateU3(theta, phi, lmbda).decompose()
        1-qubit circuit with 2 instructions:
        ├── GPhase(lmbda=(-1/2)*(lambda + phi + theta)) @ q0
        └── U(theta, phi, lambda) @ q0
        1-qubit circuit with 2 instructions:
        ├── GPhase(lmbda=(-1/2)*(lambda + phi + theta)) @ q0
        └── U(theta, phi, lambda) @ q0
        >>> GateU3(theta, phi, lmbda).matrix()
        [(1/2)*(-I*sin((1/2)*(lambda + phi + theta)) + cos((1/2)*(lambda + phi + theta)))*(1 + I*sin(theta) + cos(theta)), -1/2*I*(-I*sin((1/2)*(lambda + phi + theta)) + cos((1/2)*(lambda + phi + theta)))*(I*sin(lambda) + cos(lambda))*(1 - (I*sin(theta) + cos(theta)))]
        [1/2*I*(-I*sin((1/2)*(lambda + phi + theta)) + cos((1/2)*(lambda + phi + theta)))*(I*sin(phi) + cos(phi))*(1 - (I*sin(theta) + cos(theta))), (1/2)*(I*sin(lambda + phi) + cos(lambda + phi))*(-I*sin((1/2)*(lambda + phi + theta)) + cos((1/2)*(lambda + phi + theta)))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateU3(theta, phi, lmbda), 0)
        >>> GateU3(theta, phi, lmbda).power(2), GateU3(theta, phi, lmbda).inverse()
        (U3(theta, phi, lambda)^(2), U3(-theta, -lambda, -phi))
        >>> GateU3(theta, phi, lmbda).decompose()
        1-qubit circuit with 2 instructions:
        ├── GPhase(lmbda=(-1/2)*(lambda + phi + theta)) @ q0
        └── U(theta, phi, lambda) @ q0
    """
    _name = 'U3'

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ('theta', 'phi', 'lmbda')

    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

    def matrix(self):
        return gphase(-(self.phi + self.lmbda + self.theta) / 2) * umatrix(self.theta, self.phi, self.lmbda)

    def inverse(self):
        return GateU3(-self.theta, -self.lmbda, -self.phi)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GPhase(1, -(self.theta+self.phi + self.lmbda)/2), q)
        circ.push(GateU(self.theta, self.phi, self.lmbda), q)
        return circ
