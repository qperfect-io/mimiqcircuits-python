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
from mimiqcircuits.operations.gates.standard.rotations import GateRX, GateRY, GateRZ
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.matrices import cis
from symengine import I, Matrix, cos, sin, pi


class GateRXX(mcg.Gate):
    """Two qubit RXX gate (rotation about XX).

    This gate is symmetric, and is maximally entangling at :math:`(\\theta = \\frac{\\pi}{2})`

    **Matrix representation:**

    .. math::
        \\operatorname{RXX}(\\theta) =\\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & 0 & 0 & -i\\sin(\\frac{\\theta}{2}) \\\\
            0 & \\cos(\\frac{\\theta}{2}) & -i\\sin(\\frac{\\theta}{2}) & 0 \\\\
            0 & -i\\sin(\\frac{\\theta}{2}) & \\cos(\\frac{\\theta}{2}) & 0 \\\\
            -i\\sin(\\frac{\\theta}{2}) & 0 & 0 & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta: The angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')
        >>> GateRXX(theta)
        RXX(theta)
        >>> GateRXX(theta).matrix()
        [cos((1/2)*theta), 0, 0, -I*sin((1/2)*theta)]
        [0, cos((1/2)*theta), -I*sin((1/2)*theta), 0]
        [0, -I*sin((1/2)*theta), cos((1/2)*theta), 0]
        [-I*sin((1/2)*theta), 0, 0, cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateRXX(theta), 0, 1)
        >>> GateRXX(theta).power(2), GateRXX(theta).inverse()
        (RXX(theta)^(2), RXX(-theta))
        (RXX(theta)^(2), RXX(-theta))
        >>> GateRXX(theta).matrix()
        [cos((1/2)*theta), 0, 0, -I*sin((1/2)*theta)]
        [0, cos((1/2)*theta), -I*sin((1/2)*theta), 0]
        [0, -I*sin((1/2)*theta), cos((1/2)*theta), 0]
        [-I*sin((1/2)*theta), 0, 0, cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateRXX(theta), 0, 1)
        >>> GateRXX(theta).power(2), GateRXX(theta).inverse()
        (RXX(theta)^(2), RXX(-theta))
        >>> GateRXX(theta).decompose()
    """
    _name = 'RXX'

    _num_qubits = 2
    _qragsizes = [2]

    _parnames = ('theta',)

    def __init__(self, theta):

        self.theta = theta

    def matrix(self):
        theta = self.theta
        cos2 = cos(theta / 2)
        sin2 = sin(theta / 2)

        return Matrix([[cos2, 0, 0, -I * sin2],
                       [0, cos2, -I * sin2, 0],
                       [0, -I * sin2, cos2, 0],
                       [-I * sin2, 0, 0, cos2]])

    def inverse(self):
        return GateRXX(-self.theta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(GateH(), a)
        circ.push(GateH(), b)
        circ.push(GateCX(), a, b)
        circ.push(GateRZ(self.theta), b)
        circ.push(GateCX(), a, b)
        circ.push(GateH(), b)
        circ.push(GateH(), a)
        return circ


class GateRYY(mcg.Gate):
    """Two qubit RYY gate (rotation about YY).

    This gate is symmetric, and is maximally entangling at :math:`(\\theta = \\frac{\\pi}{2})`

    **Matrix representation:**

    .. math::
        \\operatorname{RYY}(\\theta) =\\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & 0 & 0 & i\\sin(\\frac{\\theta}{2}) \\\\
            0 & \\cos(\\frac{\\theta}{2}) & -i\\sin(\\frac{\\theta}{2}) & 0 \\\\
            0 & -i\\sin(\\frac{\\theta}{2}) & \\cos(\\frac{\\theta}{2}) & 0 \\\\
            i\\sin(\\frac{\\theta}{2}) & 0 & 0 & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')o
        >>> GateRYY(theta)
        >>> GateRYY(theta).matrix()
        >>> c = Circuit().push(GateRYY(theta), 0, 1)
        >>> GateRYY(theta).power(2), GateRYY(theta).inverse()
        >>> GateRYY(theta).decompose()
    """
    _name = 'RYY'

    _num_qubits = 2
    _qragsizes = [2]

    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        theta = self.theta
        cos2 = cos(theta / 2)
        sin2 = sin(theta / 2)

        return Matrix([[cos2, 0, 0, I * sin2],
                       [0, cos2, -I * sin2, 0],
                       [0, -I * sin2, cos2, 0],
                       [I * sin2, 0, 0, cos2]])

    def inverse(self):
        return GateRYY(-self.theta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(GateRX(pi/2), a)
        circ.push(GateRX(pi/2), b)
        circ.push(GateCX(), a, b)
        circ.push(GateRZ(self.theta), b)
        circ.push(GateCX(), a, b)
        circ.push(GateRX(-pi/2), a)
        circ.push(GateRX(-pi/2), b)

        return circ


class GateRZZ(mcg.Gate):
    """Two qubit RZZ gate (rotation about ZZ)..

    This gate is symmetric, and is maximally entangling at :math:`(\\theta = \\frac{\\pi}{2})`

    **Matrix representation:**

    .. math::
        \\operatorname{RZZ}(\\theta) = \\begin{pmatrix}
            e^{-i\\frac{\\theta}{2}} & 0 & 0 & 0 \\\\
            0 & e^{i\\frac{\\theta}{2}} & 0 & 0 \\\\
            0 & 0 & e^{i\\frac{\\theta}{2}} & 0 \\\\
            0 & 0 & 0 & e^{-i\\frac{\\theta}{2}}
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')o
        >>> GateRZZ(theta)
        >>> GateRZZ(theta).matrix()
        >>> c = Circuit().push(GateRZZ(theta), 0, 1)
        >>> GateRZZ(theta).power(2), GateRZZ(theta).inverse()
        >>> GateRZZ(theta).decompose()
    """
    _num_qubits = 2
    _name = 'RZZ'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return Matrix([[cis(-self.theta / 2), 0, 0, 0],
                       [0, cis(self.theta / 2), 0, 0],
                       [0, 0, cis(self.theta / 2), 0],
                       [0, 0, 0, cis(-self.theta / 2)]])

    def inverse(self):
        return GateRZZ(-self.theta)

    def _decompose(self, circ, qubits, bits):
        a, b = qubits

        circ.push(GateCX(), a, b)
        circ.push(GateRZ(self.theta), b)
        circ.push(GateCX(), a, b)
        return circ


class GateRZX(mcg.Gate):
    """Two qubit RZX gate.

    This gate i is maximally entangling at :math:`(\\theta = \\frac{\\pi}{2})`

    **Matrix representation:**

    .. math::
        \\operatorname{RZX}(\\theta) =\\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & -i\\sin(\\frac{\\theta}{2}) & 0 & 0 \\\\
            -i\\sin(\\frac{\\theta}{2}) & \\cos(\\frac{\\theta}{2}) & 0 & 0 \\\\
            0 & 0 & \\cos(\\frac{\\theta}{2}) & i\\sin(\\frac{\\theta}{2}) \\\\
            0 & 0 & i\\sin(\\frac{\\theta}{2}) & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')o
        >>> GateRZX(theta)
        >>> GateRZX(theta).matrix()
        >>> c = Circuit().push(GateRZX(theta), 0, 1)
        >>> GateRZX(theta).power(2), GateRZX(theta).inverse()
        >>> GateRZX(theta).decompose()
    """
    _num_qubits = 2
    _name = 'RZX'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        theta = self.theta
        cos2 = cos(theta / 2)
        sin2 = sin(theta / 2)

        return Matrix([
            [cos2, -I * sin2, 0, 0],
            [-I * sin2, cos2, 0, 0],
            [0, 0, cos2, I * sin2],
            [0, 0, I * sin2, cos2]
        ])

    def inverse(self):
        return GateRZX(-self.theta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(GateH(), b)
        circ.push(GateCX(), a, b)
        circ.push(GateRZ(self.theta), b)
        circ.push(GateCX(), a, b)
        circ.push(GateH(), b)

        return circ


class GateXXplusYY(mcg.Gate):
    """Two qubit parametric XXplusYY gate.

    Also known as an XY gate. Its action is to induce a coherent rotation by some angle between :math:`\\ket{10}` and :math:`\\ket{01}`.

    **Matrix representation:**

    .. math::
        \\operatorname{XXplusYY}(\\theta, \\beta)= \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\cos(\\frac{\\theta}{2}) & -i\\sin(\\frac{\\theta}{2})e^{-i\\beta} & 0 \\\\
            0 & -i\\sin(\\frac{\\theta}{2})e^{i\\beta} & \\cos(\\frac{\\theta}{2}) & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    Parameters:
        theta: The angle in radians.
        beta: The phase angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, beta = symbols('theta beta')
        >>> GateXXplusYY(theta, beta)
        XXplusYY(theta, beta)
        >>> GateXXplusYY(theta, beta).matrix()
        [1, 0, 0, 0]
        [0, cos((1/2)*theta), -I*(-I*sin(beta) + cos(beta))*sin((1/2)*theta), 0]
        [0, -I*(I*sin(beta) + cos(beta))*sin((1/2)*theta), cos((1/2)*theta), 0]
        [0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateXXplusYY(theta, beta), 0, 1)
        >>> GateXXplusYY(theta, beta).power(2), GateXXplusYY(theta, beta).inverse()
        (XXplusYY(theta, beta)^(2), XXplusYY(-theta, beta))
        (XXplusYY(theta, beta)^(2), XXplusYY(-theta, beta))
        >>> GateXXplusYY(theta, beta).matrix()
        [1, 0, 0, 0]
        [0, cos((1/2)*theta), -I*(-I*sin(beta) + cos(beta))*sin((1/2)*theta), 0]
        [0, -I*(I*sin(beta) + cos(beta))*sin((1/2)*theta), cos((1/2)*theta), 0]
        [0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateXXplusYY(theta, beta), 0, 1)
        >>> GateXXplusYY(theta, beta).power(2), GateXXplusYY(theta, beta).inverse()
        (XXplusYY(theta, beta)^(2), XXplusYY(-theta, beta))
        >>> GateXXplusYY(theta, beta).decompose()
    """
    _num_qubits = 2
    _name = 'XXplusYY'
    _parnames = ('theta', 'beta')

    def __init__(self, theta, beta):
        self.theta = theta
        self.beta = beta

    def matrix(self):
        theta = self.theta
        beta = self.beta
        cos2 = cos(theta / 2)
        sin2 = sin(theta / 2)

        return Matrix([
            [1, 0, 0, 0],
            [0, cos2, -I * sin2 * cis(-beta), 0],
            [0, -I * sin2 * cis(beta), cos2, 0],
            [0, 0, 0, 1]
        ])

    def inverse(self):
        return GateXXplusYY(-self.theta, self.beta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(GateRZ(self.beta), a)
        circ.push(GateRZ(-pi/2), b)
        circ.push(GateSX(), b)
        circ.push(GateRZ(pi/2), b)
        circ.push(GateS(), a)
        circ.push(GateCX(), b, a)
        circ.push(GateRY(-self.theta/2), b)
        circ.push(GateRY(-self.theta/2), a)
        circ.push(GateCX(), b, a)
        circ.push(GateSDG(), a)
        circ.push(GateRZ(-pi/2), b)
        circ.push(GateSXDG(), b)
        circ.push(GateRZ(-pi/2), b)
        circ.push(GateRZ(-self.beta), a)

        return circ


class GateXXminusYY(mcg.Gate):
    """Two qubit parametric GateXXminusYY gate.

    Its action is to induce a coherent rotation by some angle between :math:`\\ket{00}` and :math:`\\ket{11}`

    **Matrix representation:**

    .. math::
        \\operatorname{XXminusYY}(\\theta, \\beta)=\\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & 0 & 0 & -i\\sin(\\frac{\\theta}{2})e^{-i\\beta} \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            -i\\sin(\\frac{\\theta}{2})e^{i\\beta} & 0 & 0 & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.
        beta (float): The angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, beta = symbols('theta beta')
        >>> GateXXminusYY(theta, beta)
        XXminusYY(theta, beta)
        >>> GateXXminusYY(theta, beta).matrix()
        [cos((1/2)*theta), 0, 0, -I*(I*sin(beta) + cos(beta))*sin((1/2)*theta)]
        [0, 1, 0, 0]
        [0, 0, 1, 0]
        [-I*(-I*sin(beta) + cos(beta))*sin((1/2)*theta), 0, 0, cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateXXminusYY(theta, beta), 0, 1)
        >>> GateXXminusYY(theta, beta).power(2), GateXXminusYY(theta, beta).inverse()
        (XXminusYY(theta, beta)^(2), XXminusYY(-theta, beta))
        (XXminusYY(theta, beta)^(2), XXminusYY(-theta, beta))
        >>> GateXXminusYY(theta, beta).matrix()
        [cos((1/2)*theta), 0, 0, -I*(I*sin(beta) + cos(beta))*sin((1/2)*theta)]
        [0, 1, 0, 0]
        [0, 0, 1, 0]
        [-I*(-I*sin(beta) + cos(beta))*sin((1/2)*theta), 0, 0, cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateXXminusYY(theta, beta), 0, 1)
        >>> GateXXminusYY(theta, beta).power(2), GateXXminusYY(theta, beta).inverse()
        (XXminusYY(theta, beta)^(2), XXminusYY(-theta, beta))
        >>> GateXXminusYY(theta, beta).decompose()
    """
    _num_qubits = 2
    _name = 'XXminusYY'
    _parnames = ('theta', 'beta')

    def __init__(self, theta, beta):
        self.theta = theta
        self.beta = beta

    def matrix(self):
        theta = self.theta
        beta = self.beta
        cos2 = cos(theta / 2)
        sin2 = sin(theta / 2)

        return Matrix([
            [cos2, 0, 0, -I * sin2 * cis(beta)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-I * sin2 * cis(-beta), 0, 0, cos2]
        ])

    def inverse(self):
        return GateXXminusYY(-self.theta, self.beta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(GateRZ(-self.beta), b)
        circ.push(GateRZ(-pi/2), a)
        circ.push(GateSX(), a)
        circ.push(GateRZ(pi/2), a)
        circ.push(GateS(), b)
        circ.push(GateCX(), a, b)
        circ.push(GateRY(self.theta/2), a)
        circ.push(GateRY(-self.theta/2), b)
        circ.push(GateCX(), a, b)
        circ.push(GateSDG(), b)
        circ.push(GateRZ(-pi/2), a)
        circ.push(GateSDG(), a)
        circ.push(GateRZ(pi/2), a)
        circ.push(GateRZ(self.beta), b)

        return circ
