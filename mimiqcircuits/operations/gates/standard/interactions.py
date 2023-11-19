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

import mimiqcircuits as mc
from mimiqcircuits.matrices import cis
from symengine import I, Matrix, cos, sin, pi, expand
import sympy as sp


class GateRXX(mc.Gate):
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
        >>> c = Circuit()
        >>> c.push(GateRXX(theta), 0, 1)
        2-qubit circuit with 1 instructions:
        └── RXX(theta) @ q0, q1
        >>> GateRXX(theta).power(2), GateRXX(theta).inverse()
        (RXX(theta)^(2), RXX(-theta))
        >>> GateRXX(theta).matrix()
        [cos((1/2)*theta), 0, 0, -I*sin((1/2)*theta)]
        [0, cos((1/2)*theta), -I*sin((1/2)*theta), 0]
        [0, -I*sin((1/2)*theta), cos((1/2)*theta), 0]
        [-I*sin((1/2)*theta), 0, 0, cos((1/2)*theta)]
        <BLANKLINE>
        >>> GateRXX(theta).decompose()
        2-qubit circuit with 7 instructions:
        ├── H @ q0
        ├── H @ q1
        ├── CX @ q0, q1
        ├── RZ(theta) @ q1
        ├── CX @ q0, q1
        ├── H @ q1
        └── H @ q0
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

        return Matrix(sp.simplify((Matrix([[cos2, 0, 0, -I * sin2],
                                           [0, cos2, -I * sin2, 0],
                                           [0, -I * sin2, cos2, 0],
                                           [-I * sin2, 0, 0, cos2]]))))

    def inverse(self):
        return GateRXX(-self.theta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(mc.GateH(), a)
        circ.push(mc.GateH(), b)
        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateRZ(self.theta), b)
        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateH(), b)
        circ.push(mc.GateH(), a)
        return circ


class GateRYY(mc.Gate):
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
        >>> theta = Symbol('theta')
        >>> GateRYY(theta)
        RYY(theta)
        >>> GateRYY(theta).matrix()
        [cos((1/2)*theta), 0, 0, I*sin((1/2)*theta)]
        [0, cos((1/2)*theta), -I*sin((1/2)*theta), 0]
        [0, -I*sin((1/2)*theta), cos((1/2)*theta), 0]
        [I*sin((1/2)*theta), 0, 0, cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateRYY(theta), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── RYY(theta) @ q0, q1
        >>> GateRYY(theta).power(2), GateRYY(theta).inverse()
        (RYY(theta)^(2), RYY(-theta))
        >>> GateRYY(theta).decompose()
        2-qubit circuit with 7 instructions:
        ├── RX((1/2)*pi) @ q0
        ├── RX((1/2)*pi) @ q1
        ├── CX @ q0, q1
        ├── RZ(theta) @ q1
        ├── CX @ q0, q1
        ├── RX((-1/2)*pi) @ q0
        └── RX((-1/2)*pi) @ q1
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

        return Matrix(sp.simplify((Matrix([[cos2, 0, 0, I * sin2],
                                           [0, cos2, -I * sin2, 0],
                                           [0, -I * sin2, cos2, 0],
                                           [I * sin2, 0, 0, cos2]]))))

    def inverse(self):
        return GateRYY(-self.theta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(mc.GateRX(pi/2), a)
        circ.push(mc.GateRX(pi/2), b)
        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateRZ(self.theta), b)
        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateRX(-pi/2), a)
        circ.push(mc.GateRX(-pi/2), b)

        return circ


class GateRZZ(mc.Gate):
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
        >>> theta = Symbol('theta')
        >>> GateRZZ(theta)
        RZZ(theta)
        >>> GateRZZ(theta).matrix()
        [exp(-1/2*I*theta), 0, 0, 0]
        [0, exp(1/2*I*theta), 0, 0]
        [0, 0, exp(1/2*I*theta), 0]
        [0, 0, 0, exp(-1/2*I*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateRZZ(theta), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── RZZ(theta) @ q0, q1
        >>> GateRZZ(theta).power(2), GateRZZ(theta).inverse()
        (RZZ(theta)^(2), RZZ(-theta))
        >>> GateRZZ(theta).decompose()
        2-qubit circuit with 3 instructions:
        ├── CX @ q0, q1
        ├── RZ(theta) @ q1
        └── CX @ q0, q1
    """
    _num_qubits = 2
    _name = 'RZZ'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return Matrix(sp.simplify(Matrix([[cis(-self.theta / 2), 0, 0, 0],
                                          [0, cis(self.theta / 2), 0, 0],
                                          [0, 0, cis(self.theta / 2), 0],
                                          [0, 0, 0, cis(-self.theta / 2)]])))

    def inverse(self):
        return GateRZZ(-self.theta)

    def _decompose(self, circ, qubits, bits):
        a, b = qubits

        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateRZ(self.theta), b)
        circ.push(mc.GateCX(), a, b)
        return circ


class GateRZX(mc.Gate):
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
        >>> theta = Symbol('theta')
        >>> GateRZX(theta)
        RZX(theta)
        >>> GateRZX(theta).matrix()
        [cos((1/2)*theta), -I*sin((1/2)*theta), 0, 0]
        [-I*sin((1/2)*theta), cos((1/2)*theta), 0, 0]
        [0, 0, cos((1/2)*theta), I*sin((1/2)*theta)]
        [0, 0, I*sin((1/2)*theta), cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateRZX(theta), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── RZX(theta) @ q0, q1
        >>> GateRZX(theta).power(2), GateRZX(theta).inverse()
        (RZX(theta)^(2), RZX(-theta))
        >>> GateRZX(theta).decompose()
        2-qubit circuit with 5 instructions:
        ├── H @ q1
        ├── CX @ q0, q1
        ├── RZ(theta) @ q1
        ├── CX @ q0, q1
        └── H @ q1
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

        return Matrix(sp.simplify(Matrix([
            [cos2, -I * sin2, 0, 0],
            [-I * sin2, cos2, 0, 0],
            [0, 0, cos2, I * sin2],
            [0, 0, I * sin2, cos2]
        ])))

    def inverse(self):
        return GateRZX(-self.theta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(mc.GateH(), b)
        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateRZ(self.theta), b)
        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateH(), b)

        return circ


class GateXXplusYY(mc.Gate):
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
        [0, cos((1/2)*theta), (sin(beta) - I*cos(beta))*sin((1/2)*theta), 0]
        [0, I*(I*sin(beta) - cos(beta))*sin((1/2)*theta), cos((1/2)*theta), 0]
        [0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateXXplusYY(theta, beta), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── XXplusYY(theta, beta) @ q0, q1
        >>> GateXXplusYY(theta, beta).power(2), GateXXplusYY(theta, beta).inverse()
        (XXplusYY(theta, beta)^(2), XXplusYY(-theta, beta))
        >>> GateXXplusYY(theta, beta).decompose()
        2-qubit circuit with 14 instructions:
        ├── RZ(beta) @ q0
        ├── RZ((-1/2)*pi) @ q1
        ├── X^(1/2) @ q1
        ├── RZ((1/2)*pi) @ q1
        ├── S @ q0
        ├── CX @ q1, q0
        ├── RY((-1/2)*theta) @ q1
        ├── RY((-1/2)*theta) @ q0
        ├── CX @ q1, q0
        ├── S† @ q0
        ├── RZ((-1/2)*pi) @ q1
        ├── (X^(1/2))† @ q1
        ├── RZ((-1/2)*pi) @ q1
        └── RZ(-beta) @ q0
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

        return Matrix(sp.simplify(Matrix([
            [1, 0, 0, 0],
            [0, cos2, -I * sin2 * cis(beta), 0],
            [0, -I * sin2 * cis(-beta), cos2, 0],
            [0, 0, 0, 1]
        ])))

    def inverse(self):
        return GateXXplusYY(-self.theta, self.beta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(mc.GateRZ(self.beta), a)
        circ.push(mc.GateRZ(-pi/2), b)
        circ.push(mc.GateSX(), b)
        circ.push(mc.GateRZ(pi/2), b)
        circ.push(mc.GateS(), a)
        circ.push(mc.GateCX(), b, a)
        circ.push(mc.GateRY(-self.theta/2), b)
        circ.push(mc.GateRY(-self.theta/2), a)
        circ.push(mc.GateCX(), b, a)
        circ.push(mc.GateSDG(), a)
        circ.push(mc.GateRZ(-pi/2), b)
        circ.push(mc.GateSXDG(), b)
        circ.push(mc.GateRZ(-pi/2), b)
        circ.push(mc.GateRZ(-self.beta), a)

        return circ


class GateXXminusYY(mc.Gate):
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
        [cos((1/2)*theta), 0, 0, I*(I*sin(beta) - cos(beta))*sin((1/2)*theta)]
        [0, 1, 0, 0]
        [0, 0, 1, 0]
        [(sin(beta) - I*cos(beta))*sin((1/2)*theta), 0, 0, cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateXXminusYY(theta, beta), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── XXminusYY(theta, beta) @ q0, q1
        >>> GateXXminusYY(theta, beta).power(2), GateXXminusYY(theta, beta).inverse()
        (XXminusYY(theta, beta)^(2), XXminusYY(-theta, beta))
        >>> GateXXminusYY(theta, beta).decompose()
        2-qubit circuit with 14 instructions:
        ├── RZ(-beta) @ q1
        ├── RZ((-1/2)*pi) @ q0
        ├── X^(1/2) @ q0
        ├── RZ((1/2)*pi) @ q0
        ├── S @ q1
        ├── CX @ q0, q1
        ├── RY((1/2)*theta) @ q0
        ├── RY((-1/2)*theta) @ q1
        ├── CX @ q0, q1
        ├── S† @ q1
        ├── RZ((-1/2)*pi) @ q0
        ├── S† @ q0
        ├── RZ((1/2)*pi) @ q0
        └── RZ(beta) @ q1
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

        return Matrix(sp.simplify(Matrix([
            [cos2, 0, 0, -I * sin2 * cis(-beta)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-I * sin2 * cis(beta), 0, 0, cos2]
        ])))

    def inverse(self):
        return GateXXminusYY(-self.theta, self.beta)

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(mc.GateRZ(-self.beta), b)
        circ.push(mc.GateRZ(-pi/2), a)
        circ.push(mc.GateSX(), a)
        circ.push(mc.GateRZ(pi/2), a)
        circ.push(mc.GateS(), b)
        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateRY(self.theta/2), a)
        circ.push(mc.GateRY(-self.theta/2), b)
        circ.push(mc.GateCX(), a, b)
        circ.push(mc.GateSDG(), b)
        circ.push(mc.GateRZ(-pi/2), a)
        circ.push(mc.GateSDG(), a)
        circ.push(mc.GateRZ(pi/2), a)
        circ.push(mc.GateRZ(self.beta), b)

        return circ
