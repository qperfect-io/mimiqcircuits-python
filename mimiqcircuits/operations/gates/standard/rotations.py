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
from mimiqcircuits.operations.gates.standard.deprecated import GateU3
from mimiqcircuits.matrices import umatrix, gphase
from symengine import pi


class GateRX(mcg.Gate):
    """Single qubit Rotation gate around the axis :math:`\\hat{x}`

    **Matrix representation:**

    .. math::
        \\operatorname{RX}(\\theta) = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
            -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta: Rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')
        >>> GateRX(theta)
        RX(theta)
        >>> GateRX(theta).matrix()
        [(1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta)), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - (I*sin(theta) + cos(theta)))]
        [(1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - (I*sin(theta) + cos(theta))), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateRX(theta), 0)
        >>> GateRX(theta).power(2), GateRX(theta).inverse()
        (RX(2*theta), RX(-theta))
        >>> GateRX(theta).decompose()
        1-qubit circuit with 2 instructions:
        ├── U(theta, (-1/2)*pi, (1/2)*pi) @ q0
        └── GPhase(lmbda=(-1/2)*theta) @ q0
        1-qubit circuit with 2 instructions:
        ├── U(theta, (-1/2)*pi, (1/2)*pi) @ q0
        └── GPhase(lmbda=(-1/2)*theta) @ q0
        >>> GateRX(theta).matrix()
        [(1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta)), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - (I*sin(theta) + cos(theta)))]
        [(1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - (I*sin(theta) + cos(theta))), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateRX(theta), 0)
        >>> GateRX(theta).power(2), GateRX(theta).inverse()
        (RX(2*theta), RX(-theta))
        >>> GateRX(theta).decompose()
        1-qubit circuit with 2 instructions:
        ├── U(theta, (-1/2)*pi, (1/2)*pi) @ q0
        └── GPhase(lmbda=(-1/2)*theta) @ q0
    """
    _name = 'RX'

    _num_qubits = 1
    _qragsizes = [1]

    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return gphase(-self.theta / 2) * umatrix(self.theta, -pi / 2, pi / 2)

    def inverse(self):
        return GateRX(-self.theta)

    def power(self, p):
        return GateRX(self.theta * p)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(self.theta, -pi/2, pi/2), q)
        circ.push(GPhase(1, -self.theta/2), q)
        return circ


class GateRY(mcg.Gate):
    """Single qubit Rotation gate around the axis :math:`\\hat{y}`

    **Matrix representation:**

    .. math::
        \\operatorname{RY}(\\theta) = \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -\\sin\\frac{\\theta}{2} \\\\
            \\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): Rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')
        >>> GateRY(theta)
        RY(theta)
        >>> GateRY(theta).matrix()
        [(1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta)), -1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - (I*sin(theta) + cos(theta)))]
        [1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - (I*sin(theta) + cos(theta))), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateRY(theta), 0)
        >>> GateRY(theta).power(2), GateRY(theta).inverse()
        (RY(2*theta), RY(-theta))
        >>> GateRY(theta).decompose()
        1-qubit circuit with 2 instructions:
        ├── U(theta, 0, 0) @ q0
        └── GPhase(lmbda=(-1/2)*theta) @ q0
        1-qubit circuit with 2 instructions:
        ├── U(theta, 0, 0) @ q0
        └── GPhase(lmbda=(-1/2)*theta) @ q0
        >>> GateRY(theta).matrix()
        [(1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta)), -1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - (I*sin(theta) + cos(theta)))]
        [1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - (I*sin(theta) + cos(theta))), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateRY(theta), 0)
        >>> GateRY(theta).power(2), GateRY(theta).inverse()
        (RY(2*theta), RY(-theta))
        >>> GateRY(theta).decompose()
        1-qubit circuit with 2 instructions:
        ├── U(theta, 0, 0) @ q0
        └── GPhase(lmbda=(-1/2)*theta) @ q0
    """
    _name = 'RY'

    _num_qubits = 1
    _qragsizes = [1]

    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return gphase(-self.theta / 2) * umatrix(self.theta, 0, 0)

    def inverse(self):
        return GateRY(-self.theta)

    def power(self, p):
        return GateRY(self.theta * p)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(self.theta, 0, 0), q)
        circ.push(GPhase(1, -self.theta/2), q)
        return circ


class GateRZ(mcg.Gate):
    """Single qubit Rotation gate around the axis :math:`\\hat{z}`

    **Matrix representation:**

    .. math::
        \\operatorname{RZ}(\\lambda) = \\begin{pmatrix}
            e^{-i\\frac{\\lambda}{2}} & 0 \\\\
            0 & e^{i\\frac{\\lambda}{2}}
        \\end{pmatrix}

    Parameters:
        lambda: Rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateRZ(lmbda)
        RZ(lambda)
        >>> GateRZ(lmbda).matrix()
        [-I*sin((1/2)*lambda) + cos((1/2)*lambda), 0]
        [0, (-I*sin((1/2)*lambda) + cos((1/2)*lambda))*(I*sin(lambda) + cos(lambda))]
        <BLANKLINE>
        >>> c = Circuit().push(GateRZ(lmbda), 0)
        >>> GateRZ(lmbda).power(2), GateRZ(lmbda).inverse()
        (RZ(2*lambda), RZ(-lambda))
        >>> GateRZ(lmbda).decompose()
        1-qubit circuit with 2 instructions:
        ├── U(0, 0, lambda) @ q0
        └── GPhase(lmbda=(-1/2)*lambda) @ q0
        1-qubit circuit with 2 instructions:
        ├── U(0, 0, lambda) @ q0
        └── GPhase(lmbda=(-1/2)*lambda) @ q0
        >>> GateRZ(lmbda).matrix()
        [-I*sin((1/2)*lambda) + cos((1/2)*lambda), 0]
        [0, (-I*sin((1/2)*lambda) + cos((1/2)*lambda))*(I*sin(lambda) + cos(lambda))]
        <BLANKLINE>
        >>> c = Circuit().push(GateRZ(lmbda), 0)
        >>> GateRZ(lmbda).power(2), GateRZ(lmbda).inverse()
        (RZ(2*lambda), RZ(-lambda))
        >>> GateRZ(lmbda).decompose()
        1-qubit circuit with 2 instructions:
        ├── U(0, 0, lambda) @ q0
        └── GPhase(lmbda=(-1/2)*lambda) @ q0
    """
    _name = 'RZ'

    _num_qubits = 1
    _qragsizes = [1]

    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return gphase(-self.lmbda / 2) * umatrix(0, 0, self.lmbda)

    def inverse(self):
        return GateRZ(-self.lmbda)

    def power(self, p):
        return GateRZ(self.lmbda * p)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(0, 0, self.lmbda), q)
        circ.push(GPhase(1, -self.lmbda/2), q)
        return circ


class GateR(mcg.Gate):
    """Single qubit Rotation gate around the axis :math:`\\cos(\\phi)\\hat{x} + \\sin(\\phi)\\hat{y}`.

    **Matrix representation:**

    .. math::
        \\operatorname R(\\theta,\\phi) = \\begin{pmatrix}
          \\cos \\frac{\\theta}{2}  & -i e^{-i\\phi} \\sin \\frac{\\theta}{2} \\\\
          -i e^{i \\phi} \\sin \\frac{\\theta}{2}  &  \\cos \\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): The rotation angle in radians.
        phi (float): The axis of rotation in radians.

    Example:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, phi = symbols('theta phi')
        >>> GateR(theta, phi)
        R(theta, phi)
        >>> GateR(theta, phi).matrix()
        [(1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta)), -1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(sin(phi) + I*cos(phi))*(1 - (I*sin(theta) + cos(theta)))]
        [1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(sin(phi) - I*cos(phi))*(1 - (I*sin(theta) + cos(theta))), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateR(theta, phi), 0)
        >>> GateR(theta, phi).power(2), GateR(theta, phi).inverse()
        (R(2*theta, phi), R(-theta, phi))
        >>> GateR(theta, phi).decompose()
        1-qubit circuit with 1 instructions:
        └── U3(theta, phi + (-1/2)*pi, -phi + (1/2)*pi) @ q0
        1-qubit circuit with 1 instructions:
        └── U3(theta, phi + (-1/2)*pi, -phi + (1/2)*pi) @ q0
        >>> GateR(theta, phi).matrix()
        [(1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta)), -1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(sin(phi) + I*cos(phi))*(1 - (I*sin(theta) + cos(theta)))]
        [1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(sin(phi) - I*cos(phi))*(1 - (I*sin(theta) + cos(theta))), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateR(theta, phi), 0)
        >>> GateR(theta, phi).power(2), GateR(theta, phi).inverse()
        (R(2*theta, phi), R(-theta, phi))
        >>> GateR(theta, phi).decompose()
        1-qubit circuit with 1 instructions:
        └── U3(theta, phi + (-1/2)*pi, -phi + (1/2)*pi) @ q0
    """
    _name = 'R'

    _num_qubits = 1
    _qragsizes = [1]

    _parnames = ('theta', 'phi')

    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

    def matrix(self):
        return GateU3(self.theta, self.phi - pi / 2, -self.phi+pi / 2).matrix()

    def inverse(self):
        return GateR(-self.theta, self.phi)

    def power(self, p):
        return GateR(self.theta * p, self.phi)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU3(self.theta, self.phi - pi/2, -self.phi + pi/2), q)
        return circ
