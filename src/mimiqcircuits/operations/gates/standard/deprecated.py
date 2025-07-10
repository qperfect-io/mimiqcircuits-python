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

import mimiqcircuits.operations.gates.gate as mcg
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.matrices import pmatrix, umatrix
from symengine import pi


class GateU1(mcg.Gate):
    r"""Single qubit generic unitary gate :math:`{U_1}`.

    Equivalent to :func:`GateP`

    **Matrix representation:**

    .. math::
        \operatorname{U1}(\lambda) = \begin{pmatrix}
            1 & 0 \\
            0 & e^{i\lambda}
        \end{pmatrix}

    Parameters:
        lambda (float): Euler angle 3 in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateU1(lmbda)
        U1(lambda)
        >>> GateU1(lmbda).matrix()
        [1.0, 0]
        [0, exp(I*lambda)]
        <BLANKLINE>
        >>> c = Circuit().push(GateU1(lmbda), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── U1(lambda) @ q[0]
        <BLANKLINE>
        >>> GateU1(lmbda).power(2), GateU1(lmbda).inverse()
        (U1(2*lambda), U1(-lambda))
        >>> GateU1(lmbda).decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, lambda, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "U1"

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ("lmbda",)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def _matrix(self):
        return pmatrix(self.lmbda)

    def inverse(self):
        return GateU1(-self.lmbda)

    def _power(self, pwr):
        return GateU1(pwr * self.lmbda)

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(GateU(0, 0, self.lmbda), q)
        return circ


class GateU2(mcg.Gate):
    r"""Single qubit generic unitary gate :math:`{U_2}`.


    **Matrix representation:**

    .. math::
        \operatorname{U2}(\phi,\lambda) = \frac{1}{\sqrt{2}}e^{-(\phi+\lambda)/2}\begin{pmatrix}
            1 & -e^{i\lambda} \\
            e^{i\phi} & e^{i(\phi+\lambda)}
        \end{pmatrix}

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
        [0.707106781186548, -0.707106781186548*exp(I*lambda)]
        [0.707106781186548*exp(I*phi), 0.707106781186548*exp(I*(lambda + phi))]
        <BLANKLINE>
        >>> c = Circuit().push(GateU2(phi, lmbda), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── U2(phi, lambda) @ q[0]
        <BLANKLINE>
        >>> GateU2(phi, lmbda).power(2), GateU2(phi, lmbda).inverse()
        (U2(phi, lambda)**2, U2(-lambda - pi, -phi + pi))
        >>> GateU2(phi, lmbda).decompose()
        1-qubit circuit with 1 instructions:
        └── U((1/2)*pi, phi, lambda, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "U2"

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ("phi", "lmbda")

    def __init__(self, phi, lmbda):
        self.phi = phi
        self.lmbda = lmbda

    def _matrix(self):
        return umatrix(pi / 2, self.phi, self.lmbda)

    def inverse(self):
        return GateU2(-self.lmbda - pi, -self.phi + pi)

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(GateU(pi / 2, self.phi, self.lmbda), q)
        return circ


class GateU3(mcg.Gate):
    r"""Single qubit generic unitary gate :math:`{U_3}`.

    This gate is equivalent to :func:`GateU` up to a global phase, :math:`\operatorname{U3}(\theta,\phi,\lambda) = e^{-i(\phi + \lambda + \theta)/2} \operatorname{U}(\theta,\phi,\lambda)` 

    **Matrix representation:**

    .. math::
        \operatorname{U3}(\theta,\phi,\lambda) = \frac{1}{2}e^{-i(\phi + \lambda + \theta)/2}
        \begin{pmatrix}
        1 + e^{i\theta} & -i e^{i\lambda}(1 - e^{i\theta}) \\
        i e^{i\phi}(1 - e^{i\theta}) & e^{i(\phi + \lambda)}(1 + e^{i\theta})
        \end{pmatrix}

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
        [1.0*cos((1/2)*theta), -exp(I*lambda)*sin((1/2)*theta)]
        [exp(I*phi)*sin((1/2)*theta), exp(I*(lambda + phi))*cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateU3(theta, phi, lmbda), 0)
        >>> GateU3(theta, phi, lmbda).power(2), GateU3(theta, phi, lmbda).inverse()
        (U3(theta, phi, lambda)**2, U3(-theta, -lambda, -phi))
        >>> GateU3(theta, phi, lmbda).decompose()
        1-qubit circuit with 1 instructions:
        └── U(theta, phi, lambda, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "U3"

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ("theta", "phi", "lmbda")

    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

    def _matrix(self):
        return umatrix(self.theta, self.phi, self.lmbda)

    def inverse(self):
        return GateU3(-self.theta, -self.lmbda, -self.phi)

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(GateU(self.theta, self.phi, self.lmbda), q)
        return circ
