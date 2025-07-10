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
from mimiqcircuits.matrices import umatrix
from mimiqcircuits.operations.utils import control_one_defined
import mimiqcircuits as mc
from sympy import I, pi, sin, cos, acos, Abs, simplify, exp, Expr, log, Matrix
import numpy as np
import sympy as sp
import symengine as se
from scipy.linalg import expm, logm


class GateU(mcg.Gate):
    r"""Single qubit generic unitary phase gate.


    **Matrix representation:**

    .. math::
        \operatorname{U}(\theta, \phi, \lambda, \gamma) =
        \mathrm{e}^{i\gamma}
        \begin{pmatrix}
            \cos\left(\frac{\theta}{2}\right) & -\mathrm{e}^{i\lambda}\sin\left(\frac{\theta}{2}\right)\\
            \mathrm{e}^{i\phi}\sin\left(\frac{\theta}{2}\right) & \mathrm{e}^{i(\phi+\lambda)}\cos\left (\frac{\theta}{2}\right)
        \end{pmatrix}

    Parameters:
        theta (float): Euler angle 1 in radians.
        phi (float): Euler angle 2 in radians.
        lambda (float): Euler angle 3 in radians.
        gamma (float, optional): Euler angle 4 in radians (default is 0).

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta, phi, lmbda, gamma = symbols('theta phi lambda gamma')
        >>> GateU(theta, phi, lmbda, gamma)
        U(theta, phi, lambda, gamma)
        >>> GateU(theta, phi, lmbda, gamma).matrix()
        [exp(I*gamma)*cos((1/2)*theta), -exp(I*(gamma + lambda))*sin((1/2)*theta)]
        [exp(I*(gamma + phi))*sin((1/2)*theta), exp(I*(gamma + lambda + phi))*cos((1/2)*theta)]
        <BLANKLINE>
        >>> c = Circuit().push(GateU(theta, phi, lmbda, gamma), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── U(theta, phi, lambda, gamma) @ q[0]
        <BLANKLINE>
        >>> GateU(theta, phi, lmbda, gamma).power(2), GateU(theta, phi, lmbda, gamma).inverse()
        (U(theta, phi, lambda, gamma)**2, U(-theta, -lambda, -phi, -gamma))
        >>> GateU(theta, phi, lmbda, gamma).decompose()
        1-qubit circuit with 1 instructions:
        └── U(theta, phi, lambda, gamma) @ q[0]
        <BLANKLINE>
        >>> c = Circuit().push(GateU(theta, phi, lmbda, gamma), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── U(theta, phi, lambda, gamma) @ q[0]
        <BLANKLINE>
        >>> GateU(theta, phi, lmbda, gamma).power(2), GateU(theta, phi, lmbda, gamma).inverse()
        (U(theta, phi, lambda, gamma)**2, U(-theta, -lambda, -phi, -gamma))
        >>> GateU(theta, phi, lmbda, gamma).decompose()
        1-qubit circuit with 1 instructions:
        └── U(theta, phi, lambda, gamma) @ q[0]
        <BLANKLINE>
    """

    _name = "U"

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ("theta", "phi", "lmbda", "gamma")

    def __init__(self, theta, phi, lmbda, gamma=0.0):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda
        self.gamma = gamma

    def _matrix(self):
        return umatrix(self.theta, self.phi, self.lmbda, self.gamma)

    def inverse(self):
        return GateU(-self.theta, -self.lmbda, -self.phi, -self.gamma)

    def _control(self, n):
        return control_one_defined(
            n, self, mc.GateCU(self.theta, self.phi, self.lmbda, self.gamma)
        )

    def _power(self, p):

        if self.is_symbolic():
            return mc.Power(self, p)

        def to_numeric(value):
            if (
                isinstance(value, (se.Basic, sp.Basic))
                and value == se.pi
                or value == se.pi
            ):
                return float(value)
            return value

        theta_value = to_numeric(self.theta)
        phi_value = to_numeric(self.phi)
        lambda_value = to_numeric(self.lmbda)
        gamma_value = to_numeric(self.gamma)

        numeric_gate = GateU(theta_value, phi_value, lambda_value, gamma_value)

        matrix = numeric_gate.matrix().tolist()

        matrix_np = np.array(self.convert_to_numeric(matrix))

        pow_matrix = expm(p * logm(matrix_np))

        # Compute the angles based on the resulting matrix and prevent error raising becouse of division by zero.
        with np.errstate(divide="ignore"):
            theta_p = 2 * np.arccos(np.abs(pow_matrix[0, 0]))
            gamma_p = np.angle(pow_matrix[0, 0])
            phi_p = np.angle(pow_matrix[1, 0] / np.sin(theta_p / 2)) - gamma_p
            lambda_p = np.angle(-pow_matrix[0, 1] / np.sin(theta_p / 2)) - gamma_p

        return GateU(theta_p, phi_p, lambda_p, gamma_p)

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(self, q)
        return circ

    def convert_to_numeric(self, matrix):
        """
        Convert a symbolic matrix to a numeric numpy array.
        """
        return np.array(
            [[complex(elem.evalf()) for elem in row] for row in matrix],
            dtype=np.complex128,
        )
