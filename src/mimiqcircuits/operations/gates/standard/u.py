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
"""U (Generic Unitary) gate."""

import mimiqcircuits.operations.gates.gate as mcg
from mimiqcircuits.matrices import umatrix
from mimiqcircuits import numerics as _nm
from mimiqcircuits.operations.utils import control_one_defined
from mimiqcircuits.operations.power import Power
import numpy as np
import symengine as se
from mimiqcircuits._sympycheck import _is_basic


def _negate(x):
    # Adding `0.0` turns a negated float zero into `+0.0`, so that `-0.0` and
    # `0.0` serialize to the same bytes.
    return -x + 0.0 if isinstance(x, float) else -x


class GateU(mcg.Gate):
    r"""Single qubit generic unitary gate with a global phase.

    Any single qubit unitary can be written as a ``GateU``. The fourth
    angle ``gamma`` sets a global phase.

    Matrix Representation:
        .. math::
            \operatorname{U}(\theta, \phi, \lambda, \gamma) =
            \mathrm{e}^{i\gamma}
            \begin{pmatrix}
                \cos\left(\frac{\theta}{2}\right) & -\mathrm{e}^{i\lambda}\sin\left(\frac{\theta}{2}\right)\\
                \mathrm{e}^{i\phi}\sin\left(\frac{\theta}{2}\right) & \mathrm{e}^{i(\phi+\lambda)}\cos\left (\frac{\theta}{2}\right)
            \end{pmatrix}

    Args:
        theta: Euler angle :math:`\theta` in radians.
        phi: Euler angle :math:`\phi` in radians.
        lmbda: Euler angle :math:`\lambda` in radians.
        gamma: Global phase :math:`\gamma` in radians. Defaults to ``0.0``.

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
        1-qubit circuit with 1 instruction:
        └── U(theta, phi, lambda, gamma) @ q[0]
        <BLANKLINE>
        >>> GateU(theta, phi, lmbda, gamma).power(2), GateU(theta, phi, lmbda, gamma).inverse()
        (U(theta, phi, lambda, gamma)**2, U(-theta, -lambda, -phi, -gamma))
        >>> GateU(theta, phi, lmbda, gamma).decompose()
        1-qubit circuit with 1 instruction:
        └── U(theta, phi, lambda, gamma) @ q[0]
        <BLANKLINE>

    Note:
        A power of a ``GateU`` with symbolic angles is wrapped in
        :class:`Power`. With numeric angles it is computed numerically and
        returned as a new ``GateU``.
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

    def _matrix_numeric(self, theta, phi, lmbda, gamma):
        return _nm.umatrix(theta, phi, lmbda, gamma)

    def inverse(self):
        return GateU(
            _negate(self.theta), _negate(self.lmbda), _negate(self.phi), _negate(self.gamma)
        )

    def _control(self, n):
        from mimiqcircuits.operations.gates.standard.cu import GateCU
        return control_one_defined(
            n, self, GateCU(self.theta, self.phi, self.lmbda, self.gamma)
        )

    def _power(self, p):
        if self.is_symbolic():
            return Power(self, p)

        def to_numeric(value):
            if (
                _is_basic(value)
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

        from scipy.linalg import expm, logm

        pow_matrix = expm(float(p) * logm(matrix_np))

        theta_p = 2 * np.arccos(np.clip(np.abs(pow_matrix[0, 0]), 0, 1))
        gamma_p = np.angle(pow_matrix[0, 0])
        if abs(np.sin(theta_p / 2)) < 1e-12:
            # Diagonal matrix: the off-diagonal entries are zero, so phi and
            # lambda cannot be read from them. Only phi + lambda is fixed (by
            # the entry [1, 1] = e^{i(phi+lambda+gamma)}), so pick phi = 0 and
            # put the whole phase in lambda.
            phi_p = 0.0
            lambda_p = np.angle(pow_matrix[1, 1]) - gamma_p
        else:
            phi_p = np.angle(pow_matrix[1, 0] / np.sin(theta_p / 2)) - gamma_p
            lambda_p = np.angle(-pow_matrix[0, 1] / np.sin(theta_p / 2)) - gamma_p

        return GateU(theta_p, phi_p, lambda_p, gamma_p)

    def supports_canonical_rewrite(self):
        """Return ``False``: U is a primitive of the canonical gate set."""
        return False

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(self, q)
        return circ

    def convert_to_numeric(self, matrix):
        """Convert a nested list of SymEngine numbers to a ``complex128`` array.

        Args:
            matrix: Rows of numeric SymEngine expressions, as returned by
                ``matrix().tolist()``.

        Returns:
            numpy.ndarray: The same entries as complex floats.
        """
        return np.array(
            [[complex(elem.evalf()) for elem in row] for row in matrix],
            dtype=np.complex128,
        )
