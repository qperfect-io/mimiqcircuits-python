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
from mimiqcircuits.operations.utils import power_idempotent, control_one_defined
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.gates.standard.u import GateU
import mimiqcircuits as mc
from symengine import pi, I, Matrix


class GateX(mcg.Gate):
    r""" Single qubit Pauli-X gate.

    **Matrix representation:**

    .. math::
        \operatorname{X} = \begin{pmatrix}
            0 & 1 \\
            1 & 0
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateX()
        X
        >>> GateX().matrix()
        [0, 1.0]
        [1.0, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateX(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── X @ q[0]
        <BLANKLINE>
        >>> GateX().power(2), GateX().inverse()
        (ID, X)
        >>> GateX().decompose()
        1-qubit circuit with 1 instructions:
        └── U(pi, 0, pi, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "X"

    _num_qubits = 1
    _qregsizes = [1]

    def inverse(self):
        return self

    def _power(self, p):
        pmod = p % 2

        # sqrt(X) = SX
        if pmod == 1 / 2:
            return mc.GateSX()

        # SX * SX^3 = X * X = ID => SX^3 = SXDG
        if pmod == 3 / 2:
            return mc.GateSXDG()

        # X^(2n) = ID
        # X^(2n + 1) = X
        return power_idempotent(self, p)

    def _control(self, n):
        return control_one_defined(n, self, mc.GateCX(), mc.GateCCX(), mc.GateC3X())

    def _matrix(self):
        return Matrix([[0, 1], [1, 0]])

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(GateU(pi, 0, pi), q)
        return circ


class GateY(mcg.Gate):
    r"""Single qubit Pauli-Y gate.

    **Matrix representation:**

    .. math::
        \operatorname{Y} = \begin{pmatrix}
            0 & -i \\
            i & 0
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateY()
        Y
        >>> GateY().matrix()
        [0, -0.0 - 1.0*I]
        [0.0 + 1.0*I, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateY(), 0)
        >>> GateY().power(2), GateY().inverse()
        (ID, Y)
        >>> GateY().decompose()
        1-qubit circuit with 1 instructions:
        └── U(pi, (1/2)*pi, (1/2)*pi, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "Y"

    _num_qubits = 1
    _qregsizes = [1]

    def inverse(self):
        return self

    def _power(self, p):
        # Y^(2n) = ID
        # Y^(2n + 1) = Y
        return power_idempotent(self, p)

    def _control(self, n):
        return control_one_defined(n, self, mc.GateCY())

    def _matrix(self):
        return Matrix([[0, -I], [I, 0]])

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(GateU(pi, pi / 2, pi / 2), q)
        return circ


class GateZ(mcg.Gate):
    r"""Single qubit Pauli-Z gate.

    **Matrix representation:**

    .. math::
        \operatorname{Z} = \begin{pmatrix}
            1 & 0 \\
            0 & -1
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateZ()
        Z
        >>> GateZ().matrix()
        [1.0, 0]
        [0, -1.0]
        <BLANKLINE>
        >>> c = Circuit().push(GateZ(), 0)
        >>> GateZ().power(2), GateZ().inverse()
        (ID, Z)
        >>> GateZ().decompose()
        1-qubit circuit with 1 instructions:
        └── P(pi) @ q[0]
        <BLANKLINE>
    """

    _name = "Z"

    _num_qubits = 1
    _qregsizes = [1]

    def inverse(self):
        return self

    def _power(self, p):
        pmod = p % 2

        # sqrt(Z) = S
        if pmod == 1 / 2:
            return mc.GateS()

        # S * S^3 = Z * Z = ID => S^3 = SDG
        if pmod == 3 / 2:
            return mc.GateSDG()

        # sqrt(S) = T
        if pmod == 1 / 4:
            return mc.GateT()

        # T * T^7 = S^2 * S^2 = Z * Z = ID => T^7 = TDG
        if pmod == 7 / 4:
            return mc.GateTDG()

        # Z^(2n) = ID
        # Z^(2n + 1) = Z
        return power_idempotent(self, p)

    def _control(self, n):
        return control_one_defined(n, self, mc.GateCZ())

    def _matrix(self):
        return Matrix([[1, 0], [0, -1]])

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(GateP(pi), q)
        return circ
