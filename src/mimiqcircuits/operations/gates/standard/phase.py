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

import mimiqcircuits as mc
from mimiqcircuits.matrices import pmatrix
from mimiqcircuits.operations.utils import control_one_defined


class GateP(mc.Gate):
    r"""Single qubit Phase gate.

    **Matrix representation:**

    .. math::
        \operatorname{P}(\lambda) =
        \operatorname{U}(0,0,\lambda) =
        \begin{pmatrix}
            1 & 0 \\
            0 & \mathrm{e}^{i\lambda}
        \end{pmatrix}

    Parameters:
        lambda: Phase angle

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateP(lmbda)
        P(lambda)
        >>> GateP(lmbda).matrix()
        [1.0, 0]
        [0, exp(I*lambda)]
        <BLANKLINE>
        >>> c = Circuit().push(GateP(lmbda), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── P(lambda) @ q[0]
        <BLANKLINE>
        >>> GateP(lmbda).power(2), GateP(lmbda).inverse()
        (P(2*lambda), P(-lambda))
        >>> GateP(lmbda).decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, lambda, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "P"

    _num_qubits = 1
    _qregsizes = [1]

    _parnames = ("lmbda",)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def _matrix(self):
        return pmatrix(self.lmbda)

    def inverse(self):
        return GateP(-self.lmbda)

    def _power(self, p):
        return GateP(self.lmbda * p)

    def _control(self, n):
        return control_one_defined(
            n, self, mc.GateCP(self.lmbda), mc.GateCCP(self.lmbda)
        )

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits
        circ.push(
            mc.GateU(
                0,
                0,
                self.lmbda,
            ),
            q,
        )
        return circ
