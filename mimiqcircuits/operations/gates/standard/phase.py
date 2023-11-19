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
from mimiqcircuits.matrices import pmatrix
from symengine import Matrix
import sympy as sp


class GateP(mc.Gate):
    """Single qubit Phase gate.

    **Matrix representation:**

    .. math::
        \\operatorname{P}(\\lambda) = \\begin{pmatrix}
            1 & 0 \\\\
            0 & e^{i\\lambda}
        \\end{pmatrix}

    Parameters:
        lambda: Phase angle

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateP(lmbda)
        P(lambda)
        >>> GateP(lmbda).matrix()
        [1, 0]
        [0, exp(I*lambda)]
        <BLANKLINE>
        >>> c = Circuit().push(GateP(lmbda), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── P(lambda) @ q0
        >>> GateP(lmbda).power(2), GateP(lmbda).inverse()
        (P(2*lambda), P(-lambda))
        >>> GateP(lmbda).decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, lambda) @ q0
    """
    _name = 'P'

    _num_qubits = 1
    _qragsizes = [1]

    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return Matrix(sp.simplify((pmatrix(self.lmbda))))

    def inverse(self):
        return GateP(-self.lmbda)

    def power(self, p):
        return GateP(self.lmbda * p)

    def _decompose(self, circ, qubits, bits):
        q = qubits
        circ.push(mc.GateU(0, 0, self.lmbda), q)
        return circ
