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
from mimiqcircuits.matrices import pmatrix


class GateP(mcg.Gate):
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
        >>> lmbda = Symbol('lambda')
        >>> GateP(lmbda)
        >>> GateP(lmbda).matrix()
        >>> c = Circuit().push(GateP(lmbda), 0)
        >>> GateP(lmbda).power(2), GateP(lmbda).inverse()
        >>> GateP(lmbda).decompose()
    """
    _name = 'P'

    _num_qubits = 1
    _qragsizes = [1]

    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return pmatrix(self.lmbda)

    def inverse(self):
        return GateP(-self.lmbda)

    def power(self, p):
        return GateP(self.lmbda * p)

    def _decompose(self, circ, qubits, bits):
        q = qubits
        circ.push(GateU(0, 0, self.lmbda), q)
        return circ
