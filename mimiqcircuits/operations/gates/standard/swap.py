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
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from symengine import Matrix
import sympy as sp


class GateSWAP(mcg.Gate):
    """Two qubit SWAP gate.

    See Also :func:`GateISWAP`

    **Matrix representation:**

    .. math::
        \\operatorname{SWAP} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSWAP()
        SWAP
        >>> GateSWAP().matrix()
        [1, 0, 0, 0]
        [0, 0, 1, 0]
        [0, 1, 0, 0]
        [0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateSWAP(), 0, 1)
        >>> GateSWAP().power(2), GateSWAP().inverse()
        (SWAP^(2), SWAP)
        >>> GateSWAP().decompose()
        2-qubit circuit with 3 instructions:
        ├── CX @ q0, q1
        ├── CX @ q1, q0
        └── CX @ q0, q1
    """
    _name = 'SWAP'

    _num_qubits = 2
    _qregsizes = [2]

    def matrix(self):
        return Matrix(sp.simplify(Matrix([
            [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]
        ])))

    def inverse(self):
        return self

    def _decompose(self, circ, qubits, bits):
        c, t = qubits
        circ.push(GateCX(), c, t)
        circ.push(GateCX(), t, c)
        circ.push(GateCX(), c, t)
        return circ
