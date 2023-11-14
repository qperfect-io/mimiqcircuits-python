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
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.s import GateS
from symengine import Matrix, I


class GateISWAP(mcg.Gate):
    """ Two qubit ISWAP gate.

    See Also :func:`GateISWAPDG` and :func:`GateSWAP`

    **Matrix representation:**

    .. math::
        \\operatorname{ISWAP} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & i & 0 \\\\
            0 & i & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateISWAP()
        ISWAP
        >>> GateISWAP().matrix()
        [1, 0, 0, 0]
        [0, 0, I, 0]
        [0, I, 0, 0]
        [0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateISWAP(), 0, 1)
        >>> GateISWAP().power(2), GateISWAP().inverse()
        (ISWAP^(2), ISWAP†)
        (ISWAP^(2), ISWAP†)
        >>> GateISWAP().matrix()
        [1, 0, 0, 0]
        [0, 0, I, 0]
        [0, I, 0, 0]
        [0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateISWAP(), 0, 1)
        >>> GateISWAP().power(2), GateISWAP().inverse()
        (ISWAP^(2), ISWAP†)
        >>> GateISWAP().decompose()
    """
    _name = 'ISWAP'

    _num_qubits = 2
    _qragsizes = [2]

    def matrix(self):
        return Matrix([
            [1, 0, 0, 0], [0, 0, I, 0], [0, I, 0, 0], [0, 0, 0, 1]
        ])

    def _decompose(self, circ, qubits, bits):
        c, t = qubits
        circ.push(GateS(), c)
        circ.push(GateS(), t)
        circ.push(GateH(), c)
        circ.push(GateCX(), c, t)
        circ.push(GateCX(), t, c)
        circ.push(GateH(), t)
        return circ
