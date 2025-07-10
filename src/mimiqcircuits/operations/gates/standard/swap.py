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
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.utils import power_idempotent, control_one_defined
import mimiqcircuits as mc
from symengine import Matrix


class GateSWAP(mcg.Gate):
    r"""Two qubit SWAP gate.

    See Also :func:`GateISWAP`

    **Matrix representation:**

    .. math::
        \operatorname{SWAP} = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateSWAP()
        SWAP
        >>> GateSWAP().matrix()
        [1.0, 0, 0, 0]
        [0, 0, 1.0, 0]
        [0, 1.0, 0, 0]
        [0, 0, 0, 1.0]
        <BLANKLINE>
        >>> c = Circuit().push(GateSWAP(), 0, 1)
        >>> GateSWAP().power(2), GateSWAP().inverse()
        (⨷ ² ID, SWAP)
        >>> GateSWAP().decompose()
        2-qubit circuit with 3 instructions:
        ├── CX @ q[0], q[1]
        ├── CX @ q[1], q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
    """

    _name = "SWAP"

    _num_qubits = 2
    _qregsizes = [2]

    def _matrix(self):
        return Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def inverse(self):
        return self

    def _power(self, p):
        # SWAP^(2n) = ID
        # SWAP^(2n + 1) = SWAP
        return power_idempotent(self, p)

    def _control(self, n):
        return control_one_defined(n, self, mc.GateCSWAP())

    def _decompose(self, circ, qubits, bits, zvars):
        c, t = qubits
        circ.push(GateCX(), c, t)
        circ.push(GateCX(), t, c)
        circ.push(GateCX(), c, t)
        return circ
