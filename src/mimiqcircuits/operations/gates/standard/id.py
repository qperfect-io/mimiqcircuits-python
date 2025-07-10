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
from mimiqcircuits.matrices import umatrixpi


class GateID(mc.Gate):
    r"""Single qubit Identity gate.

    **Matrix representation:**

    .. math::
        \operatorname{ID} = \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateID()
        ID
        >>> GateID().matrix()
        [1.0, 0]
        [0, 1.0]
        <BLANKLINE>
        >>> c = Circuit().push(GateID(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── ID @ q[0]
        <BLANKLINE>
        >>> GateID().power(2), GateID().inverse()
        (ID, ID)
        >>> GateID().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, 0, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "ID"

    _num_qubits = 1
    _qregsizes = [1]

    def inverse(self):
        return self

    def _power(self, n):
        return self

    def isidentity(self):
        return True

    def _matrix(self):
        return umatrixpi(0, 0, 0)

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        circ.push(mc.GateU(0, 0, 0), q)
        return circ
