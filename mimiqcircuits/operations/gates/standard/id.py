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
from mimiqcircuits.matrices import umatrixpi
import symengine as se
import sympy as sp


class GateID(mc.Gate):
    """Single qubit Identity gate.

    **Matrix representation:**

    .. math::
        \\operatorname{ID} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & 1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateID()
        ID
        >>> GateID().matrix()
        [1, 0]
        [0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateID(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── ID @ q0
        >>> GateID().power(2), GateID().inverse()
        (ID^(2), ID)
        >>> GateID().decompose()
        1-qubit circuit with 1 instructions:
        └── U(0, 0, 0) @ q0
    """
    _name = 'ID'

    _num_qubits = 1
    _qragsizes = [1]

    def inverse(self):
        return self

    def matrix(self):
        return se.Matrix(sp.simplify(umatrixpi(0, 0, 0)))

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(mc.GateU(0, 0, 0), q)
        return circ


class GateID2(mc.Control):

    """Two-qubit identity gate

    See also :func:`GateID` and :func:`Parallel`

    The gate is implemented as a parallel application of a GateID.

    **Matrix representation:**

    .. math::
        \\operatorname{ID2} = \\begin{pmatrix}
            1 & 0 & 0 & 0\\\\
            0 & 1 & 0 & 0\\\\
            0 & 0 & 1 & 0\\\\
            0 & 0 & 0 & 1\\\\
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateID2()
        CID
        >>> GateID2().matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, 1, 0]
        [0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateID2(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── CID @ q0, q1
        >>> GateID2().power(2), GateID2().inverse()
        (C(ID^(2)), CID)
        >>> GateID2().decompose()
        2-qubit circuit with 1 instructions:
        └── CID @ q0, q1
    """
    _name = 'ID2'

    def __init__(self):
        super().__init__(1, GateID())
