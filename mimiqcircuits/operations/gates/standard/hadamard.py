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
from mimiqcircuits.operations.utils import power_idempotent
from mimiqcircuits.matrices import gphasepi, umatrixpi
from mimiqcircuits.operations.utils import control_one_defined
from symengine import pi, sqrt, Matrix


class GateH(mc.Gate):
    r"""Single qubit Hadamard gate.

    **Matrix representation:**

    .. math::
        \operatorname{H} = \frac{1}{\sqrt{2}} \begin{pmatrix}
            1 & 1 \\
            1 & -1
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateH()
        H
        >>> GateH().matrix()
        [0.707106781186548, 0.707106781186548]
        [0.707106781186548, -0.707106781186548]
        <BLANKLINE>
        >>> c = Circuit().push(GateH(), 0)
        >>> c
        1-qubit circuit with 1 instructions:
        └── H @ q[0]
        <BLANKLINE>
        >>> GateH().power(2), GateH().inverse()
        (ID, H)
        >>> GateH().decompose()
        1-qubit circuit with 1 instructions:
        └── U((1/2)*pi, 0, pi, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "H"

    _num_qubits = 1
    _qregsizes = [1]

    def inverse(self):
        return self

    def _power(self, p):
        # H^(2n) = ID
        # H^(2n+1) = H
        return power_idempotent(self, p)

    def _control(self, n):
        return control_one_defined(self, n)

    def _matrix(self):
        return Matrix([[1, 1], [1, -1]]) / sqrt(2)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(mc.GateU(pi/2, 0, pi), q)
        return circ
