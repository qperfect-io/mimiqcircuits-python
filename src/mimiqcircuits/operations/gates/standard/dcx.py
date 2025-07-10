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
from symengine import Matrix
import mimiqcircuits as mc


class GateDCX(mcg.Gate):
    r"""Two qubit double-CNOT gate.

    A two qubit Clifford gate consisting of two back-to-back CNOTs with
    alternate controls.

    **Matrix representation:**

    .. math::
        \operatorname{DCX} =\begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 1 & 0 & 0
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateDCX()
        DCX
        >>> GateDCX().matrix()
        [1.0, 0, 0, 0]
        [0, 0, 1.0, 0]
        [0, 0, 0, 1.0]
        [0, 1.0, 0, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateDCX(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── DCX @ q[0,1]
        <BLANKLINE>
        >>> GateDCX().power(2), GateDCX().inverse()
        (DCX†, DCX†)
        >>> GateDCX().decompose()
        2-qubit circuit with 2 instructions:
        ├── CX @ q[0], q[1]
        └── CX @ q[1], q[0]
        <BLANKLINE>
    """

    _num_qubits = 2
    _qregsizes = [2]
    _name = "DCX"

    def _matrix(self):
        return Matrix(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

    def _power(self, p):
        pmod = p % 3

        if pmod == 1:
            return self

        if pmod == 2:
            return mc.Inverse(self)

        if pmod == 0:
            return mc.GateID().parallel(2)

    def _decompose(self, circ, qubits, bits, zvars):
        a, b = qubits
        circ.push(GateCX(), a, b)
        circ.push(GateCX(), b, a)
        return circ
