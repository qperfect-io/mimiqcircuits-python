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
from mimiqcircuits.operations.gates.standard.interactions import GateRZX
from mimiqcircuits.operations.utils import power_idempotent
from mimiqcircuits.operations.gates.standard.pauli import GateX
from symengine import sqrt, I, pi, Matrix


class GateECR(mcg.Gate):
    r"""Two qubit ECR (echo) gate.

    **Matrix representation:**

    .. math::
        \operatorname{ECR} =\begin{pmatrix}
            0 & \frac{1}{\sqrt{2}} & 0 & \frac{i}{\sqrt{2}} \\
            \frac{1}{\sqrt{2}} & 0 & \frac{-i}{\sqrt{2}} & 0 \\
            0 & \frac{i}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
            \frac{-i}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} & 0
        \end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateECR()
        ECR
        >>> GateECR().matrix()
        [0, 0, 0.707106781186548, 0.0 + 0.707106781186548*I]
        [0, 0, 0.0 + 0.707106781186548*I, 0.707106781186548]
        [0.707106781186548, -0.0 - 0.707106781186548*I, 0, 0]
        [-0.0 - 0.707106781186548*I, 0.707106781186548, 0, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateECR(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── ECR @ q[0,1]
        <BLANKLINE>
        >>> GateECR().power(2), GateECR().inverse()
        (⨷ ² ID, ECR)
        >>> GateECR().decompose()
        2-qubit circuit with 3 instructions:
        ├── RZX((1/4)*pi) @ q[0,1]
        ├── X @ q[0]
        └── RZX((-1/4)*pi) @ q[0,1]
        <BLANKLINE>
    """

    _name = "ECR"

    _num_qubits = 2
    _qregsizes = [2]

    def _matrix(self):
        return Matrix(
            [[0, 0, 1, I], [0, 0, I, 1], [1, -I, 0, 0], [-I, 1, 0, 0]]
        ) / sqrt(2)

    def inverse(self):
        return self

    def _power(self, p):
        return power_idempotent(self, p)

    def _decompose(self, circ, qubits, bits, zvars):
        a, b = qubits

        circ.push(GateRZX(pi / 4), a, b)
        circ.push(GateX(), a)
        circ.push(GateRZX(-pi / 4), a, b)
        return circ
