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
from mimiqcircuits.operations.gates.standard.interactions import GateRZX
from mimiqcircuits.operations.gates.standard.pauli import GateX
from symengine import sqrt, I, pi, Matrix
import sympy as sp


class GateECR(mcg.Gate):
    """Two qubit ECR (echo) gate.

    **Matrix representation:**

    .. math::
        \\operatorname{ECR} =\\begin{pmatrix}
            0 & \\frac{1}{\\sqrt{2}} & 0 & \\frac{i}{\\sqrt{2}} \\ \\\\
            \\frac{1}{\\sqrt{2}} & 0 & \\frac{-i}{\\sqrt{2}} & 0 \\\\
            0 & \\frac{i}{\\sqrt{2}} & 0 & \\frac{i}{\\sqrt{2}} \\\\
            \\frac{-i}{\\sqrt{2}} & 0 & \\frac{1}{\\sqrt{2}} & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateECR()
        ECR
        >>> GateECR().matrix()
        [0, (1/2)*sqrt(2), 0, (0.0 + 0.5*I)*sqrt(2)]
        [(1/2)*sqrt(2), 0, -1/2*I*sqrt(2), 0]
        [0, 1/2*I*sqrt(2), 0, (1/2)*sqrt(2)]
        [-1/2*I*sqrt(2), 0, (1/2)*sqrt(2), 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateECR(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── ECR @ q0, q1
        >>> GateECR().power(2), GateECR().inverse()
        (ECR^(2), ECR)
        >>> GateECR().decompose()
        2-qubit circuit with 3 instructions:
        ├── RZX((1/4)*pi) @ q0, q1
        ├── X @ q0
        └── RZX((-1/4)*pi) @ q0, q1
    """
    _name = 'ECR'

    _num_qubits = 2
    _qregsizes = [2]

    def matrix(self):
        return Matrix(sp.simplify(1/sqrt(2) * Matrix([[0, 1, 0, 1j],
                                   [1, 0, -I, 0],
                                   [0, I, 0, 1],
                                   [-I, 0, 1, 0]])))

    def inverse(self):
        return self

    def _decompose(self, circ, qubits, bits):
        a, b = qubits

        circ.push(GateRZX(pi/4), a, b)
        circ.push(GateX(), a)
        circ.push(GateRZX(-pi/4), a, b)
        return circ
