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
from mimiqcircuits.operations.gates.standard.pauli import GateX
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from symengine import Matrix
import sympy as sp


class GateDCX(mcg.Gate):
    """Two qubit double-CNOT gate.

    A two qubit Clifford gate consisting of two back-to-back CNOTs with
    alternate controls.

    **Matrix representation:**

    .. math::
        \\operatorname{DCX} =\\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateDCX()
        DCX
        >>> GateDCX().matrix()
        [1.0, 0, 0, 0]
        [0, 0, 0, 1.0]
        [0, 1.0, 0, 0]
        [0, 0, 1.0, 0]
        <BLANKLINE>
        >>> c = Circuit().push(GateDCX(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── DCX @ q0, q1
        >>> GateDCX().power(2), GateDCX().inverse()
        (DCX^(2), DCX†)
        >>> GateDCX().decompose()
        2-qubit circuit with 2 instructions:
        ├── CX @ q0, q1
        └── CX @ q1, q0
    """
    _num_qubits = 2
    _qregsizes = [2]
    _name = 'DCX'

    def matrix(self):
        return Matrix(sp.simplify(Matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])))

    def _decompose(self, circ, qubits, bits):
        a, b = qubits
        circ.push(GateCX(), a, b)
        circ.push(GateCX(), b, a)
        return circ
