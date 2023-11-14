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
from mimiqcircuits.operations.utils import power_nhilpotent
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.generalized.gphase import GPhase
from mimiqcircuits.matrices import gphasepi, umatrixpi
from symengine import pi


class GateH(mcg.Gate):
    """Single qubit Hadamard gate.

    **Matrix representation:**

    .. math::
        \\operatorname{H} = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            1 & 1 \\\\
            1 & -1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateH()
        H
        >>> GateH().matrix()
        [(1/2)*(-I*sin(0.25*pi) + cos(0.25*pi))*(1 + I*sin(0.5*pi) + cos(0.5*pi)), 1/2*I*(-I*sin(0.25*pi) + cos(0.25*pi))*(1 - (I*sin(0.5*pi) + cos(0.5*pi)))]
        [1/2*I*(-I*sin(0.25*pi) + cos(0.25*pi))*(1 - (I*sin(0.5*pi) + cos(0.5*pi))), (-1/2)*(-I*sin(0.25*pi) + cos(0.25*pi))*(1 + I*sin(0.5*pi) + cos(0.5*pi))]
        <BLANKLINE>
        >>> c = Circuit().push(GateH(), 0)
        >>> GateH().power(2), GateH().inverse()
        (ID, H)
        >>> GateH().decompose()
        1-qubit circuit with 2 instructions:
        ├── U((1/2)*pi, 0, pi) @ q0
        └── GPhase(lmbda=(-1/4)*pi) @ q0
        1-qubit circuit with 2 instructions:
        ├── U((1/2)*pi, 0, pi) @ q0
        └── GPhase(lmbda=(-1/4)*pi) @ q0
        >>> GateH().matrix()
        [(1/2)*(-I*sin(0.25*pi) + cos(0.25*pi))*(1 + I*sin(0.5*pi) + cos(0.5*pi)), 1/2*I*(-I*sin(0.25*pi) + cos(0.25*pi))*(1 - (I*sin(0.5*pi) + cos(0.5*pi)))]
        [1/2*I*(-I*sin(0.25*pi) + cos(0.25*pi))*(1 - (I*sin(0.5*pi) + cos(0.5*pi))), (-1/2)*(-I*sin(0.25*pi) + cos(0.25*pi))*(1 + I*sin(0.5*pi) + cos(0.5*pi))]
        <BLANKLINE>
        >>> c = Circuit().push(GateH(), 0)
        >>> GateH().power(2), GateH().inverse()
        (ID, H)
        >>> GateH().decompose()
        1-qubit circuit with 2 instructions:
        ├── U((1/2)*pi, 0, pi) @ q0
        └── GPhase(lmbda=(-1/4)*pi) @ q0
    """
    _name = 'H'

    _num_qubits = 1
    _qragsizes = [1]

    def inverse(self):
        return self

    def power(self, p):
        return power_nhilpotent(self, p)

    def matrix(self):
        return gphasepi(-1/4) * umatrixpi(1/2, 0, 1)

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(pi/2, 0, pi), q)
        circ.push(GPhase(1, -pi/4), q)
        return circ
