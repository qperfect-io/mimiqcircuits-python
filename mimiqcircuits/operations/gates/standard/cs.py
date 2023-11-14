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

from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.cphase import GateCP
import mimiqcircuits.operations.control as mctrl
from symengine import pi


class GateCS(mctrl.Control):
    """Controlled-S gate.

    See Also :func:`GateS`

    **Matrix representation:**:

    .. math::
        \\operatorname{CS} =\\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & i
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCS(), GateCS().num_controls, GateCS().num_targets
        (CS, 1, 1)
        >>> GateCS().matrix()
        >>> c = Circuit().push(GateCS(), 0, 1)
        >>> GateCS().power(2), GateCS().inverse()
        (CZ^(1.0), CS†)
        >>> GateCS().decompose()
        2-qubit circuit with 1 instructions:
        └── CP((1/2)*pi) @ q0, q1
        2-qubit circuit with 1 instructions:
        └── CP((1/2)*pi) @ q0, q1
        >>> GateCS().matrix()
        >>> c = Circuit().push(GateCS(), 0, 1)
        >>> GateCS().power(2), GateCS().inverse()
        (CZ^(1.0), CS†)
        >>> GateCS().decompose()
        2-qubit circuit with 1 instructions:
        └── CP((1/2)*pi) @ q0, q1
    """

    def __init__(self):
        super().__init__(1, GateS())

    def _decompose(self, circ, qubits, bits):
        a, b = qubits

        circ.push(GateCP(pi/2), a, b)

        return circ


class GateCSDG(mctrl.Control):
    """Two qubit Controlled-S gate.

    **Matrix representation:**

    .. math::
        \\operatorname{CS}^{\\dagger} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & -i
            \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCSDG(), GateCSDG().num_controls, GateCSDG().num_targets
        (CInverse, 1, 1)
        >>> GateCSDG().matrix()
        >>> c = Circuit().push(GateCSDG(), 0, 1)
        >>> GateCSDG().power(2), GateCSDG().inverse()
        (CInverse^(2), CS)
        >>> GateCSDG().decompose()
        2-qubit circuit with 1 instructions:
        └── CP((-1/2)*pi) @ q0, q1
        2-qubit circuit with 1 instructions:
        └── CP((-1/2)*pi) @ q0, q1
        >>> GateCSDG().matrix()
        >>> c = Circuit().push(GateCSDG(), 0, 1)
        >>> GateCSDG().power(2), GateCSDG().inverse()
        (CInverse^(2), CS)
        >>> GateCSDG().decompose()
        2-qubit circuit with 1 instructions:
        └── CP((-1/2)*pi) @ q0, q1
    """

    def __init__(self):
        super().__init__(1, GateSDG())

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(GateCP(-pi/2), a, b)

        return circ
