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

from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.deprecated import GateU1
import mimiqcircuits.operations.control as mctrl
from symengine import pi


class GateCSX(mctrl.Control):
    """Two qubit Controled-SX (control on second qubit) gate.

    By convention, the first qubit is the control and second one is the
    targets.


    **Matrix representation:**

    .. math::

        \\operatorname{CSX} =\\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & \\frac{1+i}{\\sqrt{2}} & \\frac{1-i}{\\sqrt{2}} \\\\
            0 & 0 & \\frac{1-i}{\\sqrt{2}} & \\frac{1+i}{\\sqrt{2}}
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCSX(), GateCSX().num_controls, GateCSX().num_targets, GateCSX().num_qubits
        (C(X^(1/2)), 1, 1, 2)
        >>> GateCSX().matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, 0.5 + 0.5*I, 0.5 - 0.5*I]
        [0, 0, 0.5 - 0.5*I, 0.5 + 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateCSX(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── C(X^(1/2)) @ q0, q1
        >>> GateCSX().power(2), GateCSX().inverse()
        (C(X^(1.0)), C((X^(1/2))†))
        >>> GateCSX().decompose()
        2-qubit circuit with 3 instructions:
        ├── H @ q1
        ├── CU1((1/2)*pi) @ q0, q1
        └── H @ q1
    """

    def __init__(self):
        super().__init__(1, GateSX())

    def _decompose(self, circ, qubits, bits):
        a, b = qubits

        circ.push(GateH(), b)
        circ.push(GateU1(pi/2).control(1), a, b)
        circ.push(GateH(), b)

        return circ


class GateCSXDG(mctrl.Control):
    """Two qubit :math:`{CSX}^\\dagger` gate.

    **Matrix representation:**

    .. math::
        \\operatorname{CSX}^{\\dagger} =\\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & \\frac{1-i}{\\sqrt{2}} & \\frac{1+i}{\\sqrt{2}} \\\\
            0 & 0 & \\frac{1+i}{\\sqrt{2}} & \\frac{1-i}{\\sqrt{2}}
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCSXDG(), GateCSXDG().num_controls, GateCSXDG().num_targets, GateCSXDG().num_qubits
        (C((X^(1/2))†), 1, 1, 2)
        >>> GateCSXDG().matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, 0.5 - 0.5*I, 0.5 + 0.5*I]
        [0, 0, 0.5 + 0.5*I, 0.5 - 0.5*I]
        <BLANKLINE>
        >>> c = Circuit().push(GateCSXDG(), 0, 1)
        >>> c
        2-qubit circuit with 1 instructions:
        └── C((X^(1/2))†) @ q0, q1
        >>> GateCSXDG().power(2), GateCSXDG().inverse()
        (C(((X^(1/2))†)^(2)), C(X^(1/2)))
        >>> GateCSXDG().decompose()
        2-qubit circuit with 3 instructions:
        ├── H @ q1
        ├── CU1((-1/2)*pi) @ q0, q1
        └── H @ q1
    """

    def __init__(self):
        super().__init__(1, GateSXDG())

    def _decompose(self, circ, qubits, bits):
        a, b = range(self.num_qubits)

        circ.push(GateH(), b)
        circ.push(GateU1(-pi/2).control(1), a, b)
        circ.push(GateH(), b)

        return circ
