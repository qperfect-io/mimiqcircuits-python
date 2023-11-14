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

import mimiqcircuits.operations.control as mctrl
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.hadamard import GateH


class GateCX(mctrl.Control):
    """Two qubit Controlled-X gate (or CNOT).

    **Matrix representation:**

    .. math::
        \\operatorname{CX} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            0 & 0 & 1 & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCX(), GateCX().num_controls, GateCX().num_targets
        >>> GateCX().matrix()
        >>> c = Circuit().push(GateCX(), 0, 1)
        >>> GateCX().power(2), GateCX().inverse()
        >>> GateCX().decompose()
    """

    def __init__(self):
        super().__init__(1, GateX())

    def _decompose(self, circ, qubits, bits):
        circ.push(self, *qubits)
        return circ


class GateCY(mctrl.Control):
    """Two qubit Controlled-Y gate.

    **Matrix representation:**

    .. math::
        \\operatorname{CY} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & -i \\\\
            0 & 0 & i & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCY(), GateCY().num_controls, GateCY().num_targets
        >>> GateCY().matrix()
        >>> c = Circuit().push(GateCY(), 0, 1)
        >>> GateCY().power(2), GateCY().inverse()
        >>> GateCY().decompose()
    """

    def __init__(self):
        super().__init__(1, GateY())

    def _decompose(self, circ, qubits, bits):
        c, t = qubits
        circ.push(GateSDG(), t)
        circ.push(GateCX(), c, t)
        circ.push(GateS(), c)
        return circ


class GateCZ(mctrl.Control):
    """Two qubit Controlled-Z gate.

    **Matrix representation:**

    .. math::
        \\operatorname{CZ} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & -1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCZ(), GateCZ().num_controls, GateCZ().num_targets
        >>> GateCZ().matrix()
        >>> c = Circuit().push(GateCZ(), 0, 1)
        >>> GateCZ().power(2), GateCZ().inverse()
        >>> GateCZ().decompose()
    """

    def __init__(self):
        super().__init__(1, GateZ())

    def _decompose(self, circ, qubits, bits):
        c, t = qubits
        circ.push(GateH(), t)
        circ.push(GateCX(), c, t)
        circ.push(GateH(), t)
        return circ
