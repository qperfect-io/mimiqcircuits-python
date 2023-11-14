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
from mimiqcircuits.operations.gates.standard.swap import GateSWAP
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.cnx import GateCCX


class GateCSWAP(mctrl.Control):
    """Three qubit Controlled-SWAP gate.

    By convention, the first qubit is the control and last two are the
    targets.

    Examples:
        >>> from mimiqcircuits import *
        >>> GateCSWAP(), GateCSWAP().num_controls, GateCSWAP().num_targets
        (CSWAP, 1, 2)
        >>> GateCSWAP().matrix()
        [1, 0, 0, 0, 0, 0, 0, 0]
        [0, 1, 0, 0, 0, 0, 0, 0]
        [0, 0, 1, 0, 0, 0, 0, 0]
        [0, 0, 0, 1, 0, 0, 0, 0]
        [0, 0, 0, 0, 1, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 1, 0]
        [0, 0, 0, 0, 0, 1, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateCSWAP(), 0, 1, 2)
        >>> GateCSWAP().power(2), GateCSWAP().inverse()
        (CSWAP^(2), CSWAP)
        (CSWAP^(2), CSWAP)
        >>> GateCSWAP().matrix()
        [1, 0, 0, 0, 0, 0, 0, 0]
        [0, 1, 0, 0, 0, 0, 0, 0]
        [0, 0, 1, 0, 0, 0, 0, 0]
        [0, 0, 0, 1, 0, 0, 0, 0]
        [0, 0, 0, 0, 1, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 1, 0]
        [0, 0, 0, 0, 0, 1, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 1]
        <BLANKLINE>
        >>> c = Circuit().push(GateCSWAP(), 0, 1, 2)
        >>> GateCSWAP().power(2), GateCSWAP().inverse()
        (CSWAP^(2), CSWAP)
        >>> GateCSWAP().decompose()
    """

    def __init__(self):
        super().__init__(1, GateSWAP())

    def _decompose(self, circ, qubits, bits):
        c, t1, t2 = qubits
        circ.push(GateCX(), t2, t1)
        circ.push(GateCCX(), c, t1, t2)
        circ.push(GateCX(), t2, t1)
        return circ
