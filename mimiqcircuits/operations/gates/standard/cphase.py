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
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.phase import GateP


class GateCP(mctrl.Control):
    """Two qubit Controlled-Phase gate.

    See Also :func:`GateP`

    **Matrix representation:**

    .. math::
        \\operatorname{CP}(\\lambda) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & e^{i\\lambda}
        \\end{pmatrix}

    Parameters:
        lambda: Phase angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi
        >>> lmbda = Symbol('lambda')
        >>> GateCP(lmbda), GateCP(lmbda).num_controls, GateCP(lmbda).num_targets
        >>> GateCP(lmbda).matrix()
        >>> c = Circuit().push(GateCP(lmbda), 0, 1)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(1, GateP(*args, **kwargs))

    def _decompose(self, circ, qubits, bits):
        c, t = qubits
        lmbda = self.op.lmbda
        circ.push(GateP(lmbda/2), c)
        circ.push(GateCX(), c, t)
        circ.push(GateP(-lmbda/2), t)
        circ.push(GateCX(), c, t)
        circ.push(GateP(lmbda/2), t)
        return circ
