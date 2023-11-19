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
    
    By convention, the first qubit is the control and the second is
    the target

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
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateCP(lmbda), GateCP(lmbda).num_controls, GateCP(lmbda).num_targets, GateCP(lmbda).num_qubits
        (CP(lambda), 1, 1, 2)
        >>> GateCP(lmbda).matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, 1, 0]
        [0, 0, 0, exp(I*lambda)]
        <BLANKLINE>
        >>> c = Circuit().push(GateCP(lmbda), 10, 11)
        >>> c
        12-qubit circuit with 1 instructions:
        └── CP(lambda) @ q10, q11
        >>> GateCP(lmbda).decompose()
        2-qubit circuit with 5 instructions:
        ├── P((1/2)*lambda) @ q0
        ├── CX @ q0, q1
        ├── P((-1/2)*lambda) @ q1
        ├── CX @ q0, q1
        └── P((1/2)*lambda) @ q1
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
