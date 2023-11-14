
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

import mimiqcircuits as mc


class GateCCP(mc.Control):
    """Two qubit Controlled-Phase gate.

    By convention, the first two qubits are the controls and the third is the
    the target

    Arguments:
        lmbda: Phase angle.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateCCP(lmbda), GateCCP(lmbda).num_controls, GateCCP(lmbda).num_targets
        (C₂P((lambda,)), 2, 1)
        (C₂P((lambda,)), 2, 1)
        >>> GateCCP(lmbda).matrix()
        >>> c = Circuit().push(GateCCP(lmbda), 0, 1, 2)
        >>> GateCCP(lmbda).power(2), GateCCP(lmbda).inverse()
        >>> GateCCP(lmbda).decompose()
    """

    def __init__(self, *args):
        super().__init__(2, mc.GateP(args))

    def _decompose(self, circ, qubits, bits):
        c1, c2, t = qubits
        lmbda = self.op.lmbda
        circ.push(mc.GateCP(lmbda / 2), c2, t)
        circ.push(mc.GateCX(), c1, c2)
        circ.push(mc.GateCP(-lmbda / 2), c2, t)
        circ.push(mc.GateCX(), c1, c2)
        circ.push(mc.GateCP(lmbda / 2), c1, t)
        return circ
