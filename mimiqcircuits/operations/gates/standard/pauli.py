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
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.generalized.gphase import GPhase
from mimiqcircuits.matrices import umatrixpi, gphasepi
from symengine import pi
from sympy import simplify


class GateX(mcg.Gate):
    """ Single qubit Pauli-X gate.

    **Matrix representation:**

    .. math::
        \\operatorname{X} = \\begin{pmatrix}
            0 & 1 \\\\
            1 & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateX()
        X
        >>> GateX().matrix()
        Matrix([
        [0, 1],
        [1, 0]])
        >>> c = Circuit().push(GateX(), 0)
        >>> GateX().power(2), GateX().inverse()
        (ID, X)
        >>> GateX().decompose()
        1-qubit circuit with 2 instructions:
        ├── U(pi, 0, pi) @ q0
        └── GPhase(lmbda=(-1/2)*pi) @ q0
        1-qubit circuit with 2 instructions:
        ├── U(pi, 0, pi) @ q0
        └── GPhase(lmbda=(-1/2)*pi) @ q0
        >>> GateX().matrix()
        Matrix([
        [0, 1],
        [1, 0]])
        >>> c = Circuit().push(GateX(), 0)
        >>> GateX().power(2), GateX().inverse()
        (ID, X)
        >>> GateX().decompose()
        1-qubit circuit with 2 instructions:
        ├── U(pi, 0, pi) @ q0
        └── GPhase(lmbda=(-1/2)*pi) @ q0
    """
    _name = 'X'

    _num_qubits = 1
    _qragsizes = [1]

    def inverse(self):
        return self

    def power(self, p):
        return power_nhilpotent(self, p)

    def matrix(self):
        return simplify(umatrixpi(1, 0, 1) * gphasepi(-1/2))

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(pi, 0, pi), q)
        circ.push(GPhase(1, -pi/2), q)
        return circ


class GateY(mcg.Gate):
    """Single qubit Pauli-Y gate.

    **Matrix representation:**

    .. math::
        \\operatorname{Y} = \\begin{pmatrix}
            0 & -i \\\\
            i & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateY()
        Y
        >>> GateY().matrix()
        Matrix([
        [0, -I],
        [I,  0]])
        >>> c = Circuit().push(GateY(), 0)
        >>> GateY().power(2), GateY().inverse()
        (ID, Y)
        >>> GateY().decompose()
        1-qubit circuit with 2 instructions:
        ├── U(pi, (1/2)*pi, (1/2)*pi) @ q0
        └── GPhase(lmbda=(-1/2)*pi) @ q0
        1-qubit circuit with 2 instructions:
        ├── U(pi, (1/2)*pi, (1/2)*pi) @ q0
        └── GPhase(lmbda=(-1/2)*pi) @ q0
        >>> GateY().matrix()
        Matrix([
        [0, -I],
        [I,  0]])
        >>> c = Circuit().push(GateY(), 0)
        >>> GateY().power(2), GateY().inverse()
        (ID, Y)
        >>> GateY().decompose()
        1-qubit circuit with 2 instructions:
        ├── U(pi, (1/2)*pi, (1/2)*pi) @ q0
        └── GPhase(lmbda=(-1/2)*pi) @ q0
    """
    _name = 'Y'

    _num_qubits = 1
    _qragsizes = [1]

    def inverse(self):
        return self

    def power(self, p):
        return power_nhilpotent(self, p)

    def matrix(self):
        return simplify(umatrixpi(1, 1/2, 1/2) * gphasepi(-1/2))

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateU(pi, pi/2, pi/2), q)
        circ.push(GPhase(1, -pi/2), q)
        return circ


class GateZ(mcg.Gate):
    """Single qubit Pauli-Z gate.

    **Matrix representation:**

    .. math::
        \\operatorname{Z} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & -1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import *
        >>> GateZ()
        Z
        >>> GateZ().matrix()
        Matrix([
        [1,  0],
        [0, -1]])
        >>> c = Circuit().push(GateZ(), 0)
        >>> GateZ().power(2), GateZ().inverse()
        (ID, Z)
        >>> GateZ().decompose()
        1-qubit circuit with 1 instructions:
        └── P(pi) @ q0
        1-qubit circuit with 1 instructions:
        └── P(pi) @ q0
        >>> GateZ().matrix()
        Matrix([
        [1,  0],
        [0, -1]])
        >>> c = Circuit().push(GateZ(), 0)
        >>> GateZ().power(2), GateZ().inverse()
        (ID, Z)
        >>> GateZ().decompose()
        1-qubit circuit with 1 instructions:
        └── P(pi) @ q0
    """
    _name = 'Z'

    _num_qubits = 1
    _qragsizes = [1]

    def inverse(self):
        return self

    def power(self, p):
        return power_nhilpotent(self, p)

    def matrix(self):
        return simplify(umatrixpi(0, 0, 1))

    def _decompose(self, circ, qubits, bits):
        q = qubits[0]
        circ.push(GateP(pi), q)
        return circ
