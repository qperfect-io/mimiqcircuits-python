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
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.gates.standard.rotations import GateRX, GateRY, GateRZ
from symengine import pi


class GateCRX(mctrl.Control):
    """Two qubit Controlled-RX gate.

    See Also :func:`GateRX`, :func:`GateCRY`, :func:`GateCRZ`

    **Matrix representation:**

    .. math::
        \\operatorname{CRX}(\\theta) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
            0 & 0 & -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta: The rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')
        >>> GateCRZ(theta), GateCRZ(theta).num_controls, GateCRZ(theta).num_targets
        (CRZ(theta), 1, 1)
        >>> GateCRZ(theta).matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, -I*sin((1/2)*theta) + cos((1/2)*theta), 0]
        [0, 0, 0, (-I*sin((1/2)*theta) + cos((1/2)*theta))*(I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRZ(theta), 0, 1)
        >>> GateCRZ(theta).power(2), GateCRZ(theta).inverse()
        (CRZ(2*theta), CRZ(-theta))
        (CRZ(2*theta), CRZ(-theta))
        >>> GateCRZ(theta).matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, -I*sin((1/2)*theta) + cos((1/2)*theta), 0]
        [0, 0, 0, (-I*sin((1/2)*theta) + cos((1/2)*theta))*(I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRZ(theta), 0, 1)
        >>> GateCRZ(theta).power(2), GateCRZ(theta).inverse()
        (CRZ(2*theta), CRZ(-theta))
        >>> GateCRZ(theta).decompose()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(1, GateRX(*args, **kwargs))

    def _decompose(self, circ, qubits, bits):
        c, t = qubits
        theta = self.operation.theta
        circ.push(GateP(pi/2), t)
        circ.push(GateCX(), c, t)
        circ.push(GateU(-theta/2, 0, 0), t)
        circ.push(GateCX(), c, t)
        circ.push(GateU(theta/2, -pi/2, 0), t)
        return circ


class GateCRY(mctrl.Control):
    """Two qubit Controlled-RY gate.

    See Also :func:`GateRY`, :func:`GateCRX`, :func:`GateCRZ`

    **Matrix representation:**

    .. math::
        \\operatorname{CRY}(\\theta) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
            0 & 0 &  \\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}


    Parameters:
        theta: The rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> theta = Symbol('theta')
        >>> GateCRY(theta), GateCRY(theta).num_controls, GateCRY(theta).num_targets
        (CRY(theta), 1, 1)
        >>> GateCRY(theta).matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta)), -1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - I*sin(theta) - cos(theta))]
        [0, 0, 1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - I*sin(theta) - cos(theta)), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRY(theta), 0, 1)
        >>> GateCRY(theta).power(2), GateCRY(theta).inverse()
        (CRY(2*theta), CRY(-theta))
        (CRY(2*theta), CRY(-theta))
        >>> GateCRY(theta).matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta)), -1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - I*sin(theta) - cos(theta))]
        [0, 0, 1/2*I*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 - I*sin(theta) - cos(theta)), (1/2)*(-I*sin((1/2)*theta) + cos((1/2)*theta))*(1 + I*sin(theta) + cos(theta))]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRY(theta), 0, 1)
        >>> GateCRY(theta).power(2), GateCRY(theta).inverse()
        (CRY(2*theta), CRY(-theta))
        >>> GateCRY(theta).decompose()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(1, GateRY(*args, **kwargs))

    def _decompose(self, circ, qubits, bits):
        c, t = qubits
        theta = self.operation.theta
        circ.push(GateRY(theta/2), t)
        circ.push(GateCX(), c, t)
        circ.push(GateRY(-theta/2), t)
        circ.push(GateCX(), c, t)
        return circ


class GateCRZ(mctrl.Control):
    """Two qubit Controlled-RZ gate.

    See Also :func:`GateRZ`, :func:`GateCRX`, :func:`GateCRY`

    **Matrix representation:**

    .. math::
        \\operatorname{CRZ}(\\theta) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & e^{-i\\frac{\\lambda}{2}} & 0 \\\\
            0 & 0 & 0 & e^{i\\frac{\\lambda}{2}}
        \\end{pmatrix}

    Parameters:
        theta: The rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> lmbda = Symbol('lambda')
        >>> GateCRZ(lmbda), GateCRZ(lmbda).num_controls, GateCRZ(lmbda).num_targets
        (CRZ(lambda), 1, 1)
        >>> GateCRZ(lmbda).matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, -I*sin((1/2)*lambda) + cos((1/2)*lambda), 0]
        [0, 0, 0, (-I*sin((1/2)*lambda) + cos((1/2)*lambda))*(I*sin(lambda) + cos(lambda))]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRZ(lmbda), 0, 1)
        >>> GateCRZ(lmbda).power(2), GateCRZ(lmbda).inverse()
        (CRZ(2*lambda), CRZ(-lambda))
        (CRZ(2*lambda), CRZ(-lambda))
        >>> GateCRZ(lmbda).matrix()
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, -I*sin((1/2)*lambda) + cos((1/2)*lambda), 0]
        [0, 0, 0, (-I*sin((1/2)*lambda) + cos((1/2)*lambda))*(I*sin(lambda) + cos(lambda))]
        <BLANKLINE>
        >>> c = Circuit().push(GateCRZ(lmbda), 0, 1)
        >>> GateCRZ(lmbda).power(2), GateCRZ(lmbda).inverse()
        (CRZ(2*lambda), CRZ(-lambda))
        >>> GateCRZ(lmbda).decompose()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(1, GateRZ(*args, **kwargs))

    def _decompose(self, circ, qubits, bits):
        c, t = qubits
        lmbda = self.operation.lmbda
        circ.push(GateRZ(lmbda/2), t)
        circ.push(GateCX(), c, t)
        circ.push(GateRZ(-lmbda/2), t)
        circ.push(GateCX(), c, t)
        return circ
