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

import numpy as np
import inspect
from abc import abstractmethod
from mimiqcircuits.operation import Operation
from mimiqcircuits.unicode import UNICODE_TO_NAMES, NAMES_TO_UNICODE


def _decomplex(m):
    if np.iscomplexobj(m):
        if np.all(np.isreal(m)):
            return np.real(m)
        else:

            return m
    else:
        return m


def cis(x):
    return np.exp(1j * x)


def cispi(x):
    return np.exp(1j * np.pi * x)


def pmatrixpi(lmbda):
    return np.array([[1, 0], [0, cispi(lmbda)]])


def pmatrix(lmbda):
    return np.array([[1, 0], [0, cis(lmbda)]])


def ctrl_fs(mat):
    """
    Returns the controlled version of a given 2x2 matrix, with control on the first then the second qubit.

    Args:
        mat (numpy.ndarray): A 2x2 matrix to be controlled.

    Returns:
        numpy.ndarray: A 4x4 controlled matrix.
    """
    return np.dot(ctrl(mat), ctrl2(mat))


def ctrl_sf(mat):
    """
    Returns the controlled version of a given 2x2 matrix, with control on the second then the first qubit.

    Args:
        mat (numpy.ndarray): A 2x2 matrix to be controlled.

    Returns:
        numpy.ndarray: A 4x4 controlled matrix.
    """
    return np.dot(ctrl2(mat), ctrl(mat))


def ctrl(mat):
    """
    Returns the controlled version of a given 2x2 matrix.

    Args:
        mat (numpy.ndarray): A 2x2 matrix to be controlled.

    Returns:
        numpy.ndarray: A 4x4 controlled matrix.
    """
    dim = mat.shape[0]
    return np.block([
        [np.identity(dim), np.zeros(mat.shape)],
        [np.zeros(mat.shape), mat]
    ])


def ctrl2(mat):
    """
    Returns the controlled version of a given 2x2 matrix, with control on the second qubit.

    Args:
        mat (numpy.ndarray): A 2x2 matrix to be controlled.

    Returns:
        numpy.ndarray: A 4x4 controlled matrix.
    """
    return np.block([
        [1, 0, 0, 0],
        [0, mat[0, 0], 0, mat[0, 1]],
        [0, 0, 1, 0],
        [0, mat[1, 0], 0, mat[1, 1]]
    ])


def gphase(lmbda):
    return cis(lmbda)


def gphasepi(lmbda):
    return cispi(lmbda)


def umatrix(theta, phi, lmbda, gamma=0.0):
    costheta2 = np.cos(theta / 2)
    sintheta2 = np.sin(theta / 2)
    return np.array([
        [cis(gamma) * costheta2, -cis(lmbda + gamma) * sintheta2],
        [cis(phi + gamma) * sintheta2, cis(phi + lmbda + gamma) * costheta2]
    ])


def umatrixpi(theta, phi, lmbda, gamma=0.0):
    costheta2 = np.cos((theta/2) * np.pi)
    sintheta2 = np.sin((theta/2) * np.pi)
    return np.array([
        [cispi(gamma) * costheta2, -cispi(lmbda + gamma) * sintheta2],
        [
            cispi(phi + gamma) * sintheta2,
            cispi(phi + lmbda + gamma) * costheta2
        ]
    ])


def rmatrix(theta, phi):
    costheta2 = np.cos(theta / 2)
    sintheta2 = np.sin(theta / 2)
    return np.array([
        [costheta2, -1j*cis(-phi) * sintheta2],
        [-1j*cis(phi)*sintheta2, costheta2]
    ])


def rmatrixpi(theta, phi):
    costheta2 = np.cos(theta/2 * np.pi)
    sintheta2 = np.sin(theta/2 * np.pi)
    return np.array([
        [costheta2, -1j*cispi(-phi) * sintheta2],
        [-1j*cispi(phi)*sintheta2, costheta2]
    ])


def rxmatrixpi(theta):
    return umatrixpi(theta, -1/2, 1/2)


def rxmatrix(theta):
    return rxmatrixpi(theta / np.pi)


def rymatrixpi(theta):
    return umatrixpi(theta, 0, 0)


def rymatrix(theta):
    return rymatrixpi(theta / np.pi)


def rzmatrixpi(lmbda):
    return gphasepi(-lmbda / 2) * umatrixpi(0, 0, lmbda)


def rzmatrix(lmbda):
    return rzmatrixpi(lmbda / np.pi)


class Gate(Operation):
    _num_qubits = None
    _num_bits = 0
    _name = None
    _parnames = ()

    @property
    def parnames(self):
        return self._parnames

    @parnames.setter
    def parnames(self, value):
        raise ValueError('Cannot set parnames. Read only parameter.')

    @abstractmethod
    def matrix(self):
        pass

    @abstractmethod
    def inverse(self):
        pass

    @staticmethod
    def from_json(d):
        name = d['name']
        num_qubits = d['N']
        num_bits = d['M']

        if d['name'] == 'Custom':
            matrix = np.array(d['U']).reshape(2**num_qubits, 2**num_qubits).transpose()
            return GateCustom(matrix=matrix, num_qubits=num_qubits)

        if 'params' in d:
            nd = { UNICODE_TO_NAMES.get(k, k): v for k, v in d['params'].items() }
            return GATES[name](**nd)

        return GATES[name]()

    def to_json(self):
        d = super().to_json()

        if len(self.parnames) != 0:
            d['params'] = { NAMES_TO_UNICODE.get(k, k): getattr(self, k) for k in self.parnames }

        if isinstance(self, GateCustom):
            d['matrix'] = self.matrix.transpose().flatten().tolist()

        return d

    def __eq__(self, other):
        return isinstance(other, type(self)) and\
            self.__dict__ == other.__dict__

    def __str__(self):
        pars = ''
        if len(self.parnames) != 0:
            pars += '('
            pars += ', '.join([f'{pn}={getattr(self, pn)}' for pn in self.parnames])
            pars += ')'
        return self.name + pars

    def __repr__(self):
        return str(self)


class GateX(Gate):
    """ Single qubit Pauli-X gate.

    **Matrix representation:**

    .. math::
        \\operatorname{X} = \\begin{pmatrix}
            0 & 1 \\\\
            1 & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateX
        >>> GateX().matrix()
            array([[0, 1],
                   [1, 0]])
        >>> c=Circuit()
        >>> c.push(GateX(),0)
            1-qubit circuit with 1 instructions:
             └── X @ q0
    """
    _num_qubits = 1
    _name = 'X'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[0, 1], [1, 0]])


class GateY(Gate):
    """Single qubit Pauli-Y gate.

    **Matrix representation:**

    .. math::
        \\operatorname{Y} = \\begin{pmatrix}
            0 & -i \\\\
            i & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateY
        >>> GateY().matrix()
            array([[ 0.+0.j, -0.-1.j],
                   [ 0.+1.j,  0.+0.j]])
        >>> c=Circuit()
        >>> c.push(GateY(),0)
            1-qubit circuit with 1 instructions:
             └── Y @ q0
    """
    _num_qubits = 1
    _name = 'Y'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[0, -1j], [1j, 0]])


class GateZ(Gate):
    """Single qubit Pauli-Z gate.

    **Matrix representation:**

    .. math::
        \\operatorname{Z} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & -1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateZ
        >>> GateZ().matrix()
            array([[ 1,  0],
                   [ 0, -1]])
        >>> c=Circuit()
        >>> c.push(GateZ(),0)
            1-qubit circuit with 1 instructions:
             └── Z @ q0
    """
    _num_qubits = 1
    _name = 'Z'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[1, 0], [0, -1]])


class GateH(Gate):
    """Single qubit Hadamard gate.

    **Matrix representation:**

    .. math::
        \\operatorname{H} = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            1 & 1 \\\\
            1 & -1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateH
        >>> GateH().matrix()
            array([[ 0.70710678,  0.70710678],
                   [ 0.70710678, -0.70710678]])
        >>> c=Circuit()
        >>> c.push(GateH(),0)
            1-qubit circuit with 1 instructions:
             └── H @ q0
    """
    _num_qubits = 1
    _name = 'H'

    def inverse(self):
        return self

    def matrix(self):
        return 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])


class GateS(Gate):
    """Single qubit gate S.

    It induces a :math:`\\frac{\\pi}{2}` phase gate.

    **Matrix representation:**

    .. math::
        \\operatorname{S} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & i
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateS
        >>> GateS().matrix()
            array([[1.000000e+00+0.j, 0.000000e+00+0.j],
                   [0.000000e+00+0.j, 6.123234e-17+1.j]])
        >>> c=Circuit()
        >>> c.push(GateS(),0)
            1-qubit circuit with 1 instructions:
             └── S @ q0
    """
    _num_qubits = 1
    _name = 'S'

    def inverse(self):
        return GateSDG()

    def matrix(self):
        # return np.array([[1, 0], [0, 1j]])
        return _decomplex(pmatrixpi(1 / 2))


class GateSDG(Gate):
    """Single qubit S-dagger gate (conjugate transpose of the S gate).

    **Matrix representation:**

    .. math::
        \\operatorname{S}^\\dagger = \\begin{pmatrix}
            1 & 0 \\\\
            0 & -i
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateSDG
        >>> GateSDG().matrix()
            array([[1.000000e+00+0.j, 0.000000e+00+0.j],
                   [0.000000e+00+0.j, 6.123234e-17-1.j]])
        >>> c=Circuit()
        >>> c.push(GateSDG(),0)
            1-qubit circuit with 1 instructions:
             └── SDG @ q0
    """
    _num_qubits = 1
    _name = 'SDG'

    def inverse(self):
        return GateS()

    def matrix(self):
        return _decomplex(pmatrixpi(-1 / 2))


class GateT(Gate):
    """ Single qubit T gate.

    **Matrix representation:**

    .. math::
        \\operatorname{T} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & \\exp\\left(\\frac{i\\pi}{4}\\right)
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateT
        >>> GateT().matrix()
            array([[1.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.70710678+0.70710678j]])
        >>> c=Circuit()
        >>> c.push(GateT(),0)
            1-qubit circuit with 1 instructions:
             └── T @ q0
    """
    _num_qubits = 1
    _name = 'T'

    def inverse(self):
        return GateTDG()

    def matrix(self):
        return _decomplex(pmatrixpi(1 / 4))


class GateTDG(Gate):
    """Single qubit T-dagger gate (conjugate transpose of the T gate).

    **Matrix representation:**

    .. math::
        \\operatorname{T}^\\dagger = \\begin{pmatrix}
            1 & 0 \\\\
            0 & \\exp\\left(\\frac{-i\\pi}{4}\\right)
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateTDG
        >>> GateTDG().matrix()
            array([[1.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.70710678-0.70710678j]])
        >>> c=Circuit()
        >>> c.push(GateTDG(),0)
            1-qubit circuit with 1 instructions:
             └── TDG @ q0
    """
    _num_qubits = 1
    _name = 'TDG'

    def inverse(self):
        return GateT()

    def matrix(self):
        return _decomplex(pmatrixpi(-1 / 4))


class GateSX(Gate):
    """Single qubit :math:`\\sqrt{X}` gate.

    **Matrix representation:**

    .. math::
        \\sqrt{\\operatorname{X}} = \\frac{1}{2} \\begin{pmatrix}
            1+i & 1-i \\\\
            1-i & 1+i
        \\end{pmatrix}

   Examples:
        >>> from mimiqcircuits import Circuit, GateSX
        >>> GateSX().matrix()
            array([[0.5+0.5j, 0.5-0.5j],
                   [0.5-0.5j, 0.5+0.5j]])
        >>> c=Circuit()
        >>> c.push(GateSX(),0)
            1-qubit circuit with 1 instructions:
             └── SX @ q0
    """
    _num_qubits = 1
    _name = 'SX'

    def inverse(self):
        return GateSXDG()

    def matrix(self):
        return _decomplex(gphase(np.pi / 4) * rxmatrix(np.pi / 2))


class GateSXDG(Gate):
    """Single qubit :math:`\\sqrt{X}^\\dagger` gate (conjugate transpose of the :math:`\\sqrt{X}` gate).
    
    **Matrix representation:**

    .. math::
        \\sqrt{\\operatorname{X}}^\\dagger = \\frac{1}{2} \\begin{pmatrix}
            1-i & 1+i \\\\
            1+i & 1-i
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateSXDG
        >>> GateSXDG().matrix()
            array([[0.5-0.5j, 0.5+0.5j],
                   [0.5+0.5j, 0.5-0.5j]])
        >>> c=Circuit()
        >>> c.push(GateSXDG(),0)
            1-qubit circuit with 1 instructions:
             └── SXDG @ q0
    """
    _num_qubits = 1
    _name = 'SXDG'

    def inverse(self):
        return GateSX()

    def matrix(self):
        return _decomplex(gphase(-np.pi / 4) * rxmatrix(-np.pi / 2))


class GateID(Gate):
    """Single qubit Identity gate.

    **Matrix representation:**

    .. math::
        \\operatorname{ID} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & 1
        \\end{pmatrix}

   Examples:
        >>> from mimiqcircuits import Circuit, GateID
        >>> GateID().matrix()
            array([[1., 0.],
                   [0., 1.]])
        >>> c=Circuit()
        >>> c.push(GateID(),0)
            1-qubit circuit with 1 instructions:
             └── ID @ q0
    """
    _num_qubits = 1
    _name = 'ID'

    def inverse(self):
        return self

    def matrix(self):
        return _decomplex(umatrixpi(0, 0, 0))


class GateU(Gate):
    """Single qubit generic unitary gate.

    **Matrix representation:**

    .. math::
        \\operatorname{U}(\\theta, \\phi, \\lambda) = \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\
            e^{i\\phi}\\sin\\frac{\\theta}{2} & e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): Euler angle 1 in radians.
        phi (float): Euler angle 2 in radians.
        lambda (float): Euler angle 3 in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateU
        >>> import nump as np
        >>> GateU(np.pi/3,np.pi/3,np.pi/3).matrix()
            array([[ 0.8660254+0.j       , -0.25     -0.4330127j],
                   [ 0.25     +0.4330127j, -0.4330127+0.75j     ]])
        >>> c=Circuit()
        >>> c.push(GateU(np.pi/3,np.pi/3,np.pi/3),0)
            1-qubit circuit with 1 instructions:
             └── U(theta=1.0471975511965976, phi=1.0471975511965976, lmbda=1.0471975511965976) @ q0
    """
    _num_qubits = 1
    _name = 'U'
    _parnames = ('theta', 'phi', 'lmbda')

    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

    def matrix(self):
        return umatrix(self.theta, self.phi, self.lmbda)

    def inverse(self):
        return GateU(-self.theta, -self.lmbda, -self.phi)


class GateU1(Gate):
    """Single qubit generic unitary gate :math:`{U_1}`.

    Equivalent to :func:`GateP`

    **Matrix representation:**

    .. math::
        \\operatorname{U1}(\\lambda) = \\begin{pmatrix}
            1 & 0 \\\\
            0 & e^{i\\lambda}
        \\end{pmatrix}

    Parameters:
        lambda (float): Euler angle 3 in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateU1
        >>> import nump as np
        >>> GateU1(np.pi/4).matrix()
            array([[1.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.70710678+0.70710678j]])
        >>> c=Circuit()
        >>> c.push(GateU1(np.pi/3),0)
            1-qubit circuit with 1 instructions:
             └── U1(lmbda=1.0471975511965976) @ q0
    """
    _num_qubits = 1
    _name = 'U1'
    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return pmatrix(self.lmbda)

    def inverse(self):
        return GateU1(-self.lmbda)


class GateU2(Gate):
    """Single qubit generic unitary gate :math:`{U_2}`.

    Equivalent to :func:`GateU2DG`

    **Matrix representation:**

    .. math::
        \\operatorname{U2}(\\phi,\\lambda) = \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
            1 & -e^{i\\lambda} \\\\
            e^{i\\phi} & e^{i(\\phi+\\lambda)}
        \\end{pmatrix}

    Parameters:
        phi (float): Euler angle in radians.
        lambda (float): Euler angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateU2
        >>> import nump as np
        >>> GateU2(np.pi/2,np.pi/4).matrix()
            array([[ 0.27059805-0.65328148j, -0.65328148+0.27059805j],
                   [ 0.65328148+0.27059805j,  0.27059805+0.65328148j]])
        >>> c=Circuit()
        >>> c.push(GateU2(np.pi/2,np.pi/4),0)
            1-qubit circuit with 1 instructions:
             └── U2(phi=1.5707963267948966, lmbda=0.7853981633974483) @ q0
    """
    _num_qubits = 1
    _name = 'U2'
    _parnames = ('phi', 'lmbda')

    def __init__(self, phi, lmbda):
        self.phi = phi
        self.lmbda = lmbda

    def matrix(self):
        return gphase(-(self.phi + self.lmbda)/2) * umatrixpi(1/2, (self.phi/np.pi), (self.lmbda/np.pi))

    def inverse(self):
        return GateU2DG(self.phi, self.lmbda,)


class GateU2DG(Gate):
    """Single qubit generic unitary gate :math:`{U_2}^\\dagger`.

    **Matrix representation:**

    .. math::
        \\operatorname{U2}^\\dagger(\\phi,\\lambda) = \\frac{1}{\\sqrt{2}}\\begin{pmatrix}
            1 & -e^{i\\lambda} \\\\
            e^{i\\phi} & e^{i(\\phi+\\lambda)}
        \\end{pmatrix}

    Parameters:
        phi (float): Euler angle 2 in radians.
        lambda (float): Euler angle 3 in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateU2DG
        >>> import nump as np
        >>> GateU2DG(np.pi/2, np.pi/4).matrix()
            array([[ 0.27059805+0.65328148j,  0.65328148-0.27059805j],
                   [-0.65328148-0.27059805j,  0.27059805-0.65328148j]])
        >>> import numpy as np
        >>> c = Circuit()
        >>> c.push(GateU2DG(np.pi/2, np.pi/4), 0)
            1-qubit circuit with 1 instructions:
             └── U2DG(phi=1.5707963267948966, lmbda=0.7853981633974483) @ q0
    """
    _num_qubits = 1
    _name = 'U2DG'
    _parnames = ('phi', 'lmbda')

    def __init__(self, phi, lmbda):
        self.phi = phi
        self.lmbda = lmbda

    def matrix(self):
        return gphase((self.phi + self.lmbda)/2) * umatrixpi(-1/2, (-self.lmbda/np.pi), (-self.phi/np.pi))

    def inverse(self):
        return GateU2(self.phi, self.lmbda,)


class GateU3(Gate):
    """Single qubit generic unitary gate :math:`{U_3}`.

    **Matrix representation:**

    .. math::
        \\operatorname{U3}(\\theta,\\phi,\\lambda) = \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -e^{i\\lambda}\\sin\\frac{\\theta}{2} \\\\
            e^{i\\phi}\\sin\\frac{\\theta}{2} & e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): Euler angle 1 in radians.
        phi (float): Euler angle 2 in radians.
        lambda (float): Euler angle 3 in radians.
   
    Examples:
        >>> from mimiqcircuits import Circuit, GateU3
        >>> import nump as np
        >>> GateU3(np.pi/2,np.pi/4,np.pi/2).matrix()
            array([[ 0.27059805-0.65328148j, -0.65328148-0.27059805j],
                   [ 0.65328148-0.27059805j,  0.27059805+0.65328148j]])
        >>> import numpy as np  
        >>> c=Circuit()
        >>> c.push(GateU3(np.pi/3,np.pi/3,np.pi/3),0)
            1-qubit circuit with 1 instructions:
             └── U3(theta=1.0471975511965976, phi=1.0471975511965976, lmbda=1.0471975511965976) @ q0
    """
    _num_qubits = 1
    _name = 'U3'
    _parnames = ('theta', 'phi', 'lmbda')

    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

    def matrix(self):
        return gphase(-(self.phi + self.lmbda)/2) * umatrix(self.theta, self.phi, self.lmbda)

    def inverse(self):
        return GateU3(-self.theta, -self.lmbda, -self.phi)


class GateR(Gate):
    """Single qubit Rotation gate around the axis :math:`\\cos(\\phi)\\hat{x} + \\sin(\\phi)\\hat{y}`.

    **Matrix representation:**

    .. math::
        \\operatorname R(\\theta,\\phi) = \\begin{pmatrix}
          \\cos \\frac{\\theta}{2}  & -i e^{-i\\phi} \\sin \\frac{\\theta}{2} \\\\
          -i e^{i \\phi} \\sin \\frac{\\theta}{2}  &  \\cos \\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): The rotation angle in radians.
        phi (float): The axis of rotation in radians.

    Example:
        >>> from mimiqcircuits import Circuit, GateR
        >>> import nump as np
        >>> GateR(np.pi/2,np.pi/4).matrix()
            array([[ 0.70710678+0.j , -0.5       -0.5j],
                   [ 0.5       -0.5j,  0.70710678+0.j ]])
        >>> c=Circuit()
        >>> c.push(GateR(np.pi/3,np.pi/3),0)
            1-qubit circuit with 1 instructions:
             └── R(theta=1.0471975511965976, phi=1.0471975511965976) @ q0
    """
    _num_qubits = 1
    _name = 'R'
    _parnames = ('theta', 'phi')

    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

    def matrix(self):
        return rmatrix(self.theta, self.phi)

    def inverse(self):
        return GateR(-self.theta, self.phi)


class GateRX(Gate):
    """Single qubit Rotation gate around the axis :math:`\\hat{x}`

    **Matrix representation:**

    .. math::
        \\operatorname{RX}(\\theta) = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
            -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta: Rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateRX
        >>> import nump as np
        >>> GateRX(np.pi/2).matrix()
            array([[ 7.07106781e-01+0.j        , -4.32978028e-17-0.70710678j],
                   [ 4.32978028e-17-0.70710678j,  7.07106781e-01+0.j        ]])
        >>> c=Circuit()
        >>> c.push(GateRX(np.pi/3),0)
            1-qubit circuit with 1 instructions:
             └── RX(theta=1.0471975511965976) @ q0
    """
    _num_qubits = 1
    _name = 'RX'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return rxmatrix(self.theta)

    def inverse(self):
        return GateRX(-self.theta)


class GateRY(Gate):
    """Single qubit Rotation gate around the axis :math:`\\hat{y}`

    **Matrix representation:**

    .. math::
        \\operatorname{RY}(\\theta) = \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -\\sin\\frac{\\theta}{2} \\\\
            \\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): Rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit,GateRY
        >>> import nump as np
        >>> GateRY(np.pi/2).matrix()
            array([[ 0.70710678+0.j, -0.70710678-0.j],
                   [ 0.70710678+0.j,  0.70710678+0.j]])
        >>> c=Circuit()
        >>> c.push(GateRY(np.pi/3),0)
            1-qubit circuit with 1 instructions:
             └── RY(theta=1.0471975511965976) @ q0
    """
    _num_qubits = 1
    _name = 'RY'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return rymatrix(self.theta)

    def inverse(self):
        return GateRY(-self.theta)


class GateRZ(Gate):
    """Single qubit Rotation gate around the axis :math:`\\hat{z}`

    **Matrix representation:**

    .. math::
        \\operatorname{RZ}(\\lambda) = \\begin{pmatrix}
            e^{-i\\frac{\\lambda}{2}} & 0 \\\\
            0 & e^{i\\frac{\\lambda}{2}}
        \\end{pmatrix}

    Parameters:
        lambda (float): Rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateRZ
        >>> import nump as np
        >>> GateRZ(np.pi/2).matrix()
            array([[0.70710678-0.70710678j, 0.        -0.j        ],
                   [0.        +0.j        , 0.70710678+0.70710678j]])
        >>> c=Circuit()
        >>> c.push(GateRZ(np.pi/3),0)
            1-qubit circuit with 1 instructions:
             └── RZ(theta=1.0471975511965976) @ q0
    """
    _num_qubits = 1
    _name = 'RZ'
    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return rzmatrix(self.lmbda)

    def inverse(self):
        return GateRZ(-self.lmbda)

class GateP(Gate):
    """Single qubit Phase gate.

    **Matrix representation:**

    .. math::
        \\operatorname{P}(\\lambda) = \\begin{pmatrix}
            1 & 0 \\\\
            0 & e^{i\\lambda}
        \\end{pmatrix}

    Parameters:
        lambda (float): Phase angle

    Examples:
        >>> from mimiqcircuits import Circuit, GateP
        >>> import nump as np
        >>> GateP(np.pi/4).matrix()
            array([[1.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.70710678+0.70710678j]])
        >>> c=Circuit()
        >>> c.push(GateP(np.pi/3),0)
            1-qubit circuit with 1 instructions:
             └── P(theta=1.0471975511965976) @ q0
    """
    _num_qubits = 1
    _name = 'P'
    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return pmatrix(self.lmbda)

    def inverse(self):
        return GateP(-self.lmbda)


class GateCX(Gate):
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
        >>> from mimiqcircuits import Circuit, GateCX
        >>> GateCX().matrix()
            array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 0., 1.],
                   [0., 0., 1., 0.]])
        >>> c=Circuit()
        >>> c.push(GateCX(),0,1)
            2-qubit circuit with 1 instructions:
             └── CX @ q0, q1
    """
    _num_qubits = 2
    _name = 'CX'

    def matrix(self):
        return ctrl(GateX().matrix())

    def inverse(self):
        return self


class GateCY(Gate):
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
        >>> from mimiqcircuits import Circuit,GateCY
        >>> GateCY().matrix()
            array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                   [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                   [ 0.+0.j,  0.+0.j,  0.+0.j, -0.-1.j],
                   [ 0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j]])
        >>> c=Circuit()
        >>> c.push(GateCY(),0,1)
            2-qubit circuit with 1 instructions:
             └── CY @ q0, q1
    """
    _num_qubits = 2
    _name = 'CY'

    def matrix(self):
        return ctrl(GateY().matrix())

    def inverse(self):
        return self


class GateCZ(Gate):
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
        >>> from mimiqcircuits import Circuit, GateCZ
        >>> GateCZ().matrix()
            array([[ 1.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  0.,  0., -1.]])
        >>> c=Circuit()
        >>> c.push(GateCZ(),0,1)
            2-qubit circuit with 1 instructions:
             └── CZ @ q0, q1
    """
    _num_qubits = 2
    _name = 'CZ'

    def matrix(self):
        return ctrl(GateZ().matrix())

    def inverse(self):
        return self


class GateCH(Gate):
    """Two qubit Controlled-Hadamard gate.
    
    **Matrix representation:**

    .. math::
        \\operatorname{CH} = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            1 & 1 & 0 & 0 \\\\
            1 & -1 & 0 & 0 \\\\
            0 & 0 & 1 & 1 \\\\
            0 & 0 & 1 & -1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit,GateCH
        >>> GateCH().matrix()
            array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                   [ 0.        ,  1.        ,  0.        ,  0.        ],
                   [ 0.        ,  0.        ,  0.70710678,  0.70710678],
                   [ 0.        ,  0.        ,  0.70710678, -0.70710678]])
        >>> c=Circuit()
        >>> c.push(GateCH(),0,1)
            2-qubit circuit with 1 instructions:
             └── CH @ q0, q1
    """
    _num_qubits = 2
    _name = 'CH'

    def matrix(self):
        return ctrl(GateH().matrix())

    def inverse(self):
        return self


class GateSWAP(Gate):
    """Two qubit SWAP gate.
    
    See Also :func:`GateISWAP`

    **Matrix representation:**

    .. math::
        \\operatorname{SWAP} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateSWAP
        >>> GateSWAP().matrix()
            array([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
        >>> c=Circuit()
        >>> c.push(GateSWAP(),0,1)
            2-qubit circuit with 1 instructions:
             └── SWAP @ q0, q1
    """
    _num_qubits = 2
    _name = 'SWAP'

    def matrix(self):
        return np.array([
            [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]
        ])

    def inverse(self):
        return self


class GateISWAP(Gate):
    """ Two qubit ISWAP gate.

    See Also :func:`GateISWAPDG` and :func:`GateSWAP`

    **Matrix representation:**

    .. math::
        \\operatorname{ISWAP} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & i & 0 \\\\
            0 & i & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateISWAP
        >>> GateISWAP().matrix()
            array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                   [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],
                   [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j],
                   [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
        >>> c=Circuit()
        >>> c.push(GateISWAP(),0,1)
            2-qubit circuit with 1 instructions:
             └── ISWAP @ q0, q1
    """
    _num_qubits = 2
    _name = 'ISWAP'

    def matrix(self):
        return np.array([
            [1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]
        ])

    def inverse(self):
        return GateISWAPDG()


class GateISWAPDG(Gate):
    """
    Two qubit :math:`{ISWAP}^{\\dagger}` gate (conjugate transpose of :math:`{ISWAP}` gate)

    See Also :func:`GateISWAP`, :func:`GateSWAP`

    **Matrix representation:**

    .. math::

        \\operatorname{ISWAP}^{\\dagger} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & -i & 0 \\\\
            0 & -i & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateISWAPDG
        >>> GateISWAPDG().matrix()
            array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                   [ 0.+0.j,  0.+0.j, -0.-1.j,  0.+0.j],
                   [ 0.+0.j, -0.-1.j,  0.+0.j,  0.+0.j],
                   [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
        >>> c=Circuit()
        >>> c.push(GateISWAPDG(),0,1)
            2-qubit circuit with 1 instructions:
             └── ISWAPDG @ q0, q1
    """
    _num_qubits = 2
    _name = 'ISWAPDG'

    def matrix(self):
        return np.array([
            [1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]
        ])

    def inverse(self):
        return GateISWAP()


class GateCU(Gate):
    """Two qubit generic unitary gate.

    equivalent to the [qiskit CU-Gate](https://qiskit.org/documentation/stubs/qiskit.circuit.library.CUGate.html)

    
    **Matrix representation:**

    .. math::
        \\operatorname{CU}(\\theta, \\phi, \\lambda, \\gamma) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & e^{i\\gamma}\\cos\\frac{\\theta}{2} & -e^{i(\\gamma+\\lambda)}\\sin\\frac{\\theta}{2} \\\\
            0 & 0 & e^{i(\\gamma+\\phi)}\\sin\\frac{\\theta}{2} & e^{i(\\gamma+\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): Euler angle 1 in radians.
        phi (float): Euler angle 2 in radians.
        lambda (float): Euler angle 3 in radians.
        gamma (float): Global phase of the CU gate.

    Examples:
        >>> from mimiqcircuits import Circuit, GateCU
        >>> GateCU(np.pi/3, np.pi/3, np.pi/3, 0).matrix()
            array([[ 1.       +0.j       ,  0.       +0.j       ,
                     0.       +0.j       ,  0.       +0.j       ],
                   [ 0.       +0.j       ,  1.       +0.j       ,
                     0.       +0.j       ,  0.       +0.j       ],
                   [ 0.       +0.j       ,  0.       +0.j       ,
                     0.8660254+0.j       , -0.25     -0.4330127j],
                   [ 0.       +0.j       ,  0.       +0.j       ,
                     0.25     +0.4330127j, -0.4330127+0.75j     ]])
        >>> c=Circuit()
        >>> c.push(GateCU(np.pi/3, np.pi/3, np.pi/3, 0),0,1)
            2-qubit circuit with 1 instructions:
             └── CU(theta=1.0471975511965976, phi=1.0471975511965976, lmbda=1.0471975511965976, gamma=0) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CU'
    _parnames = ('theta', 'phi', 'lmbda', 'gamma')

    def __init__(self, theta, phi, lmbda, gamma):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda
        self.gamma = gamma

    def matrix(self):
        return ctrl(umatrix(self.theta, self.phi, self.lmbda, self.gamma))

    def inverse(self):
        return GateCU(-self.theta, -self.lmbda, -self.phi, -self.gamma)


class GateCR(Gate):
    """Two qubit Controlled-R gate.

    See Also :func:`GateR`
    
    **Matrix representation:**

    .. math::
        \\operatorname{CR}(\\theta, \\phi) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & \\cos\\frac{\\theta}{2} & -i e^{-i\\phi}\\sin\\frac{\\theta}{2} \\\\
            0 & 0 & -i e^{i\\phi}\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters:
        theta (float): The rotation angle in radians.
        phi (float): The phase angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateCR
        >>> import numpy as np 
        >>> GateCR(np.pi/3, np.pi/3).matrix()
            array([[ 1.       +0.j  ,  0.       +0.j  ,  0.       +0.j  ,
                     0.       +0.j  ],
                   [ 0.       +0.j  ,  1.       +0.j  ,  0.       +0.j  ,
                     0.       +0.j  ],
                   [ 0.       +0.j  ,  0.       +0.j  ,  0.8660254+0.j  ,
                     -0.4330127-0.25j],
                   [ 0.       +0.j  ,  0.       +0.j  ,  0.4330127-0.25j,
                     0.8660254+0.j  ]])
        >>> c=Circuit()
        >>> c.push(GateCR(np.pi/3, np.pi/3),0,1)
            2-qubit circuit with 1 instructions:
             └── CR(theta=1.0471975511965976, phi=1.0471975511965976) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CR'
    _parnames = ('theta', 'phi')

    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

    def matrix(self):
        return ctrl(rmatrix(self.theta, self.phi))

    def inverse(self):
        return GateCR(-self.theta, self.phi)


class GateCRX(Gate):
    """Two qubit Controlled-RX gate.

    See Also :func:`GateRX`

    **Matrix representation:**

    .. math::
        \\operatorname{CRX}(\\theta) = \\begin{pmatrix}
                  1 & 0 & 0 & 0 \\\\
                  0 & 1 & 0 & 0 \\\\
                  0 & 0 & \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
                  0 & 0 & -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
              \\end{pmatrix}

    Parameters:
        theta (float): The rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateCRX
        >>> import numpy as np 
        >>> GateCRX(np.pi/2).matrix()
            array([[ 1.00000000e+00+0.j        ,  0.00000000e+00+0.j        ,
                     0.00000000e+00+0.j        ,  0.00000000e+00+0.j        ],
                   [ 0.00000000e+00+0.j        ,  1.00000000e+00+0.j        ,
                     0.00000000e+00+0.j        ,  0.00000000e+00+0.j        ],
                   [ 0.00000000e+00+0.j        ,  0.00000000e+00+0.j        ,
                     7.07106781e-01+0.j        , -4.32978028e-17-0.70710678j],
                   [ 0.00000000e+00+0.j        ,  0.00000000e+00+0.j        ,
                     4.32978028e-17-0.70710678j,  7.07106781e-01+0.j        ]])
        >>> c=Circuit()
        >>> c.push(GateCRX(np.pi/3),0,1)
            2-qubit circuit with 1 instructions:
             └── CRX(theta=1.0471975511965976) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CRX'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return ctrl(rxmatrix(self.theta))

    def inverse(self):
        return GateCRX(-self.theta)


class GateCRY(Gate):
    """Two qubit Controlled-RY gate.

    See Also :func:`GateRY`

    **Matrix representation:**

    .. math::
        \\operatorname{CRY}(\\theta) = \\begin{pmatrix}
                  1 & 0 & 0 & 0 \\\\
                  0 & 1 & 0 & 0 \\\\
                  0 & 0 & \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
                  0 & 0 &  \\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
              \\end{pmatrix}

    
    Parameters:
        theta (float): The rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateCRY
        >>> import numpy as np 
        >>> GateCRY(np.pi/2).matrix()
            array([[ 1.        +0.j,  0.        +0.j,  0.        +0.j,
                     0.        +0.j],
                   [ 0.        +0.j,  1.        +0.j,  0.        +0.j,
                     0.        +0.j],
                   [ 0.        +0.j,  0.        +0.j,  0.70710678+0.j,
                     -0.70710678-0.j],
                   [ 0.        +0.j,  0.        +0.j,  0.70710678+0.j,
                     0.70710678+0.j]])
        >>> c=Circuit()
        >>> c.push(GateCRY(np.pi/2),0,1)
            2-qubit circuit with 1 instructions:
             └── CRY(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CRY'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return ctrl(rymatrix(self.theta))

    def inverse(self):
        return GateCRY(-self.theta)


class GateCRZ(Gate):
    """Two qubit Controlled-RZ gate.

    See Also :func:`GateRZ`

    **Matrix representation:**

    .. math::
        \\operatorname{CRZ}(\\theta) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & e^{-i\\frac{\\lambda}{2}} & 0 \\\\
            0 & 0 & 0 & e^{i\\frac{\\lambda}{2}}
              \\end{pmatrix}

    Parameters:
        theta (float): The rotation angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateCRZ
        >>> import numpy as np 
        >>> GateCRZ(np.pi/2).matrix()
            array([[1.        +0.j        , 0.        +0.j        ,
                    0.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 1.        +0.j        ,
                    0.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.j        ,
                    0.70710678-0.70710678j, 0.        -0.j        ],
                   [0.        +0.j        , 0.        +0.j        ,
                    0.        +0.j        , 0.70710678+0.70710678j]])

        >>> c=Circuit()
        >>> c.push(GateCRZ(np.pi/2),0,1)
            2-qubit circuit with 1 gates
             └── CRZ(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CRZ'
    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return ctrl(rzmatrix(self.lmbda))

    def inverse(self):
        return GateCRZ(-self.lmbda)


class GateCP(Gate):
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
        param lambda (float): Phase angle in radians.
    
    Examples:
        >>> from mimiqcircuits import Circuit, GateCP
        >>> import numpy as np 
        >>> GateCP(np.pi/4).matrix()
            array([[0.92387953-0.38268343j, 0.        +0.j        ,
                    0.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.92387953+0.38268343j,
                    0.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.j        ,
                    0.92387953+0.38268343j, 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.j        ,
                    0.        +0.j        , 0.92387953-0.38268343j]])
        >>> c=Circuit()
        >>> c.push(GateCP(np.pi/2),0,1)
            2-qubit circuit with 1 instructions:
             └── CP(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CP'
    _parnames = ('lmbda',)

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return ctrl(pmatrix(self.lmbda))

    def inverse(self):
        return GateCP(-self.lmbda)


class GateRZZ(Gate):
    """Two qubit RZZ gate (rotation about ZZ)..

    This gate is symmetric, and is maximally entangling at :math:`(\\theta = \\frac{\\pi}{2})`
    
    **Matrix representation:**

    .. math::
        \\operatorname{RZZ}(\\theta) = \\begin{pmatrix}
            e^{-i\\frac{\\theta}{2}} & 0 & 0 & 0 \\\\
            0 & e^{i\\frac{\\theta}{2}} & 0 & 0 \\\\
            0 & 0 & e^{i\\frac{\\theta}{2}} & 0 \\\\
            0 & 0 & 0 & e^{-i\\frac{\\theta}{2}}
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateRZZ
        >>> import numpy as np 
        >>> GateRZZ(np.pi/4).matrix()
            array([[0.92387953-0.38268343j, 0.        +0.j        ,
                    0.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.92387953+0.38268343j,
                    0.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.j        ,
                    0.92387953+0.38268343j, 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.j        ,
                    0.        +0.j        , 0.92387953-0.38268343j]])
        >>> c=Circuit()
        >>> c.push(GateRZZ(np.pi/2),0,1)
            2-qubit circuit with 1 instructions:
             └── RZZ(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'RZZ'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return np.array([[cis(-self.theta / 2), 0, 0, 0],
                        [0, cis(self.theta / 2), 0, 0],
                        [0, 0, cis(self.theta / 2), 0],
                        [0, 0, 0, cis(-self.theta / 2)]], dtype=np.complex128)

    def inverse(self):
        return GateRZZ(-self.theta)


class GateRXX(Gate):
    """Two qubit RXX gate (rotation about XX).

    This gate is symmetric, and is maximally entangling at :math:`(\\theta = \\frac{\\pi}{2})`

    **Matrix representation:**

    .. math::

        \\operatorname{RXX}(\\theta) =\\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & 0 & 0 & -i\\sin(\\frac{\\theta}{2}) \\\\
            0 & \\cos(\\frac{\\theta}{2}) & -i\\sin(\\frac{\\theta}{2}) & 0 \\\\
            0 & -i\\sin(\\frac{\\theta}{2}) & \\cos(\\frac{\\theta}{2}) & 0 \\\\
            -i\\sin(\\frac{\\theta}{2}) & 0 & 0 & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta: The angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateRXX
        >>> import numpy as np 
        >>> GateRXX(np.pi/4).matrix()
            array([[0.92387953+0.j        , 0.        +0.j        ,
                    0.        +0.j        , 0.        -0.38268343j],
                   [0.        +0.j        , 0.92387953+0.j        ,
                    0.        -0.38268343j, 0.        +0.j        ],
                   [0.        +0.j        , 0.        -0.38268343j,
                    0.92387953+0.j        , 0.        +0.j        ],
                   [0.        -0.38268343j, 0.        +0.j        ,
                    0.        +0.j        , 0.92387953+0.j        ]])
        >>> c=Circuit()
        >>> c.push(GateRXX(np.pi/2),0,1)
            2-qubit circuit with 1 instructions:
             └── RXX(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'RXX'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return np.array([[np.cos(self.theta / 2), 0, 0, -1j*np.sin(self.theta / 2)],
                        [0, np.cos(self.theta / 2), -1j *
                         np.sin(self.theta / 2), 0],
                        [0, -1j*np.sin(self.theta / 2),
                         np.cos(self.theta / 2), 0],
                        [-1j*np.sin(self.theta / 2), 0, 0, np.cos(self.theta / 2)]], dtype=np.complex128)

    def inverse(self):
        return GateRXX(-self.theta)


class GateRYY(Gate):
    """Two qubit RYY gate (rotation about YY).
    
    This gate is symmetric, and is maximally entangling at :math:`(\\theta = \\frac{\\pi}{2})`
    
    **Matrix representation:**

    .. math::
        \\operatorname{RYY}(\\theta) =\\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & 0 & 0 & i\\sin(\\frac{\\theta}{2}) \\\\
            0 & \\cos(\\frac{\\theta}{2}) & -i\\sin(\\frac{\\theta}{2}) & 0 \\\\
            0 & -i\\sin(\\frac{\\theta}{2}) & \\cos(\\frac{\\theta}{2}) & 0 \\\\
            i\\sin(\\frac{\\theta}{2}) & 0 & 0 & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateRYY
        >>> import numpy as np 
        >>> GateRYY(np.pi/4).matrix()
            array([[0.92387953+0.j        , 0.        +0.j        ,
                    0.        +0.j        , 0.        +0.38268343j],
                   [0.        +0.j        , 0.92387953+0.j        ,
                    0.        -0.38268343j, 0.        +0.j        ],
                   [0.        +0.j        , 0.        -0.38268343j,
                    0.92387953+0.j        , 0.        +0.j        ],
                   [0.        +0.38268343j, 0.        +0.j        ,
                    0.        +0.j        , 0.92387953+0.j        ]])
        >>> c=Circuit()
        >>> c.push(GateRYY(np.pi/2),0,1)
            2-qubit circuit with 1 instructions:
             └── RYY(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'RYY'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return np.array([[np.cos(self.theta / 2), 0, 0, 1j*np.sin(self.theta / 2)],
                        [0, np.cos(self.theta / 2), -1j *
                         np.sin(self.theta / 2), 0],
                        [0, -1j*np.sin(self.theta / 2),
                         np.cos(self.theta / 2), 0],
                        [1j*np.sin(self.theta / 2), 0, 0, np.cos(self.theta / 2)]], dtype=np.complex128)

    def inverse(self):
        return GateRYY(-self.theta)


class GateRXZ(Gate):
    """ Two qubit RXZ gate.

    **Matrix representation:**

    .. math::

        \\operatorname{RXZ}(\\theta) = \\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & 0 & -i\\sin(\\frac{\\theta}{2}) & 0 \\\\
            0 & \\cos(\\frac{\\theta}{2}) & 0 & i\\sin(\\frac{\\theta}{2}) \\\\
            -i\\sin(\\frac{\\theta}{2}) & 0 & \\cos(\\frac{\\theta}{2}) & 0 \\\\
            0 & i\\sin(\\frac{\\theta}{2}) & 0 & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.
    
    Examples:
        >>> from mimiqcircuits import Circuit, GateRXZ
        >>> import numpy as np 
        >>> GateRXZ(np.pi/4).matrix()

        >>> array([[0.92387953+0.j        , 0.        +0.j        ,
                    0.        -0.38268343j, 0.        +0.j        ],
                   [0.        +0.j        , 0.92387953+0.j        ,
                    0.        +0.j        , 0.        +0.38268343j],
                   [0.        -0.38268343j, 0.        +0.j        ,
                    0.92387953+0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.38268343j,
                    0.        +0.j        , 0.92387953+0.j        ]])
        >>> c=Circuit()
        >>> c.push(GateRXZ(np.pi/2),0,1)
            2-qubit circuit with 1 instructions:
             └── RXZ(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'RXZ'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return np.array([[np.cos(self.theta/2), 0, -1j*np.sin(self.theta/2), 0],
                         [0, np.cos(self.theta/2), 0, 1j*np.sin(self.theta/2)],
                         [-1j*np.sin(self.theta/2), 0,
                          np.cos(self.theta/2), 0],
                         [0, 1j*np.sin(self.theta/2), 0, np.cos(self.theta/2)]], dtype=np.complex128)

    def inverse(self):
        return GateRXZ(-self.theta)


class GateRZX(Gate):
    """Two qubit RZX gate.
    
    This gate i is maximally entangling at :math:`(\\theta = \\frac{\\pi}{2})`

    **Matrix representation:**

    .. math::
        \\operatorname{RZX}(\\theta) =\\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & -i\\sin(\\frac{\\theta}{2}) & 0 & 0 \\\\
            -i\\sin(\\frac{\\theta}{2}) & \\cos(\\frac{\\theta}{2}) & 0 & 0 \\\\
            0 & 0 & \\cos(\\frac{\\theta}{2}) & i\\sin(\\frac{\\theta}{2}) \\\\
            0 & 0 & i\\sin(\\frac{\\theta}{2}) & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateRZX
        >>> import numpy as np 
        >>> GateRZX(np.pi/4).matrix()
            array([[0.92387953+0.j        , 0.        +0.j        ,
                    0.        -0.38268343j, 0.        +0.j        ],
                   [0.        +0.j        , 0.92387953+0.j        ,
                    0.        +0.j        , 0.        +0.38268343j],
                   [0.        -0.38268343j, 0.        +0.j        ,
                    0.92387953+0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.38268343j,
                    0.        +0.j        , 0.92387953+0.j        ]])
        >>> c=Circuit()
        >>> c.push(GateRZX(np.pi/2),0,1)
            2-qubit circuit with 1 instructions:
             └── RZX(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'RZX'
    _parnames = ('theta',)

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return np.array([[np.cos(self.theta/2), -1j*np.sin(self.theta/2), 0, 0],
                         [-1j*np.sin(self.theta/2),
                          np.cos(self.theta/2), 0, 0],
                         [0, 0, np.cos(self.theta/2),  1j *
                          np.sin(self.theta/2)],
                         [0, 0,  1j*np.sin(self.theta/2), np.cos(self.theta/2)]], dtype=np.complex128)

    def inverse(self):
        return GateRZX(-self.theta)


class GateXXplusYY(Gate):
    """Two qubit parametric XXplusYY gate.
    
    Also known as an XY gate. Its action is to induce a coherent rotation by some angle between :math:`\\ket{10}` and :math:`\\ket{01}`.

    **Matrix representation:**

    .. math::
        \\operatorname{XXplusYY}(\\theta, \\beta)= \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\cos(\\frac{\\theta}{2}) & -i\\sin(\\frac{\\theta}{2})e^{-i\\beta} & 0 \\\\
            0 & -i\\sin(\\frac{\\theta}{2})e^{i\\beta} & \\cos(\\frac{\\theta}{2}) & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.
        beta (float): The phase angle in radians.

    Examples:
        >>> from mimiqcircuits import Circuit, GateXXplusYY
        >>> import numpy as np 
        >>> GateXXplusYY(np.pi/2,np.pi/2).matrix()
            array([[ 1.        +0.00000000e+00j,  0.        +0.00000000e+00j,
                     0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
                   [ 0.        +0.00000000e+00j,  0.70710678+0.00000000e+00j,
                    -0.70710678-4.32978028e-17j,  0.        +0.00000000e+00j],
                   [ 0.        +0.00000000e+00j,  0.70710678-4.32978028e-17j,
                     0.70710678+0.00000000e+00j,  0.        +0.00000000e+00j],
                   [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
                     0.        +0.00000000e+00j,  1.        +0.00000000e+00j]])
        >>> c=Circuit()
        >>> c.push(GateXXplusYY(np.pi/2,np.pi/2),0,1)
            2-qubit circuit with 1 instructions:
             └──XXplusYY(theta=1.5707963267948966, beta=1.5707963267948966) @ q0, q1

    """
    _num_qubits = 2
    _name = 'XXplusYY'
    _parnames = ('theta', 'beta')

    def __init__(self, theta, beta):
        self.theta = theta
        self.beta = beta

    def matrix(self):
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(self.theta/2), -1j *
                          np.sin(self.theta/2) * cis(-self.beta), 0],
                         [0, -1j*np.sin(self.theta/2) * cis(self.beta),
                          np.cos(self.theta/2),  0],
                         [0, 0,  0, 1]], dtype=np.complex128)

    def inverse(self):
        return GateXXplusYY(-self.theta, self.beta)


class GateXXminusYY(Gate):
    """Two qubit parametric GateXXminusYY gate.
    
    Its action is to induce a coherent rotation by some angle between :math:`\\ket{00}` and :math:`\\ket{11}`

    **Matrix representation:**

    .. math::
        \\operatorname{XXminusYY}(\\theta, \\beta)=\\begin{pmatrix}
            \\cos(\\frac{\\theta}{2}) & 0 & 0 & -i\\sin(\\frac{\\theta}{2})e^{-i\\beta} \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            -i\\sin(\\frac{\\theta}{2})e^{i\\beta} & 0 & 0 & \\cos(\\frac{\\theta}{2})
        \\end{pmatrix}

    Parameters:
        theta (float): The angle in radians.
        beta (float): The angle in radians.

    Examples:
            >>> from mimiqcircuits import Circuit, GateXXminusYY
            >>> import numpy as np
            >>> GateXXminusYY(np.pi/2,np.pi/2).matrix()
                array([[ 0.70710678+0.00000000e+00j,  0.        +0.00000000e+00j,
                         0.        +0.00000000e+00j,  0.70710678-4.32978028e-17j],
                       [ 0.        +0.00000000e+00j,  1.        +0.00000000e+00j,
                         0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
                       [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
                         1.        +0.00000000e+00j,  0.        +0.00000000e+00j],
                       [-0.70710678-4.32978028e-17j,  0.        +0.00000000e+00j,
                         0.        +0.00000000e+00j,  0.70710678+0.00000000e+00j]])
            >>> c=Circuit()
            >>> c.push(GateXXminusYY(np.pi/2,np.pi/2),0,1)
                2-qubit circuit with 1 instructions:
                 └── XXminusYY(theta=1.5707963267948966, beta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'XXminusYY'
    _parnames = ('theta', 'beta')

    def __init__(self, theta, beta):
        self.theta = theta
        self.beta = beta

    def matrix(self):
        return np.array([[np.cos(self.theta/2), 0, 0, -1j*np.sin(self.theta/2) * cis(self.beta)],
                         [0, 1, 0, 0],
                         [0, 0, 1,  0],
                         [-1j*np.sin(self.theta/2) * cis(-self.beta), 0,  0, np.cos(self.theta/2)]], dtype=np.complex128)

    def inverse(self):
        return GateXXminusYY(-self.theta, self.beta)


class GateCS(Gate):
    """Two qubit Controlled-S gate.

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
        >>> from mimiqcircuits import Circuit, GateCS
        >>> GateCS().matrix()
            array([[1.000000e+00+0.j, 0.000000e+00+0.j, 0.000000e+00+0.j,
                    0.000000e+00+0.j],
                [0.000000e+00+0.j, 1.000000e+00+0.j, 0.000000e+00+0.j,
                    0.000000e+00+0.j],
                [0.000000e+00+0.j, 0.000000e+00+0.j, 1.000000e+00+0.j,
                    0.000000e+00+0.j],
                [0.000000e+00+0.j, 0.000000e+00+0.j, 0.000000e+00+0.j,
                    6.123234e-17+1.j]])
        >>> c=Circuit()
        >>> c.push(GateCS(),0,1)
            2-qubit circuit with 1 instructions:
            └── CS @ q0, q1
    """
    _num_qubits = 2
    _name = 'CS'

    def matrix(self):
        return ctrl(GateS().matrix())

    def inverse(self):
        return GateCSDG()


class GateCSDG(Gate):
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
        >>> from mimiqcircuits import Circuit, GateCSDG
        >>> GateCSDG().matrix()
            array([[1.000000e+00+0.j, 0.000000e+00+0.j, 0.000000e+00+0.j,
                    0.000000e+00+0.j],
                [0.000000e+00+0.j, 1.000000e+00+0.j, 0.000000e+00+0.j,
                    0.000000e+00+0.j],
                [0.000000e+00+0.j, 0.000000e+00+0.j, 1.000000e+00+0.j,
                    0.000000e+00+0.j],
                [0.000000e+00+0.j, 0.000000e+00+0.j, 0.000000e+00+0.j,
                    6.123234e-17-1.j]])
        >>> c=Circuit()
        >>> c.push(GateCSDG(),0,1)
            2-qubit circuit with 1 instructions:
            └── CSDG @ q0, q1
    """
    _num_qubits = 2
    _name = 'CSDG'

    def matrix(self):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, -1j]], dtype=np.complex128)

    def inverse(self):
        return GateCS()


class GateCSX(Gate):
    """Two qubit Controled-SX (control on second qubit) gate.

    **Matrix representation:**

    .. math::

        \\operatorname{CSX} =\\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\frac{1+i}{\\sqrt{2}} & 0 & \\frac{1-i}{\\sqrt{2}} \\\\
            0 & 0 & 1 & 0 \\\\
            0 & \\frac{1-i}{\\sqrt{2}} & 0 & \\frac{1+i}{\\sqrt{2}}
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateCSX
        >>> GateCSX().matrix()
            array([[1. +0.j , 0. +0.j , 0. +0.j , 0. +0.j ],
                [0. +0.j , 0.5+0.5j, 0. +0.j , 0.5-0.5j],
                [0. +0.j , 0. +0.j , 1. +0.j , 0. +0.j ],
                [0. +0.j , 0.5-0.5j, 0. +0.j , 0.5+0.5j]])
        >>> c=Circuit()
        >>> c.push(GateCSX(),0,1)
            2-qubit circuit with 1 instructions:
             └── CSX @ q0, q1
    """
    _num_qubits = 2
    _name = 'CSX'

    def matrix(self):
        return ctrl2(GateSX().matrix())

    def inverse(self):
        return GateCSXDG()


class GateCSXDG(Gate):
    """Two qubit :math:`{CSX}^\\dagger` gate.

    **Matrix representation:**

    .. math::
        \\operatorname{CSX}^{\\dagger} =\\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\frac{1-i}{\\sqrt{2}} & 0 & \\frac{1+i}{\\sqrt{2}} \\\\
            0 & 0 & 1 & 0 \\\\
            0 & \\frac{1+i}{\\sqrt{2}} & 0 & \\frac{1-i}{\\sqrt{2}}
        \\end{pmatrix}

    Examples:
            >>> from mimiqcircuits import Circuit, GateCSXDG
            >>> GateCSXDG().matrix()
                array([[1. +0.j , 0. +0.j , 0. +0.j , 0. +0.j ],
                       [0. +0.j , 0.5-0.5j, 0. +0.j , 0.5+0.5j],
                       [0. +0.j , 0. +0.j , 1. +0.j , 0. +0.j ],
                       [0. +0.j , 0.5+0.5j, 0. +0.j , 0.5-0.5j]])
            >>> c=Circuit()
            >>> c.push(GateCSXDG(),0,1)
                2-qubit circuit with 1 instructions:
                 └── CSXDG @ q0, q1
    """
    _num_qubits = 2
    _name = 'CSXDG'

    def matrix(self):
        return ctrl2(_decomplex(gphase(-np.pi / 4) * rxmatrix(-np.pi / 2)))

    def inverse(self):
        return GateCSX()


class GateECR(Gate):
    """Two qubit ECR (echo) gate.

    **Matrix representation:**

    .. math::
        \\operatorname{ECR} =\\begin{pmatrix}
            0 & \\frac{1}{\\sqrt{2}} & 0 & \\frac{i}{\\sqrt{2}} \\ \\\\
            \\frac{1}{\\sqrt{2}} & 0 & \\frac{-i}{\\sqrt{2}} & 0 \\\\
            0 & \\frac{i}{\\sqrt{2}} & 0 & \\frac{i}{\\sqrt{2}} \\\\
            \\frac{-i}{\\sqrt{2}} & 0 & \\frac{1}{\\sqrt{2}} & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit,GateECR
        >>> GateECR().matrix()
            array([[0.        +0.j        , 0.70710678+0.j        ,
                    0.        +0.j        , 0.        +0.70710678j],
                   [0.70710678+0.j        , 0.        +0.j        ,
                    0.        -0.70710678j, 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.70710678j,
                    0.        +0.j        , 0.        +0.70710678j],
                   [0.        -0.70710678j, 0.        +0.j        ,
                    0.70710678+0.j        , 0.        +0.j        ]])
        >>> c.push(GateECR(),0,1)
        >>> c=Circuit()
            2-qubit circuit with 1 instructions:
             └── ECR @ q0, q1
    """
    _num_qubits = 2
    _name = 'ECR'

    def matrix(self):
        return 1/np.sqrt(2) * np.array([[0, 1, 0, 1j],
                                        [1, 0, -1j, 0],
                                        [0, 1j, 0, 1],
                                        [-1j, 0, 1, 0]], dtype=np.complex128)

    def inverse(self):
        return self


class GateDCX(Gate):
    """Two qubit double-CNOT gate.

    A two qubit Clifford gate consisting of two back-to-back CNOTs with alternate controls.

    **Matrix representation:**

    .. math::
        \\operatorname{DCX} =\\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateDCX
        >>> GateDCX().matrix()
            array([[1., 0., 0., 0.],
                [0., 0., 0., 1.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.]])
        >>> c=Circuit()
        >>> c.push(GateDCX(),0,1)
            2-qubit circuit with 1 instructions:
            └── DCX @ q0, q1
    """
    _num_qubits = 2
    _name = 'DCX'

    def matrix(self):
        return ctrl_fs(GateX().matrix())

    def inverse(self):
        return GateDCXDG()


class GateDCXDG(Gate):
    """Two qubit :math:`{DCX}^\\dagger` gate.

    See Also :func:`GateDCX`

    **Matrix representation:**

    .. math::
        \\operatorname{DCX}^{\\dagger} =\\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            0 & 1 & 0 & 0
        \\end{pmatrix}

    Examples:
        >>> from mimiqcircuits import Circuit, GateDCXDG
        >>> GateDCXDG().matrix()
            array([[0.        +0.j        , 0.70710678+0.j        ,
                    0.        +0.j        , 0.        -0.70710678j],
                   [0.70710678+0.j        , 0.        +0.j        ,
                    0.        +0.70710678j, 0.        +0.j        ],
                   [0.        +0.j        , 0.        -0.70710678j,
                    0.        +0.j        , 0.        +0.70710678j],
                   [0.        +0.70710678j, 0.        +0.j        ,
                    0.70710678+0.j        , 0.        +0.j        ]])

        >>> c=Circuit()
        >>> c.push(GateDCXDG(),0,1)
            2-qubit circuit with 1 instructions:
             └── DCXDG @ q0, q1
    """
    _num_qubits = 2
    _name = 'DCXDG'

    def matrix(self):
        return ctrl_sf(GateX().matrix())

    def inverse(self):
        return GateDCX()


class GateCustom(Gate):
    """One or Two qubit Custom gates.
    
    Examples:
        >>> from mimiqcircuits import Circuit,GateCustom
        >>> import numpy as np
        >>> matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 1j, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])
        >>> qubits=2
        >>> c=Circuit()
        >>> c.push(GateCustom(matrix, qubits), 0, 1)
            2-qubit circuit with 1 instructions:
            └── Custom @ q0, q1
    """

    def __init__(self, matrix=None, num_qubits=None):
        super().__init__()
        if matrix is not None:
            if not isinstance(matrix, np.ndarray):
                raise TypeError("matrix must be a NumPy array")
            if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("matrix must be a square array")

        self._matrix = matrix
        self._num_qubits = num_qubits

    _name = 'Custom'

    @property
    def matrix(self):
        return self._matrix

    @property
    def num_qubits(self):
        return self._num_qubits

    def inverse(self):
        return GateCustom(np.linalg.inv(self.matrix))


GATES = {
    c._name: c for name, c in globals().items()
    if inspect.isclass(c) and hasattr(c, '_name')
}

# export all the gates
__all__ = [name for name in globals() if name.startswith('Gate')]
