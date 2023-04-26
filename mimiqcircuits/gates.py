#
# Copyright © 2023 University of Strasbourg. All Rights Reserved.
# See AUTHORS.md for the list of authors.
#

import numpy as np
import inspect
from abc import ABC, abstractmethod


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


def gphase(lmbda):
    return np.array([[1, 0], [0, cis(lmbda)]])


def gphasepi(lmbda):
    return np.array([[1, 0], [0, cis(lmbda)]])


def umatrix(theta, phi, lmbda, gamma=0.0):
    costheta2 = np.cos(theta / 2)
    sintheta2 = np.sin(theta / 2)
    return np.array([
        [cis(gamma) * costheta2, -cis(lmbda + gamma) * sintheta2],
        [cis(phi + gamma) * sintheta2, cis(phi + lmbda + gamma) * costheta2]
    ])


def umatrixpi(theta, phi, lmbda, gamma=0.0):
    costheta2 = np.cos(theta/2 * np.pi)
    sintheta2 = np.sin(theta/2 * np.pi)
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


class Gate(ABC):
    _num_qubits = None
    _name = None

    @abstractmethod
    def matrix(self):
        pass

    @property
    def num_qubits(self):
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, value):
        raise ValueError('Cannot set num_qubits. Read only parameter.')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise ValueError('Cannot set name. Read only parameter.')

    @ abstractmethod
    def inverse(self):
        pass

    @staticmethod
    def from_json(d):
        name = d['name']
        if 'params' in d:
            return GATES[name](*d['params'])

        return GATES[name]()

    def to_json(self):
        return {'name': self.name}

    def __str__(self):
        return self._name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class GateX(Gate):
    """
    Class for Single qubit Pauli-X gate.

    #Matrix Representation
    .. math::
    
        \\operatorname{X} = \\begin{pmatrix}
            0 & 1 \\\\
            1 & 0
        \\end{pmatrix}
    :return: The Pauli-X operator X.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateX
    >>> GateX().matrix()

    array([[0, 1],
       [1, 0]])

    >>> c=Circuit()
    >>> c.add_gate(GateX(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── X @ q0
     ```
    """
    _num_qubits = 1
    _name = 'X'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[0, 1], [1, 0]])


class GateY(Gate):
    """
    Class for Single qubit Pauli-Y gate.
    
    # Matrix Representation
    .. math::
    
        \\operatorname{Y} = \\begin{pmatrix}
            0 & -i \\\\
            i & 0
        \\end{pmatrix}
    :return: The Pauli-Y operator Y.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateY
    >>> GateY().matrix()

    array([[ 0.+0.j, -0.-1.j],
      [ 0.+1.j,  0.+0.j]])

    >>> c=Circuit()
    >>> c.add_gate(GateY(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── Y @ q0
     ```
    """
    _num_qubits = 1
    _name = 'Y'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[0, -1j], [1j, 0]])


class GateZ(Gate):
    """
    Class for Single qubit Pauli-Z gate.

    # Matrix Representation
    .. math::

        \\operatorname{Z} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & -1
        \\end{pmatrix}
    :return: The Pauli-Z operator Z.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateZ
    >>> GateZ().matrix()

    array([[ 1,  0],
       [ 0, -1]])

    >>> c=Circuit()
    >>> c.add_gate(GateZ(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── Z @ q0
     ```
    """
    
    _num_qubits = 1
    _name = 'Z'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[1, 0], [0, -1]])


class GateH(Gate):
    """
    Class for Single qubit Hadamard gate.

    # Matrix Representation
    .. math::

        \\operatorname{H} = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            1 & 1 \\\\
            1 & -1
        \\end{pmatrix}

    :return: The Hadamard gate H.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateH
    >>> GateH().matrix()

    array([[ 0.70710678,  0.70710678],
       [ 0.70710678, -0.70710678]])

    >>> c=Circuit()
    >>> c.add_gate(GateH(),0)
    >>> print(c)
     1-qubit circuit with 1 gates:
     └── H @ q0
     ```
    """
    _num_qubits = 1
    _name = 'H'

    def inverse(self):
        return self

    def matrix(self):
        return 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])


class GateS(Gate):
    """
    Class for Single qubit S gate (or Phase gate).

    # Matrix Representation
    .. math::

        \\operatorname{S} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & i
        \\end{pmatrix}

    :return: The S gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateS
    >>> GateS().matrix()

    array([[1.000000e+00+0.j, 0.000000e+00+0.j],
       [0.000000e+00+0.j, 6.123234e-17+1.j]])

    >>> c=Circuit()
    >>> c.add_gate(GateS(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── S @ q0
     ```
    """
    _num_qubits = 1
    _name = 'S'

    def inverse(self):
        return GateSDG()

    def matrix(self):
        # return np.array([[1, 0], [0, 1j]])
        return _decomplex(pmatrixpi(1 / 2))


class GateSDG(Gate):
    """
    Class for Single qubit S-dagger gate (conjugate transpose of the S gate).

    # Matrix Representation
    .. math::

        \\operatorname{S}^\\dagger = \\begin{pmatrix}
            1 & 0 \\\\
            0 & -i
        \\end{pmatrix}

    :return: The S-dagger gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateSDG

    >>> GateSDG().matrix()

    array([[1.000000e+00+0.j, 0.000000e+00+0.j],
       [0.000000e+00+0.j, 6.123234e-17-1.j]])

    >>> c=Circuit()
    >>> c.add_gate(GateSDG(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── SDG @ q0
     ```
    """
    _num_qubits = 1
    _name = 'SDG'

    def inverse(self):
        return GateS()

    def matrix(self):
        return _decomplex(pmatrixpi(-1 / 2))


class GateT(Gate):
    """
    Class for Single qubit T gate.

    # Matrix Representation
    .. math::
    
        \\operatorname{T} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & \\exp\\left(\\frac{i\\pi}{4}\\right)
        \\end{pmatrix}
    :return: The T gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateT
    
    >>> GateT().matrix()

    array([[1.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.70710678+0.70710678j]])

    >>> c=Circuit()
    >>> c.add_gate(GateT(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── T @ q0
     ```
    """
    _num_qubits = 1
    _name = 'T'

    def inverse(self):
        return GateTDG()

    def matrix(self):
        return _decomplex(pmatrixpi(1 / 4))


class GateTDG(Gate):
    """
    Class for Single qubit T-dagger gate (conjugate transpose of the T gate).

    # Matrix Representation
    .. math::

        \\operatorname{T}^\\dagger = \\begin{pmatrix}
            1 & 0 \\\\
            0 & \\exp\\left(\\frac{-i\\pi}{4}\\right)
        \\end{pmatrix}

    :return: The T-dagger gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateTDG
    
    >>> GateTDG().matrix()

    array([[1.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.70710678-0.70710678j]])

    >>> c=Circuit()
    >>> c.add_gate(GateTDG(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── TDG @ q0
     ```
    """
    _num_qubits = 1
    _name = 'TDG'

    def inverse(self):
        return GateT()

    def matrix(self):
        return _decomplex(pmatrixpi(-1 / 4))


class GateSX(Gate):
    """
    Class for Single qubit √X gate.

    # Matrix Representation
    .. math::

        \\sqrt{\\operatorname{X}} = \\frac{1}{2} \\begin{pmatrix}
            1+i & 1-i \\\\
            1-i & 1+i
        \\end{pmatrix}

    :return: The √X gate gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateSX
    
    >>> GateSX().matrix()

    array([[0.92387953+0.j        , 0.        -0.j        ],
       [0.        +0.j        , 0.89515836+0.22857145j]])

    >>> c=Circuit()
    >>> c.add_gate(GateSX(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── SX @ q0
     ```
    """
    _num_qubits = 1
    _name = 'SX'

    def inverse(self):
        return GateSXDG()

    def matrix(self):
        return _decomplex(gphasepi(1 / 4) * rxmatrixpi(1 / 4))


class GateSXDG(Gate):
    """
    Class for Single qubit √X-dagger gate (conjugate transpose of the √X gate).

    # Matrix Representation
    .. math::

        \\sqrt{\\operatorname{X}}^\\dagger = \\frac{1}{2} \\begin{pmatrix}
            1-i & 1+i \\\\
            1+i & 1-i
        \\end{pmatrix}

    :return: The conjugate transpose of the √X gate gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateSXDG
    
    >>> GateSXDG().matrix()

    array([[ 0.70710678+0.j ,  0.        +0.j ],
       [-0.        +0.j ,  0.5       -0.5j]])

    >>> c=Circuit()
    >>> c.add_gate(GateSXDG(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── SXDG @ q0
     ```
    """
    _num_qubits = 1
    _name = 'SXDG'

    def inverse(self):
        return GateSX()

    def matrix(self):
        return _decomplex(gphase(-np.pi / 4) * rxmatrix(-np.pi / 2))


class GateID(Gate):
    """
    Class for Single qubit Identity gate.

    # Matrix Representation
    .. math::

        \\operatorname{I} = \\begin{pmatrix}
            1 & 0 \\\\
            0 & 1
        \\end{pmatrix}

    :return: The identity gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateID
    
    >>> GateID().matrix()

    array([[1., 0.],
       [0., 1.]])

    >>> c=Circuit()
    >>> c.add_gate(GateID(),0)
    >>> print(c)

     1-qubit circuit with 1 gates:
     └── ID @ q0
     ```
    """
    _num_qubits = 1
    _name = 'ID'

    def inverse(self):
        return self

    def matrix(self):
        return _decomplex(umatrixpi(0, 0, 0))


class GateU(Gate):
    """
    Class for Single qubit generic unitary gate.

    # Arguments
    :param theta: Euler angle 1 in radians.
    :type theta: float
    :param phi: Euler angle 2 in radians.
    :type phi: float
    :param lmbda: Euler angle 3 in radians.
    :type lmbda: float

    # Matrix Representation
    .. math::

        \\operatorname{U}(\\theta,\\phi,\\lambda) = \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -e^{i\\lambda}\\sin\\frac{\\theta}{2} \\\\
            e^{i\\phi}\\sin\\frac{\\theta}{2} & e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    :return: generic unitary gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateU
    
    >>> GateU(np.pi/3,np.pi/3,np.pi/3).matrix()

    array([[ 0.8660254+0.j       , -0.25     -0.4330127j],
       [ 0.25     +0.4330127j, -0.4330127+0.75j     ]])

    >>> c=Circuit()
    >>> c.add_gate(GateU(np.pi/3,np.pi/3,np.pi/3),0)
    >>> print(c)

     1-qubit circuit with 1 gates
     └── U(theta=1.0471975511965976, phi=1.0471975511965976, lmbda=1.0471975511965976) @ q0
     ```
    """
    _num_qubits = 1
    _name = 'U'

    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

    def matrix(self):
        return umatrix(self.theta, self.phi, self.lmbda)

    def inverse(self):
        return GateU(-self.theta, -self.phi, -self.lmbda)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.theta, self.phi, self.lmbda]
        }

    def __str__(self):
        pars = f'(theta={self.theta}, phi={self.phi}, lmbda={self.lmbda})'
        return self.name + pars


class GateR(Gate):
    """
    Class for Single qubit Rotation gate around the axis cos(ϕ)x + sin(ϕ)y.

    # Arguments
    :param theta: The rotation angle in radians.
    :type theta: float
    :param phi: The axis of rotation in radians.
    :type phi: float

    # Matrix Representation
    .. math::
        \\operatorname R(\\theta,\\phi) = \\begin{pmatrix}
          \\cos \\frac{\\theta}{2}  & -i e^{-i\\phi} \\sin \\frac{\\theta}{2} \\\\
          -i e^{i \\phi} \\sin \\frac{\\theta}{2}  &  \\cos \\frac{\\theta}{2}
        \\end{pmatrix}

    :return: Rotation gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateR
    
    >>> GateR(np.pi/2,np.pi/4).matrix()

    array([[ 0.70710678+0.j , -0.5       -0.5j],
       [ 0.5       -0.5j,  0.70710678+0.j ]])

    >>> c=Circuit()
    >>> c.add_gate(GateR(np.pi/3,np.pi/3),0)
    >>> print(c)

     1-qubit circuit with 1 gates
     └── R(theta=1.0471975511965976, phi=1.0471975511965976) @ q0
     ```
    """
    _num_qubits = 1
    _name = 'R'

    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

    def matrix(self):
        return rmatrix(self.theta, self.phi)

    def inverse(self):
        return GateR(-self.theta, self.phi)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.theta, self.phi]
        }

    def __str__(self):
        pars = f'(theta={self.theta}, phi={self.phi})'
        return self.name + pars


class GateRX(Gate):
    """
    Class for Single qubit Rotation-X gate (RX gate).

    # Arguments
    :param theta: Rotation angle in radians.
    :type theta: float

    # Matrix Representation
    .. math::

        \\operatorname{RX}(\\theta) = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
            -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    :return: Rotation-X gate (RX gate).
    :rtype: numpy.ndarray 

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateRX
    
    >>> GateRX(np.pi/2).matrix()

    array([[ 7.07106781e-01+0.j        , -4.32978028e-17-0.70710678j],
       [ 4.32978028e-17-0.70710678j,  7.07106781e-01+0.j        ]])

    >>> c=Circuit()
    >>> c.add_gate(GateRX(np.pi/3),0)
    >>> print(c)

     1-qubit circuit with 1 gates
     └── RX(theta=1.0471975511965976) @ q0
     ```
    """
    _num_qubits = 1
    _name = 'RX'

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return rxmatrix(self.theta)

    def inverse(self):
        return GateRX(-self.theta)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.theta]
        }

    def __str__(self):
        pars = f'(theta={self.theta})'
        return self.name + pars


class GateRY(Gate):
    """
    Class for Single qubit Rotation-Y gate (RY gate).

    # Arguments
    :param theta: Rotation angle in radians.
    :type theta: float

    # Matrix Representation
    .. math::

        \\operatorname{RY}(\\theta) = \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -\\sin\\frac{\\theta}{2} \\\\
            \\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    :return: Rotation-Y gate (RY gate).
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateRY
    
    >>> GateRY(np.pi/2).matrix()

    array([[ 0.70710678+0.j, -0.70710678-0.j],
       [ 0.70710678+0.j,  0.70710678+0.j]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateRY(np.pi/3),0)
    >>> print(c)

     1-qubit circuit with 1 gates
     └── RY(theta=1.0471975511965976) @ q0
     ```
    """
    _num_qubits = 1
    _name = 'RY'

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return rymatrix(self.theta)

    def inverse(self):
        return GateRY(-self.theta)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.theta]
        }

    def __str__(self):
        pars = f'(lmbda={self.theta})'
        return self.name + pars


class GateRZ(Gate):
    """
    Class for Single qubit Rotation-Z gate (RZ gate).

    # Arguments
    :param lmbda: Rotation angle in radians.
    :type lmbda: float

    # Matrix Representation
    .. math::

        \\operatorname{RZ}(\\lambda) = \\begin{pmatrix}
            e^{-i\\frac{\\lambda}{2}} & 0 \\\\
            0 & e^{i\\frac{\\lambda}{2}}
        \\end{pmatrix}

    :return: Rotation-Z gate (RZ gate).
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateRZ
    
    >>> GateRZ(np.pi/2).matrix()

    array([[1.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.24740396+0.96891242j]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateRZ(np.pi/3),0)
    >>> print(c)

     1-qubit circuit with 1 gates
     └── RZ(theta=1.0471975511965976) @ q0
     ```
    """
    _num_qubits = 1
    _name = 'RZ'

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return rzmatrix(self.lmbda)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.lmbda]
        }

    def inverse(self):
        return GateRZ(-self.lmbda)

    def __str__(self):
        pars = f'(theta={self.lmbda})'
        return self.name + pars


class GateP(Gate):
    """
    Class for Single qubit Phase gate.

    # Arguments
    :param lmbda: Rotation angle in radians.
    :type lmbda: float

    # Matrix Representation
    .. math::

        \\operatorname{P}(\\lambda) = \\begin{pmatrix}
            1 & 0 \\\\
            0 & e^{i\\lambda}
        \\end{pmatrix}

    :return: Phase gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateP
    
    >>> GateP(np.pi/4).matrix()

    array([[1.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.70710678+0.70710678j]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateP(np.pi/3),0)
    >>> print(c)

     1-qubit circuit with 1 gates
     └── P(theta=1.0471975511965976) @ q0
     ```
    """
    _num_qubits = 1
    _name = 'P'

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return pmatrix(self.lmbda)

    def inverse(self):
        return GateP(-self.lmbda)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.lmbda]
        }

    def __str__(self):
        pars = f'(lmbda={self.lmbda})'
        return self.name + pars


class GateCX(Gate):
    """
    Two qubit Controlled-X gate (or CNOT).

    .. math::

        \\operatorname{CX} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            0 & 0 & 1 & 0
        \\end{pmatrix}

    :return: Controlled-X gate (or CNOT).
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCX
    
    >>> GateCX().matrix()

    array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 1., 0.]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateCX(),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CX @ q0, q1
    """
    _num_qubits = 2
    _name = 'CX'

    def matrix(self):
        return ctrl(GateX().matrix())

    def inverse(self):
        return self


class GateCY(Gate):
    """
    Two qubit Controlled-Y gate.

    ... math::

        \\operatorname{CY} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & -i \\\\
            0 & 0 & i & 0
        \\end{pmatrix}

    :return: Controlled-Y gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCY
    
    >>> GateCY().matrix()

    array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
       [ 0.+0.j,  0.+0.j,  0.+0.j, -0.-1.j],
       [ 0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateCY(),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CY @ q0, q1
    """
    _num_qubits = 2
    _name = 'CY'

    def matrix(self):
        return ctrl(GateY().matrix())

    def inverse(self):
        return self


class GateCZ(Gate):
    """
    Two qubit Controlled-Z gate.

    .. math::

        \\operatorname{CZ} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & -1
        \\end{pmatrix}

    :return: Controlled-Z gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCZ
    
    >>> GateCZ().matrix()

    array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0., -1.]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateCZ(),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CZ @ q0, q1
    """
    _num_qubits = 2
    _name = 'CZ'

    def matrix(self):
        return ctrl(GateZ().matrix())

    def inverse(self):
        return self


class GateCH(Gate):
    """
    Two qubit Controlled-Hadamard gate.

    .. math::

        \\operatorname{CH} = \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            1 & 1 & 0 & 0 \\\\
            1 & -1 & 0 & 0 \\\\
            0 & 0 & 1 & 1 \\\\
            0 & 0 & 1 & -1
        \\end{pmatrix}

    :return: Controlled-Hadamard gate (or CH).
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCH
    
    >>> GateCH().matrix()

    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.70710678,  0.70710678],
       [ 0.        ,  0.        ,  0.70710678, -0.70710678]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateCH(),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CH @ q0, q1
    """
    _num_qubits = 2
    _name = 'CH'

    def matrix(self):
        return ctrl(GateH().matrix())

    def inverse(self):
        return self


class GateSWAP(Gate):
    """
    Two qubit SWAP gate.
    See also [`GateISWAP`](@ref)

    .. math::

        \\operatorname{SWAP} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    :return: SWAP gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateSWAP
    
    >>> GateSWAP().matrix()

    array([[1, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateSWAP(),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
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
    """
    Two qubit ISWAP gate.
    See also [`GateISWAPDG`](@ref), [`GateSWAP`](@ref)

    .. math::

        \\operatorname{ISWAP} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & i & 0 \\\\
            0 & i & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    :return: ISWAP gate.
    :rtype: numpy.ndarray
    
    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateISWAP
    
    >>> GateISWAP().matrix()

    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],
       [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateISWAP(),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
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
    Two qubit ISWAP-dagger gate (conjugate transpose of ISWAP)
    See also [`GateISWAP`](@ref), [`GateSWAP`](@ref)

    .. math::

        \\operatorname{iSWAPDG} = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & -i & 0 \\\\
            0 & -i & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{pmatrix}

    :return: ISWAP dagger (or inverse ISWAP) gate.
    :rtype: numpy.ndarray
    
    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateISWAPDG
    
    >>> GateISWAPDG().matrix()

    array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
       [ 0.+0.j,  0.+0.j, -0.-1.j,  0.+0.j],
       [ 0.+0.j, -0.-1.j,  0.+0.j,  0.+0.j],
       [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
       
    >>> c=Circuit()
    >>> c.add_gate(GateISWAPDG(),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
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
    """
    Two qubit generic unitary gate, equivalent to the qiskit CU-Gate.
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.CUGate.html`

    # Arguments

    :param theta: Euler angle 1 in radians.
    :type theta: float
    :param phi: Euler angle 2 in radians.
    :type phi: float
    :param lambda: Euler angle 3 in radians.
    :type lambda: float
    :param gamma: Global phase of the CU gate.
    :type gamma: float

    **Matrix Representation:**

    .. math::
        \\operatorname{CU}(\\theta, \\phi, \\lambda, \\gamma) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & e^{i\\gamma}\\cos\\frac{\\theta}{2} & -e^{i(\\gamma+\\lambda)}\\sin\\frac{\\theta}{2} \\\\
            0 & 0 & e^{i(\\gamma+\\phi)}\\sin\\frac{\\theta}{2} & e^{i(\\gamma+\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    :return: Controlled-U gate (CU).
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCU
    
    >>> GateCU().matrix()

    array([[ 1.       +0.j       ,  0.       +0.j       ,
         0.       +0.j       ,  0.       +0.j       ],
       [ 0.       +0.j       ,  1.       +0.j       ,
         0.       +0.j       ,  0.       +0.j       ],
       [ 0.       +0.j       ,  0.       +0.j       ,
         0.8660254+0.j       , -0.25     -0.4330127j],
       [ 0.       +0.j       ,  0.       +0.j       ,
         0.25     +0.4330127j, -0.4330127+0.75j     ]])
    
    import numpy as np 
    >>> c=Circuit()
    >>> c.add_gate(GateCU(np.pi/3, np.pi/3, np.pi/3, 0),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CU(theta=1.0471975511965976, phi=1.0471975511965976, lmbda=1.0471975511965976, gamma=0) @ q0, q1

    """
    _num_qubits = 2
    _name = 'CU'

    def __init__(self, theta, phi, lmbda, gamma):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda
        self.gamma = gamma

    def matrix(self):
        return ctrl(umatrix(self.theta, self.phi, self.lmbda, self.gamma))

    def inverse(self):
        return GateCU(-self.theta, -self.phi, -self.lmbda, -self.gamma)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.theta, self.phi, self.lmbda, self.gamma]
        }

    def __str__(self):
        pars = f'(theta={self.theta}, phi={self.phi}, lmbda={self.lmbda}'
        pars += f', gamma={self.gamma})'
        return self.name + pars


class GateCR(Gate):
    """
    Two qubit Controlled-R gate.

    # Arguments
    :param theta: The rotation angle in radians.
    :type theta: float
    :param phi: The phase angle in radians.
    :type phi: float

    .. math::

        \\operatorname{CR}(\\theta) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
            0 & 0 & -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}e^{i\\phi}
        \\end{pmatrix}

    :return: Controlled-R gate.
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCR
    
    >>> GateCR(np.pi/3, np.pi/3).matrix()

    array([[ 1.       +0.j  ,  0.       +0.j  ,  0.       +0.j  ,
         0.       +0.j  ],
       [ 0.       +0.j  ,  1.       +0.j  ,  0.       +0.j  ,
         0.       +0.j  ],
       [ 0.       +0.j  ,  0.       +0.j  ,  0.8660254+0.j  ,
        -0.4330127-0.25j],
       [ 0.       +0.j  ,  0.       +0.j  ,  0.4330127-0.25j,
         0.8660254+0.j  ]])
    
    >>> import numpy as np 
    >>> c=Circuit()
    >>> c.add_gate(GateCR(np.pi/3, np.pi/3),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CR(theta=1.0471975511965976, phi=1.0471975511965976) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CR'

    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

    def matrix(self):
        return ctrl(rmatrix(self.theta, self.phi))

    def inverse(self):
        return GateCR(-self.theta, self.phi)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.theta, self.phi]
        }

    def __str__(self):
        pars = f'(theta={self.theta}, phi={self.phi})'
        return self.name + pars


class GateCRX(Gate):
    """
    Two qubit Controlled-RX gate.

    # Arguments
    :param theta: The rotation angle in radians.
    :type theta: float

    .. math::

        \\operatorname{CRX}(\\theta) = \\begin{pmatrix}
                  1 & 0 & 0 & 0 \\\\
                  0 & 1 & 0 & 0 \\\\
                  0 & 0 & \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
                  0 & 0 & -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
              \\end{pmatrix}

    :return: Two qubit Controlled-RX gate .
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCRX
    
    >>> GateCRX(np.pi/3).matrix()

    array([[ 1.00000000e+00+0.j ,  0.00000000e+00+0.j ,  0.00000000e+00+0.j ,
         0.00000000e+00+0.j ],
       [ 0.00000000e+00+0.j ,  1.00000000e+00+0.j ,  0.00000000e+00+0.j ,
         0.00000000e+00+0.j ],
       [ 0.00000000e+00+0.j ,  0.00000000e+00+0.j ,  8.66025404e-01+0.j ,
        -3.06161700e-17-0.5j],
       [ 0.00000000e+00+0.j ,  0.00000000e+00+0.j ,  3.06161700e-17-0.5j,
         8.66025404e-01+0.j ]])
    
    >>> import numpy as np 
    >>> c=Circuit()
    >>> c.add_gate(GateCRX(np.pi/3),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CRX(theta=1.0471975511965976) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CRX'

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return ctrl(rxmatrix(self.theta))

    def inverse(self):
        return GateCRX(-self.theta)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.theta]
        }

    def __str__(self):
        pars = f'(theta={self.theta})'
        return self.name + pars


class GateCRY(Gate):
    """
    Two qubit Controlled-RY gate.

    # Arguments
    :param theta: The rotation angle in radians.
    :type theta: float

    .. math::

        \\operatorname{CRX}(\\theta) = \\begin{pmatrix}
                  1 & 0 & 0 & 0 \\\\
                  0 & 1 & 0 & 0 \\\\
                  0 & 0 & \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} \\\\
                  0 & 0 &  \\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2}
              \\end{pmatrix}

    :return: Controlled-RY gate .
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCRY
    
    >>> GateCRY(np.pi/2).matrix()

    array([[ 1.        +0.j,  0.        +0.j,  0.        +0.j,
         0.        +0.j],
       [ 0.        +0.j,  1.        +0.j,  0.        +0.j,
         0.        +0.j],
       [ 0.        +0.j,  0.        +0.j,  0.70710678+0.j,
        -0.70710678-0.j],
       [ 0.        +0.j,  0.        +0.j,  0.70710678+0.j,
         0.70710678+0.j]])
    
    >>> import numpy as np 
    >>> c=Circuit()
    >>> c.add_gate(GateCRY(np.pi/2),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CRY(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CRY'

    def __init__(self, theta):
        self.theta = theta

    def matrix(self):
        return ctrl(rymatrix(self.theta))

    def inverse(self):
        return GateCRY(-self.theta)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.theta]
        }

    def __str__(self):
        pars = f'(theta={self.theta})'
        return self.name + pars


class GateCRZ(Gate):
    """
    Two qubit Controlled-RZ gate.

    # Arguments
    :param theta: The rotation angle in radians.
    :type theta: float

    .. math::

        \\operatorname{CRX}(\\theta) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & e^{-i\\frac{\\lambda}{2}} & 0 \\\\
            0 & 0 & 0 & e^{i\\frac{\\lambda}{2}}
              \\end{pmatrix}

    :return: Controlled-RZ gate .
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCRZ
    
    >>> GateCRZ(np.pi/2).matrix()

    array([[1.        +0.j        , 0.        +0.j        ,
        0.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 1.        +0.j        ,
        0.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.        +0.j        ,
        1.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.        +0.j        ,
        0.        +0.j        , 0.24740396+0.96891242j]])
    
    >>> import numpy as np 
    >>> c=Circuit()
    >>> c.add_gate(GateCRZ(np.pi/2),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CRZ(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CRZ'

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return ctrl(rzmatrix(self.lmbda))

    def inverse(self):
        return GateCRZ(-self.lmbda)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.lmbda]
        }

    def __str__(self):
        pars = f'(lmbda={self.lmbda})'
        return self.name + pars


class GateCP(Gate):
    """
    Two qubit Controlled-Phase gate.

    # Arguments
    :param lambda: Phase angle in radians.
    :type lambda: float

    .. math::

        \\operatorname{CRX}(\\theta) = \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & e^{i\\lambda}
              \\end{pmatrix}

    :return: Controlled-Phase gate
    :rtype: numpy.ndarray

    #Examples
    --------
    ```python
    >>> from mimiqcircuits.circuit import Circuit
    >>> from mimiqcircuits.gates import GateCRZ
    
    >>> GateCP(np.pi/4).matrix()

    array([[1.        +0.j        , 0.        +0.j        ,
        0.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 1.        +0.j        ,
        0.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.        +0.j        ,
        1.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.        +0.j        ,
        0.        +0.j        , 0.70710678+0.70710678j]])
    
    >>> import numpy as np 
    >>> c=Circuit()
    >>> c.add_gate(GateCP(np.pi/2),0,1)
    >>> print(c)

     2-qubit circuit with 1 gates
     └── CP(theta=1.5707963267948966) @ q0, q1
    """
    _num_qubits = 2
    _name = 'CP'

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def matrix(self):
        return ctrl(pmatrix(self.lmbda))

    def inverse(self):
        return GateCP(-self.lmbda)

    def to_json(self):
        return {
            'name': self.name,
            'params': [self.lmbda]
        }

    def __str__(self):
        pars = f'(lmbda={self.lmbda})'
        return self.name + pars


GATES = {
    c._name: c for name, c in globals().items()
    if inspect.isclass(c) and hasattr(c, '_name')
}

# export all the gates
__all__ = [name for name in globals() if name.startswith('Gate')]
