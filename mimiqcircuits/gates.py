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
    costheta2 = np.cospi(theta / 2)
    sintheta2 = np.sinpi(theta / 2)
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
    costheta2 = np.cospi(theta / 2)
    sintheta2 = np.sinpi(theta / 2)
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
    Class for the X gate.
    """
    _num_qubits = 1
    _name = 'X'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[0, 1], [1, 0]])


class GateY(Gate):
    """
    Class for the Y gate.
    """
    _num_qubits = 1
    _name = 'Y'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[0, -1j], [1j, 0]])


class GateZ(Gate):
    """
    Class for the Z gate.
    """
    _num_qubits = 1
    _name = 'Z'

    def inverse(self):
        return self

    def matrix(self):
        return np.array([[1, 0], [0, -1]])


class GateH(Gate):
    """
    Class for the H gate.
    """
    _num_qubits = 1
    _name = 'H'

    def inverse(self):
        return self

    def matrix(self):
        return 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])


class GateS(Gate):
    _num_qubits = 1
    _name = 'S'

    def inverse(self):
        return GateSDG()

    def matrix(self):
        # return np.array([[1, 0], [0, 1j]])
        return _decomplex(pmatrixpi(1 / 2))


class GateSDG(Gate):
    _num_qubits = 1
    _name = 'SDG'

    def inverse(self):
        return GateS()

    def matrix(self):
        return _decomplex(pmatrixpi(-1 / 2))


class GateT(Gate):
    _num_qubits = 1
    _name = 'T'

    def inverse(self):
        return GateTDG()

    def matrix(self):
        return _decomplex(pmatrixpi(1 / 4))


class GateTDG(Gate):
    _num_qubits = 1
    _name = 'TDG'

    def inverse(self):
        return GateT()

    def matrix(self):
        return _decomplex(pmatrixpi(-1 / 4))


class GateSX(Gate):
    _num_qubits = 1
    _name = 'SX'

    def inverse(self):
        return GateSXDG()

    def matrix(self):
        return _decomplex(gphasepi(1 / 4) * rxmatrixpi(1 / 4))


class GateSXDG(Gate):
    _num_qubits = 1
    _name = 'SXDG'

    def inverse(self):
        return GateSX()

    def matrix(self):
        return _decomplex(gphase(-np.pi / 4) * rxmatrix(-np.pi / 2))


class GateID(Gate):
    _num_qubits = 1
    _name = 'ID'

    def inverse(self):
        return self

    def matrix(self):
        return _decomplex(umatrixpi(0, 0, 0))


class GateU(Gate):
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
    _num_qubits = 2
    _name = 'CX'

    def matrix(self):
        return ctrl(GateX().matrix())

    def inverse(self):
        return self


class GateCY(Gate):
    _num_qubits = 2
    _name = 'CY'

    def matrix(self):
        return ctrl(GateY().matrix())

    def inverse(self):
        return self


class GateCZ(Gate):
    _num_qubits = 2
    _name = 'CZ'

    def matrix(self):
        return ctrl(GateZ().matrix())

    def inverse(self):
        return self


class GateCH(Gate):
    _num_qubits = 2
    _name = 'CH'

    def matrix(self):
        return ctrl(GateH().matrix())

    def inverse(self):
        return self


class GateSWAP(Gate):
    _num_qubits = 2
    _name = 'SWAP'

    def matrix(self):
        return np.array([
            [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]
        ])

    def inverse(self):
        return self


class GateISWAP(Gate):
    _num_qubits = 2
    _name = 'ISWAP'

    def matrix(self):
        return np.array([
            [1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]
        ])

    def inverse(self):
        return GateISWAPDG()


class GateISWAPDG(Gate):
    _num_qubits = 2
    _name = 'ISWAPDG'

    def matrix(self):
        return np.array([
            [1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]
        ])

    def inverse(self):
        return GateISWAP()


class GateCU(Gate):
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
