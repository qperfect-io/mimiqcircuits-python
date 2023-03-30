#
# Copyright © 2023 University of Strasbourg. All Rights Reserved.
# See AUTHORS.md for the list of authors.
#

import numpy as np

class Gate:
    """
    Base class for quantum gates.
    """
    def __init__(self, matrix, num_qubits):
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise TypeError("matrix must be a 2D numpy array")
        expected_dim = 2**num_qubits
        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"matrix must be of shape ({expected_dim}, {expected_dim})")
        self._num_qubits = num_qubits
        self._matrix = matrix


    @property
    def num_qubits(self):
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, _):
        raise AttributeError("num_qubits is a read-only attribute")

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, _):
        raise AttributeError("matrix is a read-only attribute")


class GateX(Gate):
    """
    Class for the Pauli X gate.
    """
    _gatematrix = np.array([[0, 1], [1, 0]])

    def __init__(self):
        super().__init__(self._gatematrix, 1)


class GateRX(Gate):
    """
    Class for the rotation around the X axis gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*np.array([[0, 1], [1, 0]])
        super().__init__(matrix, 1)
