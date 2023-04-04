# Copyright © 2023 University of Strasbourg. All Rights Reserved.
# See AUTHORS.md for the list of authors.


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
    def __str__(self):
        return 'X'

class GateY(Gate):
    """
    Class for the Pauli Y gate.
    """
    _gatematrix = np.array([[0, -1j], [1j, 0]])

    def __init__(self):
        super().__init__(self._gatematrix, 1)
    def __str__(self):
        return 'Y'

class GateZ(Gate):
    """
    Class for the Pauli Z gate.
    """
    _gatematrix = np.array([[1, 0], [0, -1]])

    def __init__(self):
        super().__init__(self._gatematrix, 1)
    def __str__(self):
        return 'Z'    
       
class GateR(Gate):
    """
    Class for the R gate.
    """
    def __init__(self, theta, phi):
        if not isinstance(theta, (int, float)) or not isinstance(phi, (int, float)):
            raise TypeError("theta and phi must be numbers")
        matrix = np.array([[np.cos(theta/2), -np.exp(1j*phi)*np.sin(theta/2)],
                           [np.exp(1j*phi)*np.sin(theta/2), np.cos(theta/2)]],
                          dtype=np.complex128)
        super().__init__(matrix, 1)
    def __str__(self):
        return 'R'
        

class GateRX(Gate):
    """
    Class for the rotation around the X axis gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*np.array([[0, 1], [1, 0]])
        super().__init__(matrix, 1)
    def __str__(self):
        return 'RX'

class GateRY(Gate):
    """
    Class for the rotation around the Y axis gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*np.array([[0, -1], [1, 0]])
        super().__init__(matrix, 1)
    def __str__(self):
        return 'RY'

class GateRZ(Gate):
    """
    Class for the rotation around the Z axis gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.exp(-1j*theta/2)*np.array([[1, 0], [0, -1]])
        super().__init__(matrix, 1)
    def __str__(self):
        return 'RZ'

class GateH(Gate):
    """
    Class for the Hadamard gate.
    """
    _gatematrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def __init__(self):
        super().__init__(self._gatematrix, 1)
    def __str__(self):
        return 'H'

class GateCRX(Gate):
    """
    Class for the controlled rotation around the X axis gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.eye(4, dtype=np.complex128)
        matrix[1, 1] = np.cos(theta / 2)
        matrix[2, 2] = np.cos(theta / 2)
        matrix[1, 2] = -1j * np.sin(theta / 2)
        matrix[2, 1] = -1j * np.sin(theta / 2)
        super().__init__(matrix, 2)
    def __str__(self):
        return 'CRX'

class GateCRY(Gate):
    """
    Class for the controlled rotation around the Y axis gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.eye(4, dtype=np.complex128)
        matrix[1, 1] = np.cos(theta / 2)
        matrix[3, 3] = np.cos(theta / 2)
        matrix[1, 3] = -1j * np.sin(theta / 2)
        matrix[3, 1] = 1j * np.sin(theta / 2)
        super().__init__(matrix, 2)
    def __str__(self):
        return 'CRY'

class GateCRZ(Gate):
    """
    Class for the controlled rotation around the Z axis gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.eye(4, dtype=np.complex128)
        matrix[1, 1] = np.exp(-1j * theta / 2)
        matrix[2, 2] = np.exp(1j * theta / 2)
        super().__init__(matrix, 2)
    def __str__(self):
        return 'CRZ'

class GateP(Gate):
    """
    Class for the P gate, also known as the phase gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.eye(2, dtype=np.complex128)
        matrix[1, 1] = np.exp(1j * theta)
        super().__init__(matrix, 1)
    def __str__(self):
        return 'P'

class GateCP(Gate):
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be a number")
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, np.exp(1j*theta)]])
        super().__init__(matrix, 2)
    def __str__(self):
        return 'CP'

class GateU(Gate):
    """
    Class for the U gate.
    """
    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

        matrix = np.array([
            [np.exp(1j * (self.theta + self.lmbda) / 2) * np.cos(self.phi / 2),
             -np.exp(1j * (self.theta - self.lmbda) / 2) * np.sin(self.phi / 2)],
            [np.exp(1j * (self.theta + self.lmbda) / 2) * np.sin(self.phi / 2),
             np.exp(1j * (self.theta - self.lmbda) / 2) * np.cos(self.phi / 2)]
        ])
        super().__init__(matrix, 2)
    def __str__(self):
        return 'U'

class GateCU(Gate):
    """
    Class for the controlled-U gate.
    """
    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(1j * (self.theta + self.lmbda) / 2) * np.cos(self.phi / 2),
             -np.exp(1j * (self.theta - self.lmbda) / 2) * np.sin(self.phi / 2)],
            [0, 0, np.exp(1j * (self.theta + self.lmbda) / 2) * np.sin(self.phi / 2),
             np.exp(1j * (self.theta - self.lmbda) / 2) * np.cos(self.phi / 2)]
        ])
        super().__init__(matrix, 2)
    def __str__(self):
        return 'CU'

class GateU1(Gate):
    """
    Class for the U1 gate.
    """
    def __init__(self, lmbda):
        self.lmbda = lmbda

        matrix = np.array([
            [1, 0],
            [0, np.exp(1j * self.lmbda)]
        ])
        super().__init__(matrix, 1)
    def __str__(self):
        return 'U1'

class GateU2(Gate):
    """
    Class for the U2 gate.
    """
    def __init__(self, phi, lmbda):
        self.phi = phi
        self.lmbda = lmbda

        matrix = np.array([
            [1/np.sqrt(2), -np.exp(1j * self.lmbda) * 1/np.sqrt(2)],
            [np.exp(1j * self.phi) * 1/np.sqrt(2), np.exp(1j * (self.phi + self.lmbda)) * 1/np.sqrt(2)]
        ])
        super().__init__(matrix, 1)
    def __str__(self):
        return 'U2'   

class GateU3(Gate):
    """
    Class for the U3 gate.
    """
    def __init__(self, theta, phi, lmbda):
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda

        matrix = np.array([
            [np.cos(self.theta/2), -np.exp(1j*self.lmbda)*np.sin(self.theta/2)],
            [np.exp(1j*self.phi)*np.sin(self.theta/2), np.exp(1j*(self.phi+self.lmbda))*np.cos(self.theta/2)]
        ])

        super().__init__(matrix, 1)
    def __str__(self):
        return 'U3'

class GateCustom(Gate):
    """
    Class for a custom quantum gate with a user-defined matrix.
    """
    def __init__(self, matrix):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("matrix must be a NumPy array")
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square array")
        num_qubits = int(np.log2(matrix.shape[0]))
        super().__init__(matrix, num_qubits)
    def __str__(self):
        return 'Custom'
    
