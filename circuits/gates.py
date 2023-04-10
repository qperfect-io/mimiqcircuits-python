# Copyright © 2023 University of Strasbourg. All Rights Reserved.
# See AUTHORS.md for the list of authors.


import numpy as np


def _decomplex(m):
    if np.iscomplexobj(m):
        if np.all(np.isreal(m)):
            return np.real(m)
        else:
            return m
    else:
        return m


def ctrl(mat):
    """
    Returns the controlled version of a given 2x2 matrix.

    Args:
        mat (numpy.ndarray): A 2x2 matrix to be controlled.

    Returns:
        numpy.ndarray: A 4x4 controlled matrix.
    """
    dim = mat.shape[0]
    return np.block([[mat, np.zeros((dim, dim))], [np.zeros((dim, dim)), np.identity(dim)]])


# def get_inverse(self):
    #     """
    #     Get the inverse of the gate.

    #     Returns:
    #     Gate: The inverse of the gate.
    #     """
    #     inverse_matrix = np.linalg.inv(self.matrix)
       
    #     return Gate(inverse_matrix, self.num_qubits)

    # def __str__(self):
    #     return f"inverse"


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
    matrix = np.array([[0, 1], [1, 0]])

    def __init__(self):
        super().__init__(self.matrix, 1)

    def __str__(self):
        return 'X'

    def get_matrix(self):
        return _decomplex(self.matrix)

    def inverse(self):
        return self  # The inverse of a CX gate is itself


class GateH(Gate):
    """
    Class for the H gate.
    """
    def __init__(self):
        matrix = np.array([[1, 1],
                           [1, -1]]) / np.sqrt(2)
        super().__init__(matrix, 1)
        
        
    def inverse(self):
        return self
    
    def __str__(self):
        return 'H'
        

class GateCH(Gate):
    """
    Class for the controlled-Hadamard gate.
    """
    matrix = ctrl(GateH().matrix)

    def __init__(self):
        super().__init__(self.matrix, 2)

    def __str__(self):
        return 'CH'

    def reverse(self):
        return self


class GateCX(Gate):
    """
    Class for the controlled-X gate.
    """
    matrix = ctrl(GateX().matrix)
    
    def __init__(self):
        super().__init__(self.matrix, 2)
    
    def __str__(self):
        return 'CX'
    def inverse(self):
        return self  


# class GateCX(Gate):
#     """
#     Class for the Controlled-NOT gate.
#     """
#     matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

#     def __init__(self):
#         super().__init__(self.matrix, 2)

#     def __str__(self):
#         return 'CX'
#     def inverse(self):
#         return self 

 
class GateY(Gate):
    """
    Class for the Pauli Y gate.
    """
    matrix = np.array([[0, -1j], [1j, 0]])

    def __init__(self):
        super().__init__(self.matrix, 1)
    def __str__(self):
        return 'Y'
    def inverse(self):
        return self
    
# class GateCY(Gate):
#     """
#     Class for the controlled-Y (CY) gate.
#     """
#     matrix = np.array([[1, 0, 0, 0],
#                        [0, 1, 0, 0],
#                        [0, 0, 0, -1j],
#                        [0, 0, 1j, 0]])

#     def __init__(self):
#         super().__init__(self.matrix, 2)

#     def __str__(self):
#         return 'CY'
# def inverse(self):
#       return self

class GateCY(Gate):
    """
    Class for the controlled-Y gate.
    """
    matrix = ctrl(GateY().matrix)
    
    def __init__(self):
        super().__init__(self.matrix, 2)

    def __str__(self):
        return 'CY'
    def inverse(self):
        return self


class GateZ(Gate):
    """
    Class for the Pauli Z gate.
    """
    matrix = np.array([[1, 0], [0, -1]])

    def __init__(self):
        super().__init__(self.matrix, 1)
    def __str__(self):
        return 'Z'  
    def inverse(self):
        return self 
    

class GateCZ(Gate):
    """
    Class for the controlled-Z gate.
    """
    matrix = ctrl(GateZ().matrix)
    
    def __init__(self):
        super().__init__(self.matrix, 2)

    def __str__(self):
        return 'CZ'
    def inverse(self):
        return self


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
        self.theta = theta
        self.phi = phi
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateR(-self.theta, self.phi)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-R'
        else:
            return 'R'


class GateRX(Gate):
    """
    Class for the RX gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be a number")
        matrix = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                           [-1j*np.sin(theta/2), np.cos(theta/2)]],
                          dtype=np.complex128)
        super().__init__(matrix, 1)
        self.theta = theta
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateRX(self.theta)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-RX'
        else:
            return 'RX'


class GateRY(Gate):
    """
    Class for the RY gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be a number")
        matrix = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                           [np.sin(theta/2), np.cos(theta/2)]],
                          dtype=np.complex128)
        super().__init__(matrix, 1)
        self.theta = theta
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateRY(-self.theta)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-RY'
        else:
            return 'RY'


class GateRZ(Gate):
    """
    Class for the RZ gate.
    """
    def __init__(self, lmbda):
        if not isinstance(lmbda, (int, float)):
            raise TypeError("phi must be a number")
        matrix = np.array([[np.exp(-1j*lmbda/2), 0],
                           [0, np.exp(1j*lmbda/2)]],
                          dtype=np.complex128)
        super().__init__(matrix, 1)
        self.phi = lmbda
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateRZ(-self.phi)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-RZ'
        else:
            return 'RZ'



class GateCRX(Gate):
    """
    Class for the controlled-RX gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be a number")
        matrix = ctrl(GateRX(theta/2).matrix)
        super().__init__(matrix, 2)
        self.theta = theta
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateCRX(-self.theta)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-CRX'
        else:
            return 'CRX'


class GateCRY(Gate):
    """
    Class for the controlled-RY gate.
    """
    def __init__(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be a number")
        matrix = ctrl(GateRY(theta/2).matrix)
        super().__init__(matrix, 2)
        self.theta = theta
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateCRY(-self.theta)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-CRY'
        else:
            return 'CRY'


class GateCRZ(Gate):
    """
    Class for the controlled-RZ gate.
    """
    def __init__(self, lmbda):
        if not isinstance(lmbda, (int, float)):
            raise TypeError("lmbda must be a number")
        matrix = ctrl(GateRZ(lmbda/2).matrix)
        super().__init__(matrix, 2)
        self.lmbda = lmbda
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateCRZ(-self.lmbda)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-CRZ'
        else:
            return 'CRZ'


class GateP(Gate):
    """
    Class for the P gate, also known as the phase gate.
    """
    def __init__(self, lmbda):
        if not isinstance(lmbda, (int, float)):
            raise TypeError("angle must be a number")
        matrix = np.eye(2, dtype=np.complex128)
        matrix[1, 1] = np.exp(1j * lmbda)
        super().__init__(matrix, 1)
        self.lmbda = lmbda
        self.is_reversed = False

    def inverse(self):
        inverse_gate = GateP(-self.lmbda)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-P'
        else:
            return 'P'

    
class GateCP(Gate):
    """
    Class for the controlled-P gate.
    """
    def __init__(self, lmbda):
        if not isinstance(lmbda, (int, float)):
            raise TypeError("lmbda must be a number")
        matrix = ctrl(GateP(lmbda/2).matrix)
        super().__init__(matrix, 2)
        self.lmbda = lmbda
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateCP(-self.lmbda)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-CP'
        else:
            return 'CP'


# class GateU(Gate):
#     """
#     Class for the U gate.
#     """
#     def __init__(self, theta, phi, lmbda, gamma=0):
#         if not all(isinstance(p, (int, float)) for p in [theta, phi, lmbda, gamma]):
#             raise TypeError("theta, phi, lambda, and gamma must be numbers")
#         matrix = np.array([[np.cos(theta/2), -np.exp(1j*lmbda)*np.sin(theta/2)],
#                            [np.exp(1j*phi)*np.sin(theta/2)*np.cos(gamma), 
#                             np.exp(1j*(phi+lmbda))*np.cos(theta/2)*np.cos(gamma) 
#                             - 1j*np.sin(theta/2)*np.sin(gamma)]],
#                           dtype=np.complex128)
#         super().__init__(matrix, 1)
#         self.theta = theta
#         self.phi = phi
#         self.lmbda = lmbda
#         self.gamma = gamma
#         self.is_reversed = False
        
#     def inverse(self):
#         inverse_gate = GateU(-self.theta, -self.phi, -self.lmbda, -self.gamma)
#         inverse_gate.is_reversed = not self.is_reversed
#         return inverse_gate
    
#     def __str__(self):
#         if self.is_reversed:
#             return '-U'
#         else:
#             return 'U'


class GateU(Gate):
    """
    Class for the U gate.
    """
    def __init__(self, theta, phi, lmbda):
        if not all(isinstance(p, (int, float)) for p in [theta, phi, lmbda]):
            raise TypeError("theta, phi, and lmbda must be numbers")
        matrix = np.array([[np.cos(theta/2), -np.exp(1j*lmbda)*np.sin(theta/2)],
                           [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lmbda))*np.cos(theta/2)]],
                          dtype=np.complex128)
        super().__init__(matrix, 1)
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateU(-self.theta, -self.phi, -self.lmbda)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-U'
        else:
            return 'U'


# class GateCU(Gate):
#     """
#     Class for the controlled-U gate.
#     """
#     def __init__(self, theta, phi, lmbda, gamma):
#         if not all(isinstance(p, (int, float)) for p in [theta, phi, lmbda, gamma]):
#             raise TypeError("theta, phi, lmbda, and gamma must be numbers")
#         matrix = np.eye(4, dtype=np.complex128)
#         matrix[2:, 2:] = np.array([[np.cos(theta/2), -np.exp(1j*lmbda)*np.sin(theta/2)],
#                                    [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lmbda+gamma))*np.cos(theta/2)]],
#                                   dtype=np.complex128)
#         super().__init__(matrix, 2)
#         self.theta = theta
#         self.phi = phi
#         self.lmbda = lmbda
#         self.gamma = gamma
#         self.is_reversed = False
        
#     def inverse(self):
#         inverse_gate = GateCU(-self.theta, -self.phi, -self.lmbda, -self.gamma)
#         inverse_gate.is_reversed = not self.is_reversed
#         return inverse_gate
        
#     def __str__(self):
#         if self.is_reversed:
#             return '-CU'
#         else:
#             return 'CU'


class GateCU(Gate):
    """
    Class for the controlled-U gate.
    """
    def __init__(self, theta, phi, lmbda, gamma):
        if not all(isinstance(p, (int, float)) for p in [theta, phi, lmbda, gamma]):
            raise TypeError("theta, phi, lmbda, and gamma must be numbers")
        matrix = ctrl(GateU(theta, phi, lmbda).matrix)
        matrix[2:, 2:] = GateU(theta, phi, lmbda + gamma).matrix
        super().__init__(matrix, 2)
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda
        self.gamma = gamma
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateCU(-self.theta, -self.phi, -self.lmbda, -self.gamma)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
        
    def __str__(self):
        if self.is_reversed:
            return '-CU'
        else:
            return 'CU'
 

class GateU1(Gate):
    """
    Class for the U1 gate.
    """
    def __init__(self, lmbda):
        if not isinstance(lmbda, (int, float)):
            raise TypeError("lmbda must be a number")
        matrix = np.array([[1, 0], [0, np.exp(1j*lmbda)]], dtype=np.complex128)
        super().__init__(matrix, 1)
        self.lmbda = lmbda
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateU1(-self.lmbda)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-U1'
        else:
            return 'U1'


class GateU2(Gate):
    """
    Class for the U2 gate.
    """
    def __init__(self, phi, lmbda):
        if not all(isinstance(p, (int, float)) for p in [phi, lmbda]):
            raise TypeError("phi and lmbda must be numbers")
        matrix = np.array([[1/np.sqrt(2), -np.exp(1j*lmbda)*1/np.sqrt(2)],
                           [np.exp(1j*phi)*1/np.sqrt(2), np.exp(1j*(phi+lmbda))*1/np.sqrt(2)]],
                          dtype=np.complex128)
        super().__init__(matrix, 1)
        self.phi = phi
        self.lmbda = lmbda
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateU2(-self.phi, -self.lmbda)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-U2'
        else:
            return 'U2'


class GateU3(Gate):
    """
    Class for the U3 gate.
    """
    def __init__(self, theta, phi, lmbda):
        if not all(isinstance(p, (int, float)) for p in [theta, phi, lmbda]):
            raise TypeError("theta, phi, and lmbda must be numbers")
        matrix = np.array([[np.cos(theta/2), -np.exp(1j*lmbda)*np.sin(theta/2)],
                           [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lmbda))*np.cos(theta/2)]],
                          dtype=np.complex128)
        super().__init__(matrix, 1)
        self.theta = theta
        self.phi = phi
        self.lmbda = lmbda
        self.is_reversed = False
        
    def inverse(self):
        inverse_gate = GateU3(-self.theta, -self.phi, -self.lmbda)
        inverse_gate.is_reversed = not self.is_reversed
        return inverse_gate
    
    def __str__(self):
        if self.is_reversed:
            return '-U3'
        else:
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
    
class GateSWAP(Gate):
    """
    Class for the SWAP gate.
    """
    matrix = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])

    def __init__(self):
        super().__init__(self.matrix, 2)

    def __str__(self):
        return 'SWAP'

    def inverse(self):
        return self
    
    
class GateISWAP(Gate):
    """
    Class for the iSWAP gate.
    """
    _matrix = np.array([[1, 0, 0, 0],
                        [0, 0, 1j, 0],
                        [0, 1j, 0, 0],
                        [0, 0, 0, 1]], dtype=np.complex128)

    def __init__(self):
        super().__init__(self._matrix, 2)

    def __str__(self):
        return 'ISWAP'

    def inverse(self):
        return GateISWAPDG()


class GateISWAPDG(Gate):
    """
    Class for the inverse of iSWAP gate.
    """
    _matrix = np.array([[1, 0, 0, 0],
                        [0, 0, -1j, 0],
                        [0, -1j, 0, 0],
                        [0, 0, 0, 1]], dtype=np.complex128)

    def __init__(self):
        super().__init__(self._matrix, 2)

    def __str__(self):
        return 'ISWAPDG'

    def inverse(self):
        return GateISWAP()