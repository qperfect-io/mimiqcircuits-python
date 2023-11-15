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
from numpy import ndarray
from numpy.linalg import inv
from symengine import *

import sympy as sp


class GateCustom(mcg.Gate):
    """One or Two qubit Custom gates.

Examples:
    >>> from mimiqcircuits import Circuit, GateCustom
    >>> import numpy as np
    >>> matrix = np.array([[1, 0, 0, 0],
    ...                    [0, 1, 1j, 0],
    ...                    [0, 0, 0, 1],
    ...                    [0, 0, 1, 0]])
    >>> c = Circuit()
    >>> c.push(GateCustom(matrix), 0, 1)
    2-qubit circuit with 1 instructions:
    └── Custom([[1.0 + 0.0*I, 0.0 + 0.0*I, 0.0 + 0.0*I, 0.0 + 0.0*I],
                [0.0 + 0.0*I, 1.0 + 0.0*I, 0.0 + 1.0*I, 0.0 + 0.0*I],
                [0.0 + 0.0*I, 0.0 + 0.0*I, 0.0 + 0.0*I, 1.0 + 0.0*I],
                [0.0 + 0.0*I, 0.0 + 0.0*I, 1.0 + 0.0*I, 0.0 + 0.0*I]]) @ q0, q1
"""
    _name = 'Custom'
    _num_qubits = None
    _qregsizes = None


    def __init__(self, matrix):
        super().__init__()
        
        if isinstance(matrix, ndarray):
            mat = Matrix(matrix.tolist())
        elif isinstance(matrix, Matrix):
            mat = matrix
        else:
            raise TypeError(f"{type(matrix)} not supported in GateCustom, use symengine.Matrix or numpy.ndarray.")
        
        if mat.rows != mat.cols:
            raise ValueError("Matrix is not square")
        
        num_qubits = int(log(mat.rows, 2))

        
        self._matrix = mat
        self._num_qubits = num_qubits
        self._qregsizes = [num_qubits,]

    @property
    def matrix(self):
        return self._matrix

    @property
    def num_qubits(self):
        return self._num_qubits

    def inverse(self):
        return GateCustom(self.matrix.inv())
    
    def __str__(self):
        return f'{self.name}({self.matrix.tolist()})'

    def evaluate(self, d):
        sympy_matrix = sp.Matrix(self.matrix)
        matrix = sympy_matrix.applyfunc(lambda entry: entry.subs(d) if not isinstance(entry, (float, int)) else entry)
        evaluated_matrix = Matrix(matrix.tolist())
        return GateCustom(evaluated_matrix)

        
__all__ = ['Custom']
