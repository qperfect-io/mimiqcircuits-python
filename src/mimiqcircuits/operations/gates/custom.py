#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
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
import symengine as se
import sympy as sp
import numpy as np


class GateCustom(mcg.Gate):
    """One or Two qubit Custom gates.

    Examples:
        >>> from mimiqcircuits import Circuit, GateCustom
        >>> import numpy as np
        >>> matrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, -1j],[0, 0, 1j, 0]])
        >>> c = Circuit()
        >>> c.push(GateCustom(matrix), 0, 1)
        2-qubit circuit with 1 instructions:
        └── Custom(...) @ q[0,1]
        <BLANKLINE>
    """

    _name = "Custom"
    _num_qubits = None
    _qregsizes = None

    def __init__(self, matrix):
        super().__init__()

        if isinstance(matrix, ndarray):
            mat = se.Matrix(matrix.tolist())
        elif isinstance(matrix, se.Matrix):
            mat = matrix
        else:
            raise TypeError(
                f"{type(matrix)} not supported in GateCustom, use symengine.Matrix or numpy.ndarray."
            )

        if mat.rows != mat.cols:
            raise ValueError("Matrix is not square")

        tolerance = 1e-8
        if any(
            isinstance(mat[i, j], (se.Symbol, str))
            for i in range(mat.rows)
            for j in range(mat.cols)
        ):
            pass
        else:
            if not self.is_unitary(mat, tol=tolerance):
                raise ValueError("Matrix is not unitary")

        num_qubits = (mat.rows >> 2) + 1
        if mat.rows != 2**num_qubits:
            raise ValueError("Wrong number of the rows for the matrix")

        self.matrix = mat
        self._num_qubits = num_qubits
        self._qregsizes = [
            num_qubits,
        ]

    @property
    def _matrix(self):
        return self.matrix

    @property
    def num_qubits(self):
        return self._num_qubits

    def inverse(self):
        return GateCustom(self.matrix.inv())

    @staticmethod
    def is_unitary(matrix, tol=1e-8):
        conjugate_transpose = matrix.transpose().conjugate()
        product = matrix * conjugate_transpose
        identity_matrix = se.eye(matrix.rows)

        return all(
            abs(product[i, j] - identity_matrix[i, j]) < tol
            for i in range(matrix.rows)
            for j in range(matrix.cols)
        )

    def __repr__(self):
        return self.pretty_print()

    def __str__(self):
        return f"{self._name}(...)"

    def pretty_print(self):
        if isinstance(self.matrix, se.Matrix):
            U = np.array(self.matrix.tolist(), dtype=object)
        else:
            U = np.array(self.matrix, dtype=object)

        result = f"{self._num_qubits}-qubit GateCustom:\n"
        rows, cols = U.shape
        for i in range(rows):
            if i < rows - 1:
                result += "├── "
            else:
                result += "└── "
            result += " ".join(map(str, U[i, :]))
            if i < rows - 1:
                result += "\n"
        return result

    def evaluate(self, d):
        sympy_matrix = sp.Matrix(self.matrix)
        matrix = sympy_matrix.applyfunc(
            lambda entry: (
                entry.subs(d) if not isinstance(entry, (float, int)) else entry
            )
        )
        evaluated_matrix = se.Matrix(matrix.tolist())
        return GateCustom(evaluated_matrix)


__all__ = ["Custom"]
