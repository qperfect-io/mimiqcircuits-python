#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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

import mimiqcircuits as mc
import symengine as se
import numpy as np
import sympy as sp


class Operator(mc.AbstractOperator):
    r"""N-qubit operator specified by a :math:`2^N \times 2^N` matrix.

    .. note::
        Only one and two qubit operators are supported.

    This operator does not have to be unitary.

    See Also:
        :class:`AbstractOperator`, :class:`ExpectationValue`,
        :class:`KrausChannel`

    Parameters:
        matrix (list or np.ndarray): The :math:`2^N \times 2^N` matrix representing the operator.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import Matrix
        >>> op = Operator(Matrix([[1, 2], [3, 4]]))
        >>> op
        1-qubit Operator:
        ├── 1 2
        └── 3 4
        >>> op = Operator(Matrix([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]))
        >>> op
        2-qubit Operator:
        ├── 1 0 0 1
        ├── 0 0 0 0
        ├── 0 0 0 0
        └── 1 0 0 1

    Operators can be used for expectation values:

        >>> c = Circuit()
        >>> c.push(ExpectationValue(Operator(Matrix([[0, 1], [0, 0]]))), 1, 1)
        2-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨Operator([[0, 1], [0, 0]])⟩ @ q[1], z[1]
        <BLANKLINE>
    """

    _name = "Operator"
    _num_qubits = None
    _parnames = ("mat",)
    _num_qregs = 1

    def __init__(self, mat):
        # Check if the input is a valid matrix type (symengine, numpy, or sympy)
        if not isinstance(mat, (se.Matrix, np.ndarray, sp.Matrix)):
            raise TypeError("Must be a symengine, sympy, or numpy Matrix")

        # Convert any numpy or sympy matrices to symengine.Matrix
        mat = self._convert_to_symengine_matrix(mat)
        self.mat = mat

        # Get the matrix dimensions using symengine.Matrix properties
        dim_rows = mat.rows
        dim_cols = mat.cols

        # Check if the matrix is square
        if dim_rows != dim_cols:
            raise ValueError(f"Must be a square matrix, but got {dim_rows}x{dim_cols}")

        # Ensure the dimension is a power of 2
        if not self.is_valid_power_of_2(dim_rows):
            raise ValueError("Dimension of operator has to be 2^n with n>=1")

        # Determine the number of qubits from the matrix dimension
        self.N = int(np.log2(dim_rows))
        if self.N < 1:
            raise ValueError("Cannot define a 0-qubit operator")
        if self.N > 2:
            raise ValueError("Operators larger than 2 qubits are not supported")

        super().__init__()
        self._num_qubits = self.N
        self._qregsizes = [self._num_qubits]

    @staticmethod
    def is_valid_power_of_2(n):
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def _convert_to_symengine_matrix(self, matrix):
        """Helper function to convert numpy or sympy matrices to symengine.Matrix."""
        if isinstance(matrix, (np.ndarray, sp.Matrix)):
            return se.Matrix(matrix.tolist())
        return matrix

    def opname(self):
        """Return the operator name."""
        return "Operator"

    def _matrix(self):
        return self.mat

    @property
    def parnames(self):
        """Return the parameter names."""
        return self._parnames

    def iswrapper(self):
        """Placeholder method."""
        pass

    def unwrappedmatrix(self):
        """Return the matrix unwrapped (if needed)."""
        return self.mat

    def __repr__(self):
        """Return a pretty-printed representation."""
        return self.pretty_print()

    def __str__(self):
        """String representation for printing."""
        if self.N < 2:
            return f"{self.opname()}({se.Matrix(self.mat).tolist()})"
        else:
            return f"{self.opname()}(...)"

    def pretty_print(self):
        """Return a detailed string representation of the operator."""
        U = np.array(self.mat.tolist(), dtype=object)
        result = f"{self.N}-qubit Operator:\n"

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

    def opsquared(self):
        """Return the operator squared (O' * O)."""
        O_dagger = self.mat.T.conjugate()
        squared_matrix = O_dagger * self.mat
        return Operator(squared_matrix)

    def rescale(self, scale):
        """Return a new rescaled Operator by scaling the matrix."""
        scaled_matrix = self.mat * scale
        return Operator(scaled_matrix)

    def rescale_inplace(self, scale):
        """In-place rescaling of the operator matrix."""
        self.mat *= scale
        return self


__all__ = ["Operator"]
