#
# Copyright © 2022-2025 University of Strasbourg. All Rights Reserved.
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


def _normalize_lossy(lossy):
    if isinstance(lossy, int):
        return (int(lossy),)
    if isinstance(lossy, tuple):
        return tuple(int(q) for q in lossy)
    if isinstance(lossy, list):
        return tuple(int(q) for q in lossy)
    raise TypeError("lossy must be an int, tuple, or list of integers")


class LossyOperator(mc.AbstractOperator):
    r"""N-qubit operator specified by a :math:`2^N \times 2^N` matrix and a
    set of lossy qubit indices.

    .. note::
        Only one and two qubit lossy operators are supported.

    A ``LossyOperator`` represents a tagged custom operator used as a loss
    branch inside a :class:`Kraus` channel. The matrix is expressed in the
    computational basis, exactly like :class:`Operator`, while ``lossy`` marks
    which of the operator's local qubits leak when this branch is selected.

    The ``lossy`` indices are local to the operator and use ``1:N`` numbering,
    following the Julia implementation. They are not circuit-global qubit
    indices. When a surrounding :class:`Kraus` channel is applied to circuit
    targets ``(q_1, ..., q_N)``, each local index ``i`` in ``lossy`` maps to
    the corresponding physical target ``q_i``.

    A :class:`Kraus` channel that contains one or more ``LossyOperator``
    branches is a loss-aware channel. The helpers :meth:`Kraus.hasloss`,
    :meth:`Kraus.lossoperators`, :meth:`Kraus.survivaloperators`, and
    :meth:`Kraus.losseffect` inspect those branches.

    See Also:
        :class:`AbstractOperator`, :class:`Operator`, :class:`Kraus`

    Parameters:
        mat (symengine.Matrix | sympy.Matrix | np.ndarray): The :math:`2^N \times 2^N` matrix
            representing the operator.
        lossy (int | tuple | list, optional): One or more local qubit indices
            in ``1:N`` that leak when this branch is selected. For one-qubit
            operators, the default is ``(1,)``. For multi-qubit operators, the
            lossy qubits must be specified explicitly.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import Matrix, sqrt
        >>> op = LossyOperator(Matrix([[0, 1], [0, 0]]))
        >>> op
        1-qubit LossyOperator (lossy=(1,)):
        ├── 0 1
        └── 0 0

        >>> c = Circuit()
        >>> c.push(Kraus([Matrix([[1, 0], [0, sqrt(0.9)]]),
        ...               LossyOperator(Matrix([[0, sqrt(0.1)], [0, 0]]))]), 1)
        2-qubit circuit with 1 instruction:
        └── Kraus(Operator([[1, 0], [0, 0.948683298050514]]), LossyOperator([[0, 0.316227766016838], [0, 0]]; lossy=(1,))) @ q[1]
        <BLANKLINE>
    """

    _name = "LossyOperator"
    _num_qubits = None
    _parnames = ("mat",)
    _num_qregs = 1

    def __init__(self, mat, lossy=None):
        if not isinstance(mat, (se.Matrix, np.ndarray, sp.Matrix)):
            raise TypeError("Must be a symengine, sympy, or numpy Matrix")

        mat = self._convert_to_symengine_matrix(mat)
        self.mat = mat

        dim_rows = mat.rows
        dim_cols = mat.cols

        if dim_rows != dim_cols:
            raise ValueError(f"Must be a square matrix, but got {dim_rows}x{dim_cols}")

        if not self.is_valid_power_of_2(dim_rows):
            raise ValueError("Dimension of LossyOperator has to be 2^N with N >= 1.")

        self.N = int(np.log2(dim_rows))
        if self.N < 1:
            raise ValueError("Cannot define a 0-qubit lossy operator")
        if self.N > 2:
            raise ValueError("Lossy operators larger than 2 qubits are not supported")

        if lossy is None:
            if self.N != 1:
                raise ValueError(
                    f"LossyOperator on {self.N} qubits requires explicit lossy qubit indices."
                )
            normalized = (1,)
        else:
            normalized = _normalize_lossy(lossy)

        if len(normalized) == 0:
            raise ValueError("LossyOperator must mark at least one lossy qubit.")
        if any(q < 1 or q > self.N for q in normalized):
            raise ValueError(
                f"Lossy qubit indices must be in 1:{self.N}, got {normalized}."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"Lossy qubit indices must be unique, got {normalized}."
            )

        self.lossy = tuple(sorted(normalized))
        super().__init__()
        self._num_qubits = self.N
        self._qregsizes = [self._num_qubits]

    @staticmethod
    def is_valid_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0

    def _convert_to_symengine_matrix(self, matrix):
        if isinstance(matrix, (np.ndarray, sp.Matrix)):
            return se.Matrix(matrix.tolist())
        return matrix

    def opname(self):
        return self._name

    def _matrix(self):
        return self.mat

    def iswrapper(self):
        pass

    @property
    def parnames(self):
        return self._parnames

    def __repr__(self):
        return self.pretty_print()

    def lossyqubits(self):
        return self.lossy

    def lossytargets(self, *targets):
        if len(targets) == 1 and isinstance(targets[0], tuple):
            targets = targets[0]
        if len(targets) != self.N:
            raise ValueError(
                f"Expected {self.N} targets for a {self.N}-qubit LossyOperator, got {len(targets)}."
            )
        return tuple(targets[q - 1] for q in self.lossy)

    def unwrappedmatrix(self):
        return self.mat

    def evaluate(self, values):
        evaluated = se.Matrix(
            [[element.subs(values) for element in row] for row in self.mat.tolist()]
        )
        return LossyOperator(evaluated, self.lossy)

    def opsquared(self):
        return mc.Operator(self.mat.T.conjugate() * self.mat)

    def rescale(self, scale):
        return LossyOperator(self.mat * scale, self.lossy)

    def rescale_inplace(self, scale):
        self.mat *= scale
        return self

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        if self.lossy != other.lossy:
            return False
        if self.mat.rows != other.mat.rows or self.mat.cols != other.mat.cols:
            return False

        for i in range(self.mat.rows):
            for j in range(self.mat.cols):
                if self.mat[i, j] != other.mat[i, j]:
                    return False
        return True

    def __str__(self):
        if self.N < 2:
            return f"{self.opname()}({se.Matrix(self.mat).tolist()}; lossy={self.lossy})"
        return f"{self.opname()}(...; lossy={self.lossy})"

    def pretty_print(self):
        U = np.array(self.mat.tolist(), dtype=object)
        result = f"{self.N}-qubit LossyOperator (lossy={self.lossy}):\n"

        rows, _ = U.shape
        for i in range(rows):
            result += "├── " if i < rows - 1 else "└── "
            result += " ".join(map(str, U[i, :]))
            if i < rows - 1:
                result += "\n"
        return result


__all__ = ["LossyOperator"]
