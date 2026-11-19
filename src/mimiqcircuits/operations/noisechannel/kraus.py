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

import numpy as np
from typing import List, Union
import symengine as se
from mimiqcircuits._sympycheck import _is_sympy_matrix
from mimiqcircuits.operations.operator import AbstractOperator
from mimiqcircuits.operations.operators.custom import Operator
from mimiqcircuits.operations.operators.lossyoperator import LossyOperator
from mimiqcircuits.operations.krauschannel import krauschannel


def is_valid_power_of_2(n):
    """Check if a number is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _is_approx_identity(matrix, size, rtol=1e-12, atol=1e-12):
    """Whether ``matrix`` is close to the identity in norm, as Julia's ``isapprox``."""
    m = np.asarray(matrix, dtype=np.complex128)
    diff = np.linalg.norm(m - np.eye(size))
    return diff <= max(atol, rtol * max(np.linalg.norm(m), np.sqrt(size)))


class Kraus(krauschannel):
    r"""Custom N-qubit Kraus channel given by a list of Kraus operators.

    The channel acts on a density matrix as

    .. math::
        \mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger,

    where the Kraus operators :math:`E_k` must fulfill
    :math:`\sum_k E_k^\dagger E_k = I`. If all Kraus operators are proportional
    to unitaries, :class:`MixedUnitary` is the better choice, since it can be
    sampled as a random unitary.

    Matrices are written in the computational basis in the usual textbook order,
    with the first qubit as the left-most one: :math:`|0\rangle, |1\rangle` for
    one qubit, :math:`|00\rangle, |01\rangle, |10\rangle, |11\rangle` for two.

    A `Kraus` channel models qubit loss when one or more of its operators is a
    :class:`LossyOperator`. The entries of a :class:`LossyOperator` are
    amplitudes, not probabilities: :meth:`losseffect` returns the loss
    probabilities as :math:`\sum_k L_k^\dagger L_k` over the lossy branches
    :math:`L_k`. :meth:`hasloss`, :meth:`lossoperators` and
    :meth:`survivaloperators` split the channel into lossy and surviving
    branches.

    Args:
        E: Kraus operators, each a :math:`2^N \times 2^N` matrix (symengine,
            sympy or numpy) or an operator such as :class:`Operator`,
            :class:`Projector0` or :class:`LossyOperator`. Matrices are wrapped
            in :class:`Operator`. All entries must act on the same number of
            qubits :math:`N`.
        rtol: Tolerance of the check :math:`\sum_k E_k^\dagger E_k = I`
            (default ``1e-12``). The check is skipped when any entry is
            symbolic.

    Raises:
        ValueError: If ``E`` is empty, the operators have different sizes, the
            size is not a power of 2, :math:`N` is not 1 or 2, or the numeric
            operators do not fulfill :math:`\sum_k E_k^\dagger E_k = I`.
        TypeError: If an entry of ``E`` is neither a matrix nor an operator.

    Note:
        Only 1- and 2-qubit custom Kraus channels are supported.

    See Also:
        :class:`MixedUnitary`, :class:`GateCustom`, :class:`LossyOperator`,
        :class:`Operator`

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> # amplitude damping with gamma = 0.1, written out by hand
        >>> c = Circuit()
        >>> c.push(Kraus([Matrix([[1, 0], [0, sqrt(0.9)]]), Matrix([[0, sqrt(0.1)], [0, 0]])]), 0)
        1-qubit circuit with 1 instruction:
        └── Kraus(Operator([[1, 0], [0, 0.948683298050514]]), Operator([[0, 0.316227766016838], [0, 0]])) @ q[0]
        <BLANKLINE>

        >>> # the Kraus matrices can be read back
        >>> g = Kraus([Matrix([[1, 0], [0, sqrt(0.9)]]), Matrix([[0, sqrt(0.1)], [0, 0]])])
        >>> g.krausmatrices()
        [[1.0, 0]
        [0, 0.948683298050514]
        , [0, 0.316227766016838]
        [0, 0]
        ]

        >>> # operators can be used directly: here a Z-basis dephasing
        >>> c = Circuit()
        >>> c.push(Kraus([Projector0(), Projector1()]), 0)
        1-qubit circuit with 1 instruction:
        └── Kraus(P₀(1), P₁(1)) @ q[0]
        <BLANKLINE>

        >>> # matrices and operators can be mixed
        >>> c = Circuit()
        >>> c.push(Kraus([Matrix([[1, 0], [0, 0]]), Projector1()]), 0)
        1-qubit circuit with 1 instruction:
        └── Kraus(Operator([[1, 0], [0, 0]]), P₁(1)) @ q[0]
        <BLANKLINE>

        >>> # a lossy branch: the qubit is lost from |1> with probability 0.2
        >>> lossy = LossyOperator(Matrix([[0, 0], [0, sqrt(0.2)]]))
        >>> survival = Operator(Matrix([[1, 0], [0, sqrt(0.8)]]))
        >>> lk = Kraus([survival, lossy])
        >>> lk.hasloss()
        True
        >>> lk.lossoperators()
        [1-qubit LossyOperator (lossy=(1,)):
        ├── 0 0
        └── 0 0.447213595499958]
        >>> lk.survivaloperators()
        [1-qubit Operator:
        ├── 1 0
        └── 0 0.894427190999916]
        >>> lk.losseffect()
        1-qubit Operator:
        ├── 0.0 + 0.0*I 0.0 + 0.0*I
        └── 0.0 + 0.0*I 0.2 + 0.0*I

        >>> # symbolic parameters are substituted with evaluate
        >>> x = Symbol("x")
        >>> g = Kraus([Projector0(), Projector1(x)])
        >>> g
        Kraus(P₀(1), P₁(x))
        >>> g.evaluate({x: 1})
        Kraus(P₀(1), P₁(1))
    """

    _name = "Kraus"
    _parnames = ()

    def __init__(
        self,
        E: List[Union[se.Matrix, np.ndarray, "sympy.Matrix", AbstractOperator]],
        rtol=1e-12,
    ):
        if len(E) == 0:
            raise ValueError("Vector of Kraus operators cannot be empty")

        Es = []
        for Ek in E:
            if isinstance(Ek, AbstractOperator):
                Es.append(Ek)
            elif isinstance(Ek, (se.Matrix, np.ndarray)) or _is_sympy_matrix(Ek):
                Es.append(Operator(self._convert_to_matrix(Ek)))
            else:
                raise TypeError(f"Unsupported operator type: {type(Ek)}")

        # Determine the dimension of the first operator
        dim = self._determine_dim(Es[0])

        # Ensure all matrices or operators have the same dimension
        for Ek in Es:
            matrix_dim = Ek.matrix().rows
            if matrix_dim != dim:
                raise ValueError("All matrices must have the same dimensions.")

        if not is_valid_power_of_2(dim):
            raise ValueError("Operator dimension must be 2^n.")

        self.N = Es[0].num_qubits
        if self.N < 1:
            raise ValueError("Cannot define a 0-qubit custom noise channel")
        if self.N > 2:
            raise ValueError(
                "Custom noise channels larger than 2 qubits are not supported"
            )

        # Perform the sum of E_k^† E_k = I check directly here
        M = 1 << self.N  # Equivalent to 2^N

        if not any(any(not element.is_number for element in op.matrix()) for op in Es):
            # Perform normalization check only for purely numeric matrices
            ksum = sum(
                np.array(op.matrix().tolist(), dtype=np.complex128).conj().T
                @ np.array(op.matrix().tolist(), dtype=np.complex128)
                for op in Es
            )
            if not _is_approx_identity(ksum, M, rtol):
                raise ValueError(
                    "List of Kraus matrices should fulfill \\sum_k E_k^\\dagger E_k = I."
                )

        self.E = Es

        super().__init__()
        self._num_qubits = self.N
        self._parnames = ("E",)

    def evaluate(self, values: dict):
        """Return a copy with symbols substituted by values.

        The normalization check runs again on the result when it is fully
        numeric, with the default tolerance.

        Args:
            values: Map from symbols to the values to substitute.

        Returns:
            Kraus: A new channel with substituted operators.

        Raises:
            ValueError: If the substituted operators do not fulfill
                :math:`\\sum_k E_k^\\dagger E_k = I`.
        """
        evaluated_E = []
        for op in self.E:
            if isinstance(op, Operator):
                # Substitute symbolic parameters in the matrix
                matrix = op.matrix()
                evaluated_matrix = se.Matrix(
                    [
                        [element.subs(values) for element in row]
                        for row in matrix.tolist()
                    ]
                )
                evaluated_E.append(Operator(evaluated_matrix))
            else:
                # If it's already an AbstractOperator, we just append it
                evaluated_E.append(op.evaluate(values))

        return Kraus(evaluated_E)

    def _convert_to_matrix(self, operator):
        """Converts matrices to symengine.Matrix, stripping imaginary parts if zero."""
        if isinstance(operator, np.ndarray) or _is_sympy_matrix(operator):
            matrix = se.Matrix(operator.tolist())
        elif isinstance(operator, se.Matrix):
            matrix = operator
        else:
            raise TypeError(f"Unsupported matrix type: {type(operator)}")

        return matrix

    def _determine_dim(self, operator):
        """Determines the dimension of the operator or matrix."""
        return 1 << operator.num_qubits

    def krausoperators(self):
        """Return the Kraus operators, in the order they were given."""
        return self.E

    def hasloss(self):
        """Whether any Kraus operator is a :class:`LossyOperator`.

        Same result as :meth:`haslossybranch`.
        """
        return any(isinstance(op, LossyOperator) for op in self.E)

    def lossoperators(self):
        """Return the Kraus operators that are :class:`LossyOperator`, in order."""
        return [op for op in self.E if isinstance(op, LossyOperator)]

    def survivaloperators(self):
        """Return the Kraus operators that are not :class:`LossyOperator`, in order."""
        return [op for op in self.E if not isinstance(op, LossyOperator)]

    def losseffect(self, tol=1e-12):
        r"""Compute the loss-probability operator over all lossy branches.

        This returns the operator

        .. math::
            \Lambda = \sum_k L_k^\dagger L_k,

        where :math:`L_k` runs over the :class:`LossyOperator` branches of the
        channel.

        :math:`\Lambda` is a positive semidefinite operator whose diagonal
        entries give the loss probabilities for each computational basis state.
        For a basis state :math:`|i\rangle`,

        .. math::
            \langle i|\Lambda|i\rangle

        is the probability that :math:`|i\rangle` leaks out of the
        computational subspace. For a general state :math:`|\psi\rangle`, the
        total loss probability is :math:`\langle\psi|\Lambda|\psi\rangle`.

        The survival and lossy operators together satisfy

        .. math::
            \sum_k S_k^\dagger S_k + \Lambda = I,

        where :math:`S_k` are the non-lossy branches.

        If the channel has no lossy operators, the zero matrix is returned.

        Args:
            tol: Entries with absolute value below ``tol`` are set to zero
                (default ``1e-12``).

        Returns:
            Operator: :math:`\Lambda` as a numeric :class:`Operator`.

        Raises:
            RuntimeError: If a lossy operator has symbolic entries.

        See Also:
            :class:`Kraus`, :meth:`lossoperators`, :meth:`survivaloperators`

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import Matrix, sqrt
            >>> g = Kraus([Matrix([[1, 0], [0, sqrt(0.9)]]),
            ...            LossyOperator(Matrix([[0, 0], [0, sqrt(0.1)]]))])
            >>> g.losseffect()
            1-qubit Operator:
            ├── 0.0 + 0.0*I 0.0 + 0.0*I
            └── 0.0 + 0.0*I 0.1 + 0.0*I
        """
        M = 1 << self.N
        loss_ops = self.lossoperators()
        if not loss_ops:
            return Operator(np.zeros((M, M), dtype=np.complex128))

        effect = np.zeros((M, M), dtype=np.complex128)
        for op in loss_ops:
            effect += np.array(op.opsquared().matrix().tolist(), dtype=np.complex128)

        effect[np.abs(effect) < tol] = 0
        return Operator(effect)

    def __str__(self):
        """String representation, listing either AbstractOperators or matrix slices."""
        matrices_str = ", ".join(
            [
                (
                    f"[{matrix[0, 0]}, ... ,{matrix[-1, -1]}]"
                    if isinstance(matrix, se.Matrix)
                    else str(matrix)
                )
                for matrix in self.E
            ]
        )
        return f"{self._name}({matrices_str})"

    def unwrappedkrausmatrices(self):
        """Return the Kraus matrices; same as :meth:`krausmatrices`."""
        return self.krausmatrices()


__all__ = ["Kraus"]
