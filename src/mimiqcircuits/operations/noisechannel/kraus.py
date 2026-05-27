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
import mimiqcircuits as mc
import sympy as sp
from mimiqcircuits.operations.krauschannel import krauschannel


def is_valid_power_of_2(n):
    """Check if a number is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _is_approx_identity(matrix, size, rtol=1e-12, atol=1e-12):
    """Check if the matrix is approximately the identity matrix using relative and absolute tolerance."""
    identity_matrix = se.eye(size)
    for i in range(size):
        for j in range(size):
            diff = abs(matrix[i, j] - identity_matrix[i, j])
            # Use both relative tolerance (rtol) and absolute tolerance (atol)
            if diff > atol + rtol * abs(identity_matrix[i, j]):
                return False
    return True


class Kraus(krauschannel):
    r"""Kraus(kmatrices).

    Custom N-qubit Kraus channel specified by a list of 2^N x 2^N Kraus matrices.

    A Kraus channel is defined by:

    .. math::
        \mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger,

    where :math:`E_k` are Kraus matrices that need to fulfill :math:`\sum_k E_k^\dagger E_k = I`.

    If the Kraus matrices are all proportional to unitaries, use :func:`MixedUnitary` instead.

    The Kraus matrices are defined in the computational basis in the usual textbook order
    (the first qubit corresponds to the left-most qubit).
    For 1 qubit, we have :math:`|0\rangle`, :math:`|1\rangle`.
    For 2 qubits, we have :math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`, :math:`|11\rangle`.

    .. note::
        Currently, only 1 and 2-qubit custom Kraus channels are supported.

    A `Kraus` channel becomes loss-aware simply by including one or more
    :class:`LossyOperator` branches in ``E``. In that case, :meth:`hasloss`
    returns ``True`` and :meth:`lossoperators`, :meth:`survivaloperators`, and
    :meth:`losseffect` can be used to inspect the leakage structure.

    The entries of a :class:`LossyOperator` are amplitudes, not probabilities.
    The corresponding loss probabilities appear in :meth:`losseffect`, which
    sums :math:`L_k^\dagger L_k` over all lossy branches :math:`L_k`.

    See Also:
        :class:`MixedUnitary`
        :class:`GateCustom`
        :class:`LossyOperator`
        :class:`Operator`

    Parameters:
        kmatrices (list): List of :math:`2^N \times 2^N` complex matrices. The number of qubits is equal to :math:`N`.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> c = Circuit()
        >>> c.push(Kraus([Matrix([[1, 0], [0, sqrt(0.9)]]), Matrix([[0, sqrt(0.1)], [0, 0]])]), 0)
        1-qubit circuit with 1 instruction:
        └── Kraus(Operator([[1, 0], [0, 0.948683298050514]]), Operator([[0, 0.316227766016838], [0, 0]])) @ q[0]
        <BLANKLINE>

        Kraus Matrices

        >>> g=Kraus([Matrix([[1, 0], [0, sqrt(0.9)]]), Matrix([[0, sqrt(0.1)], [0, 0]])])
        >>> g.krausmatrices()
        [[1.0, 0]
        [0, 0.948683298050514]
        , [0, 0.316227766016838]
        [0, 0]
        ]

        # Example 2: Kraus with Projector0 and Projector1

        >>> c = Circuit()
        >>> c.push(Kraus([Projector0(), Projector1()]), 0)
        1-qubit circuit with 1 instruction:
        └── Kraus(P₀(1), P₁(1)) @ q[0]
        <BLANKLINE>

        # Example 3: Kraus with a matrix and Projector1

        >>> c = Circuit()
        >>> c.push(Kraus([Matrix([[1, 0], [0, 0]]), Projector1()]), 0)
        1-qubit circuit with 1 instruction:
        └── Kraus(Operator([[1, 0], [0, 0]]), P₁(1)) @ q[0]
        <BLANKLINE>

        # Example 4: Loss-aware Kraus with a LossyOperator branch

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
        
        # Example 5: Symbolic Kraus evaluation

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
        E: List[Union[se.Matrix, np.ndarray, sp.Matrix, mc.AbstractOperator]],
        rtol=1e-12,
    ):
        if len(E) == 0:
            raise ValueError("Vector of Kraus operators cannot be empty")

        Es = []
        for Ek in E:
            if isinstance(Ek, mc.AbstractOperator):
                Es.append(Ek)
            elif isinstance(Ek, (se.Matrix, np.ndarray, sp.Matrix)):
                Es.append(mc.Operator(self._convert_to_matrix(Ek)))
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
            if not np.allclose(ksum, np.eye(M, dtype=np.complex128), rtol=1e-12):
                raise ValueError(
                    "List of Kraus matrices should fulfill \\sum_k E_k^\\dagger E_k = I."
                )

            # Check if ksum is approximately the identity matrix
            if not _is_approx_identity(ksum, M, rtol):
                raise ValueError(
                    "List of Kraus matrices should fulfill \\sum_k E_k^\\dagger E_k = I."
                )

        self.E = Es

        super().__init__()
        self._num_qubits = self.N
        self._parnames = ("E",)

    def evaluate(self, values: dict):
        """Evaluates symbolic parameters in Kraus matrices using a dictionary of values.

        Args:
            values (dict): A dictionary where keys are symbolic variables and values are the corresponding numerical values.

        Returns:
            Kraus: A new Kraus instance with evaluated matrices.
        """
        evaluated_E = []
        for op in self.E:
            if isinstance(op, mc.Operator):
                # Substitute symbolic parameters in the matrix
                matrix = op.matrix()
                evaluated_matrix = se.Matrix(
                    [
                        [element.subs(values) for element in row]
                        for row in matrix.tolist()
                    ]
                )
                evaluated_E.append(mc.Operator(evaluated_matrix))
            else:
                # If it's already an AbstractOperator, we just append it
                evaluated_E.append(op.evaluate(values))

        return Kraus(evaluated_E)

    def _convert_to_matrix(self, operator):
        """Converts matrices to symengine.Matrix, stripping imaginary parts if zero."""
        if isinstance(operator, (np.ndarray, sp.Matrix)):
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
        """Returns the Kraus operators."""
        return self.E

    def hasloss(self):
        """Return True when the channel contains any LossyOperator branch."""
        return any(isinstance(op, mc.LossyOperator) for op in self.E)

    def lossoperators(self):
        """Return the LossyOperator branches of the channel."""
        return [op for op in self.E if isinstance(op, mc.LossyOperator)]

    def survivaloperators(self):
        """Return the non-lossy branches of the channel."""
        return [op for op in self.E if not isinstance(op, mc.LossyOperator)]

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

        .. Note::
            The entries of :class:`LossyOperator` are amplitudes, not
            probabilities. ``losseffect()`` converts them into probabilities by
            summing :math:`L_k^\dagger L_k`.

        The survival and lossy operators together satisfy

        .. math::
            \sum_k S_k^\dagger S_k + \Lambda = I,

        where :math:`S_k` are the non-lossy branches.

        If the channel has no lossy operators, the zero matrix is returned.

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
            return mc.Operator(np.zeros((M, M), dtype=np.complex128))

        effect = np.zeros((M, M), dtype=np.complex128)
        for op in loss_ops:
            effect += np.array(op.opsquared().matrix().tolist(), dtype=np.complex128)

        effect[np.abs(effect) < tol] = 0
        return mc.Operator(effect)

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
        """Returns the unwrapped Kraus matrices."""
        return self.krausmatrices()


__all__ = ["Kraus"]
