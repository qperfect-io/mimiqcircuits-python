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
"""Kraus channel definitions."""

from mimiqcircuits.operations.operation import Operation
import symengine as se
from itertools import accumulate
import numpy as np


class krauschannel(Operation):
    r"""Base class for all N-qubit Kraus channels.

    A Kraus channel acts on a density matrix :math:`\rho` as

    .. math::

        \mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger,

    where the Kraus operators :math:`E_k` fulfill :math:`\sum_k E_k^\dagger E_k \leq I`.
    The channel is completely positive and trace preserving (CPTP) when the
    equality :math:`\sum_k E_k^\dagger E_k = I` holds.

    A Kraus channel is a mixed unitary channel when every :math:`E_k` is
    proportional to a unitary :math:`U_k`, so that
    :math:`\mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger` with
    probabilities :math:`p_k` that add up to 1. Mixed unitary channels can be
    sampled as a random choice of unitary gate (see
    :meth:`~mimiqcircuits.Circuit.sample_mixedunitaries`).

    Kraus channels cannot be inverted or raised to a power: :meth:`inverse` and
    :meth:`power` raise :class:`NotImplementedError`.

    See Also:
        :meth:`krausoperators`, :meth:`krausmatrices`, :meth:`ismixedunitary`,
        :meth:`probabilities`, :meth:`unitarygates`
    """

    _name = None
    _num_qubits = None
    _num_bits = 0
    _num_cregs = 0
    _cregsizes = ()
    _num_zvars = 0

    def krausmatrices(self):
        r"""Return the matrices of the Kraus operators :math:`E_k` of the channel.

        These are the matrices of :meth:`krausoperators`, in the same order. For
        a parametric channel the entries are symbolic expressions.

        Returns:
            list: One symengine matrix per Kraus operator.

        Examples:
            >>> from mimiqcircuits import *
            >>> AmplitudeDamping(0.36).krausmatrices()
            [[1.0, 0]
            [0, 0.8]
            , [0, 0.6]
            [0, 0]
            ]
        """
        return [se.Matrix(kraus.matrix().tolist()) for kraus in self.krausoperators()]

    def unwrappedkrausmatrices(self):
        """Return the Kraus matrices of a channel whose parameters all have values.

        Returns the same matrices as :meth:`krausmatrices`, after checking that
        every parameter evaluates to a constant.

        Returns:
            list: One symengine matrix per Kraus operator.

        Raises:
            ValueError: If a parameter is still a free symbol.

        See Also:
            :meth:`krausmatrices`

        Examples:
            >>> from mimiqcircuits import *
            >>> # PauliX(p) has Kraus operators sqrt(1-p) I and sqrt(p) X
            >>> op = PauliX(0.2)
            >>> op.unwrappedkrausmatrices()
            [[0.894427190999916, 0]
            [0, 0.894427190999916]
            , [0, 0.447213595499958]
            [0.447213595499958, 0]
            ]
        """

        if self.numparams() == 0:
            return self.krausmatrices()

        import sympy as sp

        params = []
        for name in self.parnames:
            v = sp.simplify(self.getparam(name))
            if isinstance(v, (int, float)):
                params.append(v)
            elif isinstance(v, sp.Basic) and v.is_constant():
                params.append(float(v))
            else:
                raise ValueError(f"Undefined parameter {name} in {self.name}")

        return self.krausmatrices()

    @classmethod
    def isCPTP(self):
        r"""Whether the channel is completely positive and trace preserving (CPTP).

        A channel is CPTP when its Kraus operators fulfill
        :math:`\sum_k E_k^\dagger E_k = I`, and only trace decreasing when
        :math:`\sum_k E_k^\dagger E_k < I`. Every channel in the package is
        CPTP.

        Returns:
            bool: ``True``.
        """
        return True

    @classmethod
    def ismixedunitary(self):
        r"""Whether the channel is a mixed unitary channel.

        A channel is mixed unitary when every Kraus operator :math:`E_k` is
        proportional to a unitary matrix :math:`U_k`, so that

        .. math::

            \mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger,

        with probabilities :math:`0 \leq p_k \leq 1` that add up to 1. The answer
        depends only on the channel type, not on its parameters, so this is a
        class method. Only mixed unitary channels support :meth:`probabilities`,
        :meth:`cumprobabilities`, :meth:`unitarygates` and
        :meth:`unitarymatrices`.

        Returns:
            bool: ``True`` for mixed unitary channel types, ``False`` otherwise.

        Examples:
            >>> from mimiqcircuits import *
            >>> # PauliX applies X with probability p: mixed unitary
            >>> PauliX(0.1).ismixedunitary()
            True

            >>> # amplitude damping has a non-unitary Kraus operator
            >>> AmplitudeDamping(0.1).ismixedunitary()
            False
        """
        return issubclass(self, krauschannel) and False

    def inverse(self):
        """Kraus channels have no inverse.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Cannot invert Kraus channels")

    def power(op, n):
        """Kraus channels cannot be raised to a power.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Cannot take the power of Kraus channels")

    def iswrapper(self):
        return False

    def probabilities(self):
        r"""Return the probabilities :math:`p_k` of a mixed unitary channel.

        For :math:`\mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger` this is
        the list of :math:`p_k`, in the same order as :meth:`unitarygates`.

        Returns:
            list: One probability per unitary.

        Raises:
            ValueError: If the channel is not mixed unitary.

        Examples:
            >>> from mimiqcircuits import *
            >>> # identity with probability 0.9, X with probability 0.1
            >>> PauliX(0.1).probabilities()
            [0.9, 0.1]
        """
        raise ValueError(
            "Probabilities can only be returned from a mixed unitary channel."
        )

    def numparams(self):
        return len(self._parnames)

    def cumprobabilities(self):
        """Return the cumulative sums of :meth:`probabilities`.

        The last entry is 1. Drawing a uniform number ``r`` and picking the
        first index whose cumulative probability exceeds ``r`` samples one of the
        unitaries with the right probability.

        Returns:
            list: Cumulative probabilities as symengine values, symbolic
            expressions when the probabilities are symbolic.

        Raises:
            ValueError: If the channel is not mixed unitary.

        Examples:
            >>> from mimiqcircuits import *
            >>> # identity with probability 0.9, then X, Y, Z with 0.1/3 each
            >>> Depolarizing1(0.1).cumprobabilities()
            [0.9, 0.933333333333333, 0.966666666666667, 1.0]
        """
        return self._cumprobabilities(self.ismixedunitary())

    def _cumprobabilities(self, mixed_unitary):
        if not mixed_unitary:
            raise ValueError(
                "Cumulative probabilities only available for mixed unitary channels."
            )
        return [se.sympify(cp) for cp in accumulate(self.probabilities())]

    def unwrappedcumprobabilities(self):
        """Return :meth:`cumprobabilities` as plain Python floats.

        Returns:
            list[float]: Cumulative probabilities.

        Raises:
            ValueError: If the channel is not mixed unitary.

        Examples:
            >>> from mimiqcircuits import *
            >>> PauliX(0.1).unwrappedcumprobabilities()
            [0.9, 1.0]
        """
        return [float(p) for p in self.cumprobabilities()]

    def krausoperators(self):
        """Return the Kraus operators :math:`E_k` of the channel as operator objects.

        Every concrete channel implements this method; the base class raises.

        Returns:
            list: One operator (for example :class:`Operator`,
            :class:`DiagonalOp` or :class:`RescaledGate`) per Kraus operator.

        Raises:
            NotImplementedError: When called on a class that does not define it.

        Examples:
            >>> from mimiqcircuits import *
            >>> # projective noise in Z: project on |0> or on |1>
            >>> ProjectiveNoise("Z").krausoperators()
            [P₀(1), P₁(1)]
        """
        raise NotImplementedError(
            "This method should be implemented for the specific Kraus channel."
        )

    def haslossybranch(self):
        """Whether any branch of the channel loses qubits when drawn.

        A branch is lossy when its Kraus operator is a
        :class:`~mimiqcircuits.LossyOperator`.

        Returns:
            bool: ``True`` if at least one Kraus operator is lossy.

        Examples:
            >>> from mimiqcircuits import *
            >>> PauliX(0.1).haslossybranch()
            False
            >>> # a Kraus channel where |1> is lost with probability 0.2
            >>> import numpy as np
            >>> survival = Operator(np.array([[1, 0], [0, np.sqrt(0.8)]]))
            >>> lossy = LossyOperator(np.array([[0, 0], [0, np.sqrt(0.2)]]))
            >>> Kraus([survival, lossy]).haslossybranch()
            True
        """
        from mimiqcircuits.operations.operators.lossyoperator import LossyOperator

        return any(isinstance(op, LossyOperator) for op in self.krausoperators())

    def squaredkrausoperators(self):
        r"""Return :math:`E_k^\dagger E_k` for every Kraus operator :math:`E_k`.

        The sum of the returned operators is the identity for a trace
        preserving channel.

        Returns:
            list: One operator per Kraus operator, in the same order.

        Examples:
            >>> from mimiqcircuits import *
            >>> # a projector P satisfies P^dagger P = P
            >>> Kraus([Projector0(), Projector1()]).squaredkrausoperators()
            [P₀(1), P₁(1)]
        """
        return [k.opsquared() for k in self.krausoperators()]

    def unitarymatrices(self):
        r"""Return the matrices of the unitaries :math:`U_k` of a mixed unitary channel.

        For :math:`\mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger` these are
        the matrices of :meth:`unitarygates`, in the same order as
        :meth:`probabilities`. For a parametric channel the entries are symbolic
        expressions.

        Returns:
            list: One symengine matrix per unitary.

        Raises:
            ValueError: If the channel is not mixed unitary.

        See Also:
            :meth:`ismixedunitary`, :meth:`probabilities`, :meth:`krausmatrices`

        Examples:
            >>> from mimiqcircuits import *
            >>> # identity and Pauli X
            >>> PauliX(0.2).unitarymatrices()
            [[1.0, 0]
            [0, 1.0]
            , [0, 1.0]
            [1.0, 0]
            ]
        """
        if not self.ismixedunitary():
            raise ValueError(
                "unitarymatrices only available for mixed unitary Kraus channels."
            )
        return [se.Matrix(g.matrix().tolist()) for g in self.unitarygates()]

    def unwrappedunitarymatrices(self):
        """Return :meth:`unitarymatrices` with every entry evaluated to a number.

        Real entries become ``float`` and complex entries ``complex``.

        Returns:
            list: One symengine matrix with numeric entries per unitary.

        Raises:
            ValueError: If the channel is not mixed unitary.
        """
        import sympy as sp

        matrices = []

        for matrix in self.unitarymatrices():
            # Prepare a list to hold the converted values
            flat_matrix = []

            for i in range(matrix.rows):
                for j in range(matrix.cols):
                    entry = matrix[i, j]
                    # Convert symbolic expression to numerical form
                    entry_numeric = sp.N(entry)

                    # Handle real and complex parts separately
                    if entry_numeric.is_real:
                        flat_matrix.append(float(entry_numeric))
                    else:
                        flat_matrix.append(complex(entry_numeric))

            # Create a new symengine matrix from the flattened list
            new_matrix = se.Matrix(matrix.rows, matrix.cols, flat_matrix)
            matrices.append(new_matrix)

        return matrices

    def unitarygates(self):
        r"""Return the unitaries :math:`U_k` of a mixed unitary channel as gates.

        For :math:`\mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger` these are
        the :math:`U_k`, in the same order as :meth:`probabilities`.

        Returns:
            list: One gate per unitary.

        Raises:
            ValueError: If the channel is not mixed unitary.

        Examples:
            >>> from mimiqcircuits import *
            >>> PauliX(0.1).unitarygates()
            [ID, X]
        """
        if not self.ismixedunitary():
            raise ValueError("unitarygates only available for mixed unitary channels.")


__all__ = ["krauschannel"]
