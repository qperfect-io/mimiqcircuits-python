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

import sympy as sp
from mimiqcircuits.operations.operation import Operation
import symengine as se
import numpy as np


class krauschannel(Operation):
    r"""Supertype for all N-qubit Kraus channels.

    A Kraus channel is a quantum operation on a density matrix :math:`\\rho` of the form:

    .. math::

        \mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger,

    where :math:`E_k` are Kraus operators that need to fulfill :math:`\sum_k E_k^\dagger E_k \leq I`.

    **Special Properties:**

    - :func:`isCPTP`: A Kraus channel is a completely positive and trace-preserving (CPTP)
      operation when :math:`\\sum_k E_k^\dagger E_k = I`. Currently, all noise channels are CPTP.

    - :func:`ismixedunitary`: A Kraus channel is called a mixed unitary channel when the
      Kraus operators :math:`E_k` are each proportional to a unitary matrix :math:`U_k`, i.e., when
      :math:`\\mathcal{E}(\\rho) = \\sum_k p_k U_k \\rho U_k^\dagger` with some probabilities
      :math:`0 \leq p_k \leq 1` that add up to 1 and :math:`U_k^\dagger U_k = I`.

    See Also:
        :func:`krausmatrices`
        :func:`unitarymatrices`
        :func:`probabilities`
    """

    _name = None
    _num_qubits = None
    _num_bits = 0
    _num_cregs = 0
    _cregsizes = ()
    _num_zvars = 0

    def krausmatrices(self):
        """
        Returns the Kraus matrices associated with the given Kraus channel.

        A mixed unitary channel is written as:

        .. math::

            \\mathcal{E}(\\rho) = \\sum_k p_k U_k \\rho U_k^\\dagger,

        where :math:`U_k` are the unitary matrices returned by this function.

        If the Kraus channel is parametric, the matrix elements are wrapped in a
        symengine or sympy object.

        Returns:
            list: A list of symengine matrices representing the Kraus operators.
        """
        return [se.Matrix(kraus.matrix().tolist()) for kraus in self.krausoperators()]

    def unwrappedkrausmatrices(self):
        """
        Returns the unitary Kraus matrices associated to the mixed unitary Kraus channel
        without symbolic wrapper.

        See Also:
            :func:`unitarymatrices`

        Example:
            >>> from mimiqcircuits import *
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
        r"""Determine whether the Kraus channel is Completely Positive and Trace Preserving (CPTP).

        A quantum operation is CPTP if the Kraus operators fulfill the condition:

        .. math::
            \sum_k E_k^\dagger E_k = I

        If the condition is:

        .. math::
            \sum_k E_k^\dagger E_k < I

        then the operation is not CPTP.

        Note:
            Currently, all noise channels are considered CPTP.

        Parameters:
            krauschannel: The Kraus channel to check.

        Returns:
            bool: `True` if the channel is CPTP, `False` otherwise.
        """
        return issubclass(self, krauschannel) and False

    @classmethod
    def ismixedunitary(self):
        r"""Determine whether the quantum operation is a mixed unitary channel.

        A channel is considered mixed unitary if all the Kraus operators :math:`E_k` are proportional
        to a unitary matrix :math:`U_k`, i.e.,

        .. math::

            \mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger,

        with some probabilities :math:`0 \leq p_k \leq 1` that add up to 1, and :math:`U_k^\dagger U_k = I`.

        Parameters:
            krauschannel: The Kraus channel to check.

        Returns:
            bool: `True` if the channel is a mixed unitary channel, `False` otherwise.

        Examples:
            >>> from mimiqcircuits import *
            >>> PauliX(0.1).ismixedunitary()
            True

            >>> AmplitudeDamping(0.1).ismixedunitary()
            False
        """
        return issubclass(self, krauschannel) and False

    def inverse(self):
        raise NotImplementedError("Cannot invert Kraus channels")

    def power(op, n):
        raise NotImplementedError("Cannot take the power of Kraus channels")

    def iswrapper(self):
        return False

    def probabilities(self):
        """
        Returns the probabilities for each Kraus operator in a mixed unitary channel.

        A mixed unitary channel is written as:

        .. math::

            \\mathcal{E}(\\rho) = \\sum_k p_k U_k \\rho U_k^\\dagger,

        where :math:`p_k` are the probabilities.

        This method is valid only for mixed unitary channels.

        Returns:
            list: A list of probabilities for each Kraus operator.
        """
        raise ValueError(
            "Probabilities can only be returned from a mixed unitary channel."
        )

    def numparams(self):
        return len(self._parnames)

    def cumprobabilities(self):
        """
        Returns the cumulative sum of probabilities for each Kraus operator in a mixed unitary channel.

        A mixed unitary channel is written as:

        .. math::

            \\mathcal{E}(\\rho) = \\sum_k p_k U_k \\rho U_k^\\dagger,

        where :math:`p_k` are the probabilities.

        This method is valid only for mixed unitary channels.

        Returns:
            list: Cumulative probabilities for the Kraus operators.

        Examples:
            >>> from mimiqcircuits import *
            >>> Depolarizing1(0.1).cumprobabilities()
            [0.9, 0.933333333333333, 0.966666666666667, 1.0]
        """
        return self._cumprobabilities(self.ismixedunitary())

    def _cumprobabilities(self, mixed_unitary):
        if not mixed_unitary:
            raise ValueError(
                "Cumulative probabilities only available for mixed unitary channels."
            )
        return [se.RealDouble(cp) for cp in np.cumsum(self.probabilities())]

    def unwrappedcumprobabilities(self):
        """
        Returns the cumulative sum of probabilities for the mixed unitary channel without
        symbolic wrapping.

        See Also:
            :func:`cumprobabilities`

        Returns:
            list: A list of cumulative probabilities as float values.
        """
        return [float(p) for p in self.cumprobabilities()]

    def krausoperators(self):
        """
        Returns the Kraus operators associated with the given Kraus channel.

        This should be implemented for each specific channel.

        Returns:
            list: A list of matrices representing the Kraus operators.
        """
        raise NotImplementedError(
            "This method should be implemented for the specific Kraus channel."
        )

    def squaredkrausoperators(self):
        """
        Returns the square of the Kraus operators (:math:`E_k^\\dagger E_k`).
        This computes the Hermitian conjugate (dagger) of each Kraus operator and multiplies it by the operator itself.

        Returns:
            list: A list of squared Kraus operators.
        """
        return [k.opsquared() for k in self.krausoperators()]

    def unitarymatrices(self):
        """
        Unitary matrices associated with the given mixed unitary Kraus channel.

        A mixed unitary channel is written as:

        .. math::

            \\mathcal{E}(\\rho) = \\sum_k p_k U_k \\rho U_k^\\dagger,

        where :math:`U_k` are the unitary matrices.

        An error is raised if the channel is not mixed unitary (i.e., `ismixedunitary(self)==False`).

        Note:
            If the Kraus channel is parametric, the matrix elements are wrapped in a
            symbolic object (e.g., from sympy or symengine). To manipulate expressions,
            use the appropriate symbolic manipulation libraries.

        See Also:
            :func:`ismixedunitary`
            :func:`probabilities`
            :func:`krausmatrices`

        Examples:
            >>> from mimiqcircuits import *
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
        """
        Returns the unitary matrices associated with the mixed unitary channel
        without symbolic wrapping.

        Returns:
            list: A list of unitary matrices as numerical values.
        """
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
        """
        Returns the unitary gates associated with the given mixed unitary Kraus channel.

        A mixed unitary channel is written as:

        .. math::

            \\sum_k p_k U_k \\rho U_k^\\dagger,

        where :math:`U_k` are the unitary operators.

        This method is valid only for mixed unitary channels.
        """
        if not self.ismixedunitary():
            raise ValueError("unitarygates only available for mixed unitary channels.")


__all__ = ["Noise"]
