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
"""Entanglement measures (BondDim, VonNeumannEntropy, etc.)."""

from mimiqcircuits.operations.operation import Operation


class BondDim(Operation):
    r"""Record the MPS bond dimension at a cut of the system in a z-variable.

    The bond dimension is only defined for a matrix-product state (MPS), so
    it is only meaningful with the MPS simulator. An MPS can be written as

    .. math::
        |\psi \rangle = \sum_{s_1,s_2,\ldots=1}^2
        \sum_{i_1}^{\chi_1} \sum_{i_2}^{\chi_2} \ldots \sum_{i_N}^{\chi_N}
        A^{(s_1)}_{i_0i_1} A^{(s_2)}_{i_1 i_2} A^{(s_3)}_{i_2 i_3} \ldots
        A^{(s_N)}_{i_{N-1}i_N} | s_1, s_2, s_3, \ldots, s_N \rangle .

    where :math:`\chi_k` is the bond dimension, the dimension of the index
    :math:`i_k`. The first and last bond dimensions are dummies,
    :math:`\chi_0=\chi_N=1`. A bond dimension of 1 means there is no
    entanglement between the two sides of the cut.

    The instruction takes one qubit ``k`` and one z-variable. It records
    the dimension of the bond between qubits ``k - 1`` and ``k``, that is
    between qubits ``0, ..., k - 1`` and ``k, ..., N - 1``. For ``k = 0``
    the result is always 1. The quantum state is not changed.

    See Also:
        :class:`VonNeumannEntropy`, :class:`SchmidtRank`

    Examples:
        >>> from mimiqcircuits import *
        >>> # bond dimension between qubits 4 and 5, stored in z-variable 1
        >>> k = 5
        >>> c = Circuit()
        >>> c.push(BondDim(), k, 1)
        6-qubit, 2-zvar circuit with 1 instruction:
        └── BondDim @ q[5], z[1]
        <BLANKLINE>
    """

    _name = "BondDim"
    _num_zvars = 1
    _num_bits = 0
    _num_zregs = 1
    _num_qubits = 1

    def __init__(self):
        super().__init__()
        self._qregsizes = [self._num_qubits]
        self._zregsizes = [self._num_zvars]
        self._parnames = ()

    def __str__(self):
        return f"{self.name}"

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def parnames(self):
        return self._parnames

    def iswrapper(self):
        return False

    def inverse(self):
        """Return a new ``BondDim``: it does not change the state."""
        return BondDim()

    def power(self, p):
        """Not supported.

        Raises:
            TypeError: Always.
        """
        raise TypeError("BondDim^p is not defined.")

    def control(self, num_qubits):
        """Not supported.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Controlled BondDim is not defined.")

    @staticmethod
    def isunitary():
        return True


class VonNeumannEntropy(Operation):
    r"""Record the bipartite von Neumann entanglement entropy in a z-variable.

    For a pure state :math:`\rho = | \psi \rangle\langle \psi |` split
    into subsystems :math:`A` and :math:`B`, the entanglement entropy is

    .. math::
        \mathcal{S}(\rho_A) = - \mathrm{Tr}(\rho_A \log_2 \rho_A)
        = - \mathrm{Tr}(\rho_B \log_2 \rho_B)
        = \mathcal{S}(\rho_B),

    where :math:`\rho_A = \mathrm{Tr}_B(\rho)` is the reduced density
    matrix. A product state has :math:`\mathcal{S}(\rho_A)=0` and a
    maximally entangled pair of qubits gives :math:`\mathcal{S}(\rho_A)=1`.

    The instruction takes one qubit ``k`` and one z-variable. The
    bipartition is :math:`A = \{0, \ldots, k-1\}` and
    :math:`B = \{k, \ldots, N-1\}`, where :math:`N` is the number of
    qubits. For ``k = 0``, :math:`A` is empty and the result is always 0.
    The quantum state is not changed.

    With noise and quantum trajectories, the entropy of the pure state of
    each trajectory is recorded.

    See Also:
        :class:`BondDim`, :class:`SchmidtRank`

    Examples:
        >>> from mimiqcircuits import *
        >>> # entropy between qubits 0-4 and qubits 5 onwards, in z-variable 1
        >>> k = 5
        >>> c = Circuit()
        >>> c.push(VonNeumannEntropy(), k, 1)
        6-qubit, 2-zvar circuit with 1 instruction:
        └── VonNeumannEntropy @ q[5], z[1]
        <BLANKLINE>
    """

    _name = "VonNeumannEntropy"
    _num_zvars = 1
    _num_bits = 0
    _num_zregs = 1
    _num_qubits = 1

    def __init__(self):
        super().__init__()
        self._qregsizes = [self._num_qubits]
        self._zregsizes = [self._num_zvars]
        self._parnames = ()

    def __str__(self):
        return f"{self.name}"

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def parnames(self):
        return self._parnames

    def iswrapper(self):
        return False

    def inverse(self):
        """Return a new ``VonNeumannEntropy``: it does not change the state."""
        return VonNeumannEntropy()

    def power(self, p):
        """Not supported.

        Raises:
            TypeError: Always.
        """
        raise TypeError("VonNeumannEntropy^p is not defined.")

    def control(self, num_qubits):
        """Not supported.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Controlled VonNeumannEntropy is not defined.")

    @staticmethod
    def isunitary():
        return True


class SchmidtRank(Operation):
    r"""Record the Schmidt rank of a bipartition in a z-variable.

    The Schmidt decomposition of a pure state split into subsystems
    :math:`A` and :math:`B` is

    .. math::
        |\psi\rangle = \sum_{i=1}^{r} s_i |\alpha_i\rangle \otimes |\beta_i\rangle,

    where :math:`|\alpha_i\rangle` (:math:`|\beta_i\rangle`) are
    orthonormal states of :math:`A` (:math:`B`). The Schmidt rank is the
    number of terms :math:`r`. A product state gives :math:`r=1`, and
    :math:`r>1` signals entanglement.

    The instruction takes one qubit ``k`` and one z-variable. The
    bipartition is :math:`A = \{0, \ldots, k-1\}` and
    :math:`B = \{k, \ldots, N-1\}`, where :math:`N` is the number of
    qubits. For ``k = 0``, :math:`A` is empty and the result is always 1.
    The quantum state is not changed.

    For a matrix-product state that is optimally compressed, the Schmidt
    rank equals the bond dimension (see :class:`BondDim`).

    See Also:
        :class:`VonNeumannEntropy`, :class:`BondDim`

    Examples:
        >>> from mimiqcircuits import *
        >>> # Schmidt rank between qubits 0-4 and qubits 5 onwards, in z-variable 1
        >>> k = 5
        >>> c = Circuit()
        >>> c.push(SchmidtRank(), k, 1)
        6-qubit, 2-zvar circuit with 1 instruction:
        └── SchmidtRank @ q[5], z[1]
        <BLANKLINE>
    """

    _name = "SchmidtRank"
    _num_zvars = 1
    _num_bits = 0
    _num_zregs = 1
    _num_qubits = 1

    def __init__(self):
        super().__init__()
        self._qregsizes = [self._num_qubits]
        self._zregsizes = [self._num_zvars]
        self._parnames = ()

    def __str__(self):
        return f"{self.name}"

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def parnames(self):
        return self._parnames

    def iswrapper(self):
        return False

    def inverse(self):
        """Return a new ``SchmidtRank``: it does not change the state."""
        return SchmidtRank()

    def power(self, p):
        """Not supported.

        Raises:
            TypeError: Always.
        """
        raise TypeError("SchmidtRank^p is not defined.")

    def control(self, num_qubits):
        """Not supported.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Controlled SchmidtRank is not defined.")

    @staticmethod
    def isunitary():
        return True


__all__ = ["BondDim", "VonNeumannEntropy", "SchmidtRank"]
