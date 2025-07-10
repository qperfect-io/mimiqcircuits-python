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

from mimiqcircuits import Operation


class BondDim(Operation):
    r"""Operation to get the bond dimension between two halves of the system and store it in a z-register.

    The bond dimension is only defined for a matrix-product state (MPS), which
    can be written as:

    **State Representation**

    .. math::
        |\psi \rangle = \sum_{s_1,s_2,\ldots=1}^2
        \sum_{i_1}^{\chi_1} \sum_{i_2}^{\chi_2} \ldots \sum_{i_N}^{\chi_N}
        A^{(s_1)}_{i_0i_1} A^{(s_2)}_{i_1 i_2} A^{(s_3)}_{i_2 i_3} \ldots
        A^{(s_N)}_{i_{N-1}i_N} | s_1, s_2, s_3, \ldots, s_N \rangle .

    Here, :math:`\chi_k` is the bond dimension, i.e., the dimension of the index :math:`i_k`.
    The first and last bond dimensions are dummies, :math:`\chi_0=\chi_N=1`.
    A bond dimension of 1 means there is no entanglement between the two halves
    of the system.

    See Also:
        :class:`VonNeumannEntropy`, :class:`SchmidtRank`

    Examples:

        When pushing to a circuit, the qubit index `k` that we give will return
        the bond dimension :math:`i_{k-1}` in the above notation. In other words, we
        associate link `k` with qubit `k+1`. For `k=1`, the bond dimension
        returned will always be 1.

        >>> from mimiqcircuits import *
        >>> k = 5
        >>> c = Circuit()
        >>> c.push(BondDim(), k, 1)
        6-qubit, 2-zvar circuit with 1 instructions:
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
        return BondDim()

    def power(self, p):
        raise TypeError("BondDim^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled BondDim is not defined.")

    @staticmethod
    def isunitary():
        return True


class VonNeumannEntropy(Operation):
    r"""Operation to get the bipartite Von Neumann entanglement entropy and store it in a z-register.

    The entanglement entropy for a bipartition into subsystems :math:`A` and :math:`B`
    is defined for a pure state :math:`\rho = | \psi \rangle\langle \psi |` as:

    **Entanglement Entropy**

    .. math::
        \mathcal{S}(\rho_A) = - \mathrm{Tr}(\rho_A \log_2 \rho_A)
        = - \mathrm{Tr}(\rho_B \log_2 \rho_B)
        = \mathcal{S}(\rho_A)

    where :math:`\rho_A = \mathrm{Tr}_B(\rho)` is the reduced density matrix.
    A product state has :math:`\mathcal{S}(\rho_A)=0` and a maximally entangled state
    between :math:`A` and :math:`B` gives :math:`\mathcal{S}(\rho_A)=1`.

    We only consider bipartitions where :math:`A=\{1,\ldots,k-1\}` and :math:`B=\{k,\ldots,N\}`,
    for some :math:`k` and where :math:`N` is the total number of qubits.

    When the system is open (i.e., with noise) and we are using quantum trajectories,
    the entanglement entropy of each trajectory is returned during execution.

    See Also:
        :class:`BondDim`, :class:`SchmidtRank`

    Examples:

        When pushing to a circuit, the qubit index `k` takes the role of the above bipartition
        into `A` and `B`. For `k=1`, `A` is empty and the entanglement entropy will
        always return 0.

        >>> from mimiqcircuits import *
        >>> k = 5
        >>> c = Circuit()
        >>> c.push(VonNeumannEntropy(), k, 1)
        6-qubit, 2-zvar circuit with 1 instructions:
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
        return VonNeumannEntropy()

    def power(self, p):
        raise TypeError("VonNeumannEntropy^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled VonNeumannEntropy is not defined.")

    @staticmethod
    def isunitary():
        return True


class SchmidtRank(Operation):
    r"""Operation to get the Schmidt rank of a bipartition and store it in a z-register.

    A Schmidt decomposition for a bipartition into subsystems :math:`A` and :math:`B`
    is defined for a pure state as:

    **Schmidt Decomposition**

    .. math::
        |\psi\rangle = \sum_{i=1}^{r} s_i |\alpha_i\rangle \otimes |\beta_i\rangle,

    where :math:`|\alpha_i\rangle` (:math:`|\beta_i\rangle`) are orthonormal states acting
    on :math:`A` (:math:`B`). The Schmidt rank is the number of terms :math:`r` in the sum.
    A product state gives :math:`r=1`, and :math:`r>1` signals entanglement.

    We only consider bipartitions where :math:`A=\{1,\ldots,k-1\}` and :math:`B=\{k,\ldots,N\}`,
    for some :math:`k` and where :math:`N` is the total number of qubits.

    In MPS (Matrix Product States), when the state is optimally compressed, the Schmidt rank should
    be equal to the bond dimension (see :class:`BondDim`).

    See Also:
        :class:`VonNeumannEntropy`, :class:`BondDim`

    Examples:

        When pushing to a circuit, the qubit index `k` takes the role of the above bipartition
        into `A` and `B`. For `k=1`, `A` is empty and the Schmidt rank will
        always return 1.

        >>> from mimiqcircuits import *
        >>> k = 5
        >>> c = Circuit()
        >>> c.push(SchmidtRank(), k, 1)
        6-qubit, 2-zvar circuit with 1 instructions:
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
        return SchmidtRank()

    def power(self, p):
        raise TypeError("SchmidtRank^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled SchmidtRank is not defined.")

    @staticmethod
    def isunitary():
        return True


__all__ = ["BondDim", "VonNeumannEntropy", "SchmidtRank"]
