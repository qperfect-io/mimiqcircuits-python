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


import mimiqcircuits as mc
import numpy as np
from itertools import product
import symengine as se
from mimiqcircuits.operations.krauschannel import krauschannel
import sympy as sp


class Depolarizing(krauschannel):
    r"""N-qubit depolarizing noise channel.

    The Kraus operators for the depolarizing channel are given by:

    .. math::
        E_1 = \sqrt{1-p} I_N, \quad E_i = \sqrt{p/(4^N-1)} P_i,

    where :math:`p \in [0,1]` is a probability, and :math:`P_i` is an :math:`N`-qubit Pauli string
    operator, i.e., a tensor product of one-qubit Pauli operators.

    There is exactly one Kraus operator :math:`E_{i>1}` for each distinct combination of
    Pauli operators :math:`P_i`, except for the :math:`N`-qubit identity
    :math:`I_N = I \otimes I \otimes I \otimes \dots`.

    For example:

    - For one qubit, we have 3 operators :math:`P_i \in \{X, Y, Z\}`.
    - For two qubits, we have 15 operators :math:`P_i \in \{ I\otimes X, I\otimes Y,
      I\otimes Z, X\otimes I, Y\otimes I, Z\otimes I, X\otimes X, X\otimes Y, X\otimes Z,
      Y\otimes X, Y\otimes Y, Y\otimes Z, Z\otimes X, Z\otimes Y, Z\otimes Z \}`.

    This channel is a mixed unitary channel (see :func:`ismixedunitary`),
    and is a special case of :class:`PauliNoise`.

    See Also:
        :class:`PauliString`
        :class:`PauliNoise`

    Parameters:
        N (int): Number of qubits.
        p (float): Probability of error, i.e., the probability of not applying the identity.

    Examples:
        Depolarizing channels can be defined for any number of qubits:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Depolarizing(1, 0.1), 1)
        2-qubit circuit with 1 instructions:
        └── Depolarizing(0.1) @ q[1]
        <BLANKLINE>

        >>> c.push(Depolarizing(5, 0.1), 1, 2, 3, 4, 5)
        6-qubit circuit with 2 instructions:
        ├── Depolarizing(0.1) @ q[1]
        └── Depolarizing(0.1) @ q[1,2,3,4,5]
        <BLANKLINE>

        For one and two qubits, you can use the shorthand notation:

        >>> c.push(Depolarizing(1, 0.1), 1)
        6-qubit circuit with 3 instructions:
        ├── Depolarizing(0.1) @ q[1]
        ├── Depolarizing(0.1) @ q[1,2,3,4,5]
        └── Depolarizing(0.1) @ q[1]
        <BLANKLINE>

        >>> c.push(Depolarizing(2, 0.1), 1, 2)
        6-qubit circuit with 4 instructions:
        ├── Depolarizing(0.1) @ q[1]
        ├── Depolarizing(0.1) @ q[1,2,3,4,5]
        ├── Depolarizing(0.1) @ q[1]
        └── Depolarizing(0.1) @ q[1,2]
        <BLANKLINE>
    """

    _name = "Depolarizing"
    _parnames = ()
    _num_qubits = None
    _parnames = ()

    def __init__(self, N, p):
        if N < 1:
            raise ValueError("Cannot define a 0-qubit depolarizing noise channel")
        if not isinstance(p, (se.Basic, sp.Basic)) and (p < 0 or p > 1):
            raise ValueError("Probability p needs to be between 0 and 1.")
        self.N = N
        self.p = p
        self._num_qubits = self.N
        self._parnames = ("N", "p")

    @property
    def parnames(self):
        return self._parnames

    @property
    def num_qubits(self):
        return self._num_qubits

    def evaluate(self, d={}):
        if hasattr(self.p, "subs"):
            substituted_p = self.p.subs(d)
            evaluated_p = (
                float(substituted_p.evalf())
                if substituted_p.is_number
                else substituted_p
            )
        else:
            evaluated_p = self.p

        if isinstance(evaluated_p, (float, int)) and not (0 <= evaluated_p <= 1):
            raise ValueError("Probability after evaluation must be between 0 and 1")

        return Depolarizing(self.N, evaluated_p)

    def probabilities(self):
        N = self.N
        return np.concatenate(([1 - self.p], np.repeat(self.p / (4**N - 1), 4**N - 1)))

    def unitarygates(self):
        N = self.N
        paulis = ["I", "X", "Y", "Z"]
        combinations = ["".join(p) for p in product(paulis, repeat=N)]
        return [mc.PauliString(comb) for comb in combinations]

    def unitarymatrices(self):
        return [gate.matrix() for gate in self.unitarygates()]

    def krausmatrices(self):
        probabilities = self.probabilities()
        unitary_matrices = self.unitarymatrices()
        sqrt_probs = [se.sqrt(p) for p in probabilities]
        return [
            sqrt_p * u_matrix for sqrt_p, u_matrix in zip(sqrt_probs, unitary_matrices)
        ]

    def krausoperators(self):
        return [mc.Operator(Ek) for Ek in self.krausmatrices()]

    def iswrapper(self):
        pass

    @classmethod
    def ismixedunitary(self):
        return True

    def __str__(self):
        return f"{self._name}({self.p})"


class Depolarizing1(krauschannel):
    def __new__(self, p):
        return Depolarizing(1, p)


class Depolarizing2(krauschannel):
    def __new__(self, p):
        return Depolarizing(2, p)


__all__ = ["Depolarizing", "Depolarizing1", "Depolarizing2"]
