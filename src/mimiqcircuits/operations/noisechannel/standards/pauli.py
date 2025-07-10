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
from typing import List, Union
import numpy as np
import symengine as se
import sympy as sp
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString


class PauliNoise(krauschannel):
    r"""N-qubit Pauli noise channel specified by a list of probabilities and Pauli
    gates.

    A Pauli channel is defined by:

    .. math::
        \mathcal{E}(\rho) = \sum_k p_k P_k \rho P_k,

    where :math:`0 \leq p_k \leq 1` and :math:`P_k` are Pauli string operators,
    defined as tensor products of one-qubit Pauli operators. The probabilities must fulfill
    :math:`\sum_k p_k = 1`.

    This channel is a mixed unitary channel (see :func:`ismixedunitary`).

    See Also:
        :class:`Depolarizing`, :class:`PauliX`, :class:`PauliY`, :class:`PauliZ`,
        which are special cases of PauliNoise.

    Parameters:
        p (list): List of probabilities that must add up to 1.
        paulistrings (list): List of strings, each of length :math:`N`, with each character
            being either `"I"`, `"X"`, `"Y"`, or `"Z"`. The number of qubits is equal to :math:`N`.

    The lengths of `p` and `paulistrings` must be the same.

    Examples:
        PauliNoise channels can be defined for any number of qubits,
        and for any number of Pauli strings:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(PauliNoise([0.8, 0.1, 0.1], ["I", "X", "Y"]), 1)
        2-qubit circuit with 1 instructions:
        └── PauliNoise((0.8, pauli"I"), (0.1, pauli"X"), (0.1, pauli"Y")) @ q[1]
        <BLANKLINE>

        >>> c.push(PauliNoise([0.9, 0.1], ["XY", "II"]), 1, 2)
        3-qubit circuit with 2 instructions:
        ├── PauliNoise((0.8, pauli"I"), (0.1, pauli"X"), (0.1, pauli"Y")) @ q[1]
        └── PauliNoise((0.9, pauli"XY"), (0.1, pauli"II")) @ q[1]
        <BLANKLINE>

        >>> c.push(PauliNoise([0.5, 0.2, 0.2, 0.1], ["IXIX", "XYXY", "ZZZZ", "IXYZ"]), 1, 2, 3, 4)
        5-qubit circuit with 3 instructions:
        ├── PauliNoise((0.8, pauli"I"), (0.1, pauli"X"), (0.1, pauli"Y")) @ q[1]
        ├── PauliNoise((0.9, pauli"XY"), (0.1, pauli"II")) @ q[1]
        └── PauliNoise((0.5, pauli"IXIX"), (0.2, pauli"XYXY"), (0.2, pauli"ZZZZ"), (0.1, pauli"IXYZ")) @ q[1]
        <BLANKLINE>
    """

    _name = "PauliNoise"
    _num_qubits = None
    _qregsizes = [1]
    _parnames = ()

    def __init__(self, p: List[Union[float, int]], paulistr: List[str]):
        super().__init__()

        if len(paulistr) == 0:
            raise ValueError("List of Pauli strings must contain at least one element.")

        if len(p) != len(paulistr):
            raise ValueError(
                "Lists of probabilities and Pauli strings must have the same length."
            )

        # Convert strings to PauliString instances, which will raise errors if invalid
        self.paulistr = [PauliString(s) for s in paulistr]

        self._num_qubits = self.paulistr[0].num_qubits

        if self._num_qubits < 1:
            raise ValueError("Cannot define a 0-qubit Pauli noise channel.")

        for pauli_string in self.paulistr:
            if pauli_string.num_qubits != self._num_qubits:
                raise ValueError("All Pauli strings must be of the same length.")

        # Check if any probability is symbolic; skip range and sum checks if so
        if not any(isinstance(prob, (se.Basic, sp.Basic)) for prob in p):
            if not np.isclose(sum(p), 1, rtol=1e-8):
                raise ValueError("List of probabilities should add up to 1.")

            if not all(0 <= prob <= 1 for prob in p):
                raise ValueError("All probabilities must be between 0 and 1.")

        self.p = p
        self._parnames = ("p", "paulistr")

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def parnames(self):
        return self._parnames

    def evaluate(self, d={}):
        # Substitute and evaluate each probability expression if possible
        evaluated_p = [
            (
                float(prob.subs(d).evalf())
                if hasattr(prob, "subs") and prob.subs(d).is_number
                else prob.subs(d) if hasattr(prob, "subs") else prob
            )
            for prob in self.p
        ]

        # Range check: Only perform the check if `prob` is numeric
        for prob in evaluated_p:
            if isinstance(prob, (int, float)) and (prob < 0 or prob > 1):
                raise ValueError(
                    "All numeric probabilities must be between 0 and 1 after evaluation."
                )

        # Sum check: Only perform if all probabilities are numeric
        numeric_probs = [prob for prob in evaluated_p if isinstance(prob, (int, float))]
        if len(numeric_probs) == len(evaluated_p) and not np.isclose(
            sum(numeric_probs), 1, rtol=1e-8
        ):
            raise ValueError(
                "Numeric probabilities should add up to 1 after evaluation."
            )

        # Return a new PauliNoise instance with the evaluated probabilities and the same Pauli strings
        return PauliNoise(evaluated_p, [str(s) for s in self.paulistr])

    def krausmatrices(self):
        probabilities = np.sqrt(self.probabilities())
        unitary_matrices = self.unitarymatrices()
        # Element-wise multiplication
        return [
            se.Matrix((prob * unitary).tolist())
            for prob, unitary in zip(probabilities, unitary_matrices)
        ]

    def krausoperators(self):
        probabilities = np.sqrt(self.probabilities())
        pauli_strings = self.unitarygates()
        return [
            mc.RescaledGate(pauli, prob)
            for pauli, prob in zip(pauli_strings, probabilities)
        ]

    def probabilities(self):
        return self.p

    def unitarymatrices(self):
        return [
            se.Matrix(pauli_str.unwrapped_matrix().tolist())
            for pauli_str in self.paulistr
        ]

    def unitarygates(self):
        return [mc.PauliString(str) for str in self.paulistr]

    def _pauli_to_matrix(self, pauli: str):
        """Convert a Pauli string to its corresponding matrix representation."""
        matrices = {
            "I": mc.GateID().matrix(),
            "X": mc.GateX().matrix(),
            "Y": mc.GateY().matrix(),
            "Z": mc.GateZ().matrix(),
        }
        result = matrices[pauli[0]]
        for char in pauli[1:]:
            result = np.kron(result, matrices[char])

        return result

    @classmethod
    def ismixedunitary(self):
        return True

    def iswrapper(self):
        return False

    def __str__(self):
        op_name = "PauliNoise"
        ops_str = ", ".join(
            f'({p}, pauli"{U}")' for p, U in zip(self.probabilities(), self.paulistr)
        )
        return f"{op_name}({ops_str})"

    def __repr__(self):
        return self.__str__()

    def unwrappedkrausmatrices(self):
        return self.krausmatrices()


class PauliX(krauschannel):
    r"""One-qubit Pauli X noise channel (bit flip error).

    This channel is defined by the Kraus operators:

    .. math::
        E_1 = \sqrt{1-p}\,I, \quad E_2 = \sqrt{p}\,X,

    where :math:`0 \leq p \leq 1`.

    This channel is a mixed unitary channel (see :func:`ismixedunitary`),
    and is a special case of :class:`PauliNoise`.

    `PauliX(p)` is equivalent to `PauliNoise([1-p, p], ["I", "X"])`.

    Parameters:
        p (float): Probability of a bit flip error, must be in the range [0, 1].

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(PauliX(0.1), 1)
        2-qubit circuit with 1 instructions:
        └── PauliX(0.1) @ q[1]
        <BLANKLINE>
    """

    _name = "PauliX"
    _num_qubits = 1
    _parnames = ()

    def __init__(self, p: Union[float, int]):
        if not isinstance(p, (se.Basic, sp.Basic)) and (p < 0 or p > 1):
            raise ValueError("Probability should be between 0 and 1.")
        self.p = p
        super().__init__()
        self._parnames = ("p",)

    def krausmatrices(self):
        probabilities = np.sqrt(self.probabilities())
        unitary_matrices = self.unitarymatrices()
        # Element-wise multiplication
        return [
            se.Matrix((prob * unitary).tolist())
            for prob, unitary in zip(probabilities, unitary_matrices)
        ]

    def evaluate(self, d={}):
        # Substitute and evaluate `self.p` using the provided dictionary `d`
        evaluated_p = (
            float(self.p.subs(d).evalf())
            if hasattr(self.p, "subs") and self.p.subs(d).is_number
            else self.p.subs(d) if hasattr(self.p, "subs") else self.p
        )

        # Range check: Only perform if `evaluated_p` is numeric
        if isinstance(evaluated_p, (int, float)) and (
            evaluated_p < 0 or evaluated_p > 1
        ):
            raise ValueError("Probability p must be between 0 and 1 after evaluation.")

        # Return a new instance of the same class with the evaluated probability
        return type(self)(evaluated_p)

    def krausoperators(self):
        return [mc.Operator(Ek) for (Ek) in self.krausmatrices()]

    def probabilities(self):
        return [1 - self.p, self.p]

    def unitarymatrices(self):
        return [Uk.matrix() for Uk in self.unitarygates()]

    def unitarygates(self):
        return [mc.GateID(), mc.GateX()]

    @staticmethod
    def ismixedunitary():
        return True

    def __str__(self):
        return f"{self._name}({self.p})"


class PauliY(krauschannel):
    r"""One-qubit Pauli Y noise channel (bit-phase flip error).

    This channel is determined by the Kraus operators:

    .. math::
        E_1 = \sqrt{1-p}\,I, \quad E_2 = \sqrt{p}\,Y,

    where :math:`0 \leq p \leq 1`.

    This channel is a mixed unitary channel (see :func:`ismixedunitary`),
    and is a special case of :class:`PauliNoise`.

    `PauliY(p)` is equivalent to `PauliNoise([1-p, p], ["I", "Y"])`.

    Parameters:
        p (float): Probability of a bit-phase flip error, must be in the range [0, 1].

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(PauliY(0.1), 1)
        2-qubit circuit with 1 instructions:
        └── PauliY(0.1) @ q[1]
        <BLANKLINE>
    """

    _name = "PauliY"
    _num_qubits = 1
    _parnames = ()

    def __init__(self, p: Union[float, int]):
        if not isinstance(p, (se.Basic, sp.Basic)) and (p < 0 or p > 1):
            raise ValueError("Probability should be between 0 and 1.")
        self.p = p
        super().__init__()
        self._parnames = "p"

    def evaluate(self, d={}):
        evaluated_p = (
            float(self.p.subs(d).evalf())
            if hasattr(self.p, "subs") and self.p.subs(d).is_number
            else self.p.subs(d) if hasattr(self.p, "subs") else self.p
        )

        if isinstance(evaluated_p, (int, float)) and (
            evaluated_p < 0 or evaluated_p > 1
        ):
            raise ValueError("Probability p must be between 0 and 1 after evaluation.")

        return type(self)(evaluated_p)

    def krausmatrices(self):
        probabilities = np.sqrt(self.probabilities())
        unitary_matrices = self.unitarymatrices()
        # Element-wise multiplication
        return [
            se.Matrix((prob * unitary).tolist())
            for prob, unitary in zip(probabilities, unitary_matrices)
        ]

    def krausoperators(self):
        return [mc.Operator(Ek) for (Ek) in self.krausmatrices()]

    def probabilities(self):
        return [1 - self.p, self.p]

    def unitarymatrices(self):
        return [Uk.matrix() for Uk in self.unitarygates()]

    def unitarygates(self):
        return [mc.GateID(), mc.GateY()]

    @staticmethod
    def ismixedunitary():
        return True

    def __str__(self):
        return f"{self._name}({self.p})"


class PauliZ(krauschannel):
    r"""One-qubit Pauli Z noise channel (phase flip error).

    This channel is determined by the Kraus operators:

    .. math::
        E_1 = \sqrt{1-p}\,I, \quad E_2 = \sqrt{p}\,Z,

    where :math:`0 \leq p \leq 1`.

    This channel is a mixed unitary channel (see :func:`ismixedunitary`),
    and is a special case of :class:`PauliNoise`.

    `PauliZ(p)` is equivalent to `PauliNoise([1-p, p], ["I", "Z"])`.

    Parameters:
        p (float): Probability of a phase flip error, must be in the range [0, 1].

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(PauliZ(0.1), 1)
        2-qubit circuit with 1 instructions:
        └── PauliZ(0.1) @ q[1]
        <BLANKLINE>
    """

    _name = "PauliZ"
    _num_qubits = 1
    _parnames = ()

    def __init__(self, p: Union[float, int]):
        if not isinstance(p, (se.Basic, sp.Basic)) and (p < 0 or p > 1):
            raise ValueError("Probability should be between 0 and 1.")
        self.p = p
        super().__init__()
        self._parnames = "p"

    def evaluate(self, d={}):
        evaluated_p = (
            float(self.p.subs(d).evalf())
            if hasattr(self.p, "subs") and self.p.subs(d).is_number
            else self.p.subs(d) if hasattr(self.p, "subs") else self.p
        )

        if isinstance(evaluated_p, (int, float)) and (
            evaluated_p < 0 or evaluated_p > 1
        ):
            raise ValueError("Probability p must be between 0 and 1 after evaluation.")

        return type(self)(evaluated_p)

    def krausmatrices(self):
        probabilities = np.sqrt(self.probabilities())
        unitary_matrices = self.unitarymatrices()
        # Element-wise multiplication
        return [
            se.Matrix((prob * unitary).tolist())
            for prob, unitary in zip(probabilities, unitary_matrices)
        ]

    def krausoperators(self):
        return [mc.Operator(Ek) for (Ek) in self.krausmatrices()]

    def probabilities(self):
        return [1 - self.p, self.p]

    def unitarymatrices(self):
        return [Uk.matrix() for Uk in self.unitarygates()]

    def unitarygates(self):
        return [mc.GateID(), mc.GateZ()]

    @staticmethod
    def ismixedunitary():
        return True

    def __str__(self):
        return f"{self._name}({self.p})"


__all__ = ["PauliNoise", "PauliX", "PauliY", "PauliZ"]
