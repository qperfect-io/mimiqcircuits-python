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
"""PauliString definition."""

import numpy as np
from typing import Union
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.gates.standard.id import GateID
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
from functools import reduce


def convert_pauli(x: str) -> Gate:
    """Return the single qubit gate for a Pauli character.

    Args:
        x: One of ``"I"``, ``"X"``, ``"Y"`` or ``"Z"``.

    Returns:
        The matching :class:`GateID`, :class:`GateX`, :class:`GateY` or
        :class:`GateZ`.

    Raises:
        ValueError: If ``x`` is not a Pauli character.
    """
    if x == "X":
        return GateX()
    elif x == "Z":
        return GateZ()
    elif x == "Y":
        return GateY()
    elif x == "I":
        return GateID()
    else:
        raise ValueError(f"Not a valid Pauli character '{x}'")


class PauliString(Gate):
    r"""Tensor product of Pauli operators on :math:`N` qubits.

    Represents :math:`P_1 \otimes P_2 \otimes \dots \otimes P_N` with each
    :math:`P_i \in \{I, X, Y, Z\}`. The :math:`i`-th character of the
    string acts on the :math:`i`-th target qubit. A Pauli string is its own
    inverse and can only be raised to integer powers.

    Args:
        pauli: A non-empty string of ``"I"``, ``"X"``, ``"Y"`` and ``"Z"``
            characters, for example ``"IXXYZ"``.

    Raises:
        ValueError: If the string is empty or contains other characters.

    Examples:
        >>> from mimiqcircuits import *
        >>> g = PauliString("XYZ")
        >>> g.matrix()
        [0, 0, 0, 0, 0, 0, 0.0 - 1.0*I, 0]
        [0, 0, 0, 0, 0, 0, 0, 0.0 + 1.0*I]
        [0, 0, 0, 0, 0.0 + 1.0*I, 0, 0, 0]
        [0, 0, 0, 0, 0, 0.0 - 1.0*I, 0, 0]
        [0, 0, 0.0 - 1.0*I, 0, 0, 0, 0, 0]
        [0, 0, 0, 0.0 + 1.0*I, 0, 0, 0, 0]
        [0.0 + 1.0*I, 0, 0, 0, 0, 0, 0, 0]
        [0, 0.0 - 1.0*I, 0, 0, 0, 0, 0, 0]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(PauliString("XYZ"), 0, 1, 2)
        3-qubit circuit with 1 instruction:
        └── XYZ @ q[0:2]
        <BLANKLINE>
        >>> g.power(2), g.inverse()
        (III, XYZ)
        >>> g.decompose()
        3-qubit circuit with 3 instructions:
        ├── X @ q[0]
        ├── Y @ q[1]
        └── Z @ q[2]
        <BLANKLINE>
    """

    _name = "PauliString"
    _num_qregs = 1
    _parnames = ()

    def __init__(self, pauli: Union[str, chr]):
        self.pauli = str(pauli)
        super().__init__()

        self._num_qubits = len(self.pauli)
        self._qregsizes = [
            self._num_qubits,
        ]
        self._parnames = ("pauli",)

        if self._num_qubits < 1:
            raise ValueError("Pauli string cannot be empty.")

        if any(x not in "IXYZ" for x in self.pauli):
            raise ValueError("Pauli string can only contain I, X, Y, or Z.")

    @classmethod
    def from_string(cls, pauli_expr: str):
        """Build a ``PauliString`` from a string; same as ``PauliString(pauli_expr)``."""
        return cls(pauli_expr)

    @property
    def num_qubits(self):
        """Number of qubits, the length of the string."""
        return self._num_qubits

    @property
    def parnames(self):
        """Parameter names: ``("pauli",)``."""
        return self._parnames

    def inverse(self):
        # Inverse of a Pauli string is itself
        return PauliString(self.pauli)

    def _matrix(self):
        # Generate the matrix representation of the Pauli string
        ops = [convert_pauli(self.pauli[i]).matrix() for i in range(self.num_qubits)]
        return reduce(np.kron, ops)

    def unwrappedmatrix(self):
        """Return the matrix as a NumPy array with dtype ``complex128``."""
        return np.array(self._matrix(), dtype=np.complex128)

    def unwrapped_matrix(self):
        """Return the matrix as a NumPy array of SymEngine numbers (``dtype=object``)."""
        return self._matrix()

    def is_identity(self):
        """Return ``True`` if every character is ``"I"``. Same as :meth:`isidentity`."""
        return all(p == "I" for p in self.pauli)

    def _power(self, pwr):
        # Elevate Pauli string to an integer power
        if pwr % 2 == 0:
            return PauliString("I" * self.num_qubits)
        elif pwr % 1 == 0:
            return self
        else:
            raise ValueError("Pauli strings can only be elevated to an integer power.")

    def _decompose(self, circ, qreg, _, zvars):
        # Decompose the Pauli string into individual gates
        for i, q in enumerate(qreg):
            circ.push(convert_pauli(self.pauli[i]), [q])
        return circ

    def __str__(self):
        return self.pauli

    def evaluate(self, d):
        """Return ``self``: a Pauli string has no symbolic parameters."""
        return self

    def isidentity(self):
        """Return ``True`` if every character is ``"I"``."""
        return all(p == "I" for p in self.pauli)

    def asciiwidth(self, qubits, bits, zvars):
        if len(qubits) == 1:
            return len(str(self.pauli[0])) + 2
        else:
            max_label_length = max(
                len(f"{i}: {self.pauli[i]}") for i in range(len(self.pauli))
            )
            return max_label_length + 2


__all__ = ["PauliString"]
