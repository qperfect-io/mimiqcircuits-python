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
from typing import Union
import mimiqcircuits as mc
from functools import reduce


def convert_pauli(x: str) -> mc.Gate:
    if x == "X":
        return mc.GateX()
    elif x == "Z":
        return mc.GateZ()
    elif x == "Y":
        return mc.GateY()
    elif x == "I":
        return mc.GateID()
    else:
        raise ValueError(f"Not a valid Pauli character '{x}'")


class PauliString(mc.Gate):
    """PauliString.

    N-qubit tensor product of Pauli operators.

    The PauliString gate can represent any N-qubit tensor product of operators
    of the form
    P_1 ⊗ P_2 ⊗ P_3 ⊗ ... ⊗ P_N,
    where each P_i ∈ { I, X, Y, Z } is a Pauli operator, including the identity.

    This gate can be initialized by passing as argument a string of length N
    where each element is either 'I', 'X', 'Y', or 'Z'.
    For example, PauliString("IXXYZ")

    Examples:
        >>> from mimiqcircuits import *
        >>> g= PauliString("XYZ")
        >>> g.matrix()
        [0, 0, 0, 0, 0, 0, -0.0 - 1.0*I, 0]
        [0, 0, 0, 0, 0, 0, 0, 0.0 + 1.0*I]
        [0, 0, 0, 0, 0.0 + 1.0*I, 0, 0, 0]
        [0, 0, 0, 0, 0, -0.0 - 1.0*I, 0, 0]
        [0, 0, -0.0 - 1.0*I, 0, 0, 0, 0, 0]
        [0, 0, 0, 0.0 + 1.0*I, 0, 0, 0, 0]
        [0.0 + 1.0*I, 0, 0, 0, 0, 0, 0, 0]
        [0, -0.0 - 1.0*I, 0, 0, 0, 0, 0, 0]
        <BLANKLINE>
        >>> c= Circuit()
        >>> c.push(PauliString("XYZ"),0,1,2)
        3-qubit circuit with 1 instructions:
        └── XYZ @ q[0,1,2]
        <BLANKLINE>
        >>> c.decompose()
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
        return cls(pauli_expr)

    def __str__(self):
        return f"PauliString({self.pauli})"

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def parnames(self):
        return self._parnames

    def inverse(self):
        # Inverse of a Pauli string is itself
        return PauliString(self.pauli)

    def _matrix(self):
        # Generate the matrix representation of the Pauli string
        ops = [convert_pauli(self.pauli[i]).matrix() for i in range(self.num_qubits)]
        return reduce(np.kron, ops)

    def unwrapped_matrix(self):
        # Return the matrix representation of the Pauli string
        return self._matrix()

    def is_identity(self):
        # Check if the Pauli string represents the identity operator
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
        tail = " ⊗ "
        pauli_str = tail.join(self.pauli)
        return f"{self.pauli}"

    def evaluate(self, d):
        return self

    def isidentity(self):
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
