#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
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
"""Delay gate operation."""

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.gates.standard.id import GateID


class Delay(Gate):
    r"""Single qubit delay gate.

    Acts as the identity. The duration ``t`` does not change the action of
    the gate: it marks idle time, so a noise model can attach idle noise to
    it and use ``t`` to set its strength. A delay is never rewritten into
    other gates by canonical rewriting, so the duration is kept. It cannot be
    inverted or raised to a power.

    Matrix Representation:
        .. math::
            \operatorname{Delay}(t) =
            \begin{pmatrix}
                1 & 0 \\
                0 & 1
            \end{pmatrix}

    Args:
        t: Duration of the delay. Defaults to ``1``.

    Examples:
        >>> from mimiqcircuits import *
        >>> Delay(0.3)
        Delay(0.3)
        >>> Delay(0.1).matrix()
        [1.0, 0]
        [0, 1.0]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(Delay(0.1), 0)
        1-qubit circuit with 1 instruction:
        └── Delay(0.1) @ q[0]
        <BLANKLINE>
        >>> Delay(0.2).decompose()
        1-qubit circuit with 1 instruction:
        └── ID @ q[0]
        <BLANKLINE>
    """

    _name = "Delay"
    _num_qubits = 1
    _num_qregs = 1
    _num_cregs = 0
    _num_bits = 0
    _parnames = ("t",)

    def __init__(self, t: float = 1):
        self.t = t
        super().__init__()

    def __str__(self):
        return f"Delay({self.t})"

    def _matrix(self):
        """Return the matrix representation of the Delay gate."""
        return GateID().matrix()

    def inverse(self):
        """Raise, as a delay cannot be inverted.

        Raises:
            ValueError: Always.
        """
        raise ValueError("Cannot invert a Delay gate.")

    def power(self, _):
        """Raise, as a delay cannot be raised to a power.

        Raises:
            ValueError: Always.
        """
        raise ValueError("Cannot raise a Delay gate to any power.")

    def supports_canonical_rewrite(self):
        """Return ``False``: rewriting a delay into identity gates would lose its duration."""
        return False

    def _decompose(self, circ: Circuit, qubits, bitsm, zvars):
        """Decompose the Delay gate into a GateID for the given circuit."""
        for qubit in qubits:
            circ.push(GateID(), [qubit])
        return circ


__all__ = ["Delay"]
