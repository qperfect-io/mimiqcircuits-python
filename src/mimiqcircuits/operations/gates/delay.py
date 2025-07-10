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

from mimiqcircuits import Gate, GateID, Circuit


class Delay(Gate):
    r"""
    Delay(t)

    1-qubit delay gate

    This gate is equivalent to a GateID gate, except that it is parametrized by
    a time parameter `t`. The parameter does not affect the action of the gate. The only
    purpose of this gate is to act as a placeholder to indicate idle noise, in
    which case the parameter `t` can later be used to further specify the noise properties.

    The gate can be created by calling `Delay(t)` where `t` is a number.

    **Matrix representation:**

    .. math::
        \operatorname{Delay}(t) =
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}


    Examples:
        >>> Delay(0.3)
        Delay(0.3)

        >>> delay_gate = Delay(0.1)
        >>> delay_gate.matrix()
        [1.0, 0]
        [0, 1.0]
        <BLANKLINE>

        >>> c = Circuit()
        >>> c.push(delay_gate, 0)
        1-qubit circuit with 1 instructions:
        └── Delay(0.1) @ q[0]
        <BLANKLINE>

        Decomposition

        >>> delay_gate = Delay(0.2)
        >>> delay_gate.decompose()
        1-qubit circuit with 1 instructions:
        └── ID @ q[0]
        <BLANKLINE>
    """

    _name = "Delay"
    _num_qubits = 1
    _num_qregs = 1
    _num_cregs = 0
    _num_bits = 0
    _parnames = "t"

    def __init__(self, t: float = 1):
        self.t = t
        super().__init__()

    def __str__(self):
        return f"Delay({self.t})"

    def _matrix(self):
        """Return the matrix representation of the Delay gate."""
        return GateID().matrix()

    def inverse(self):
        raise ValueError("Cannot invert a Delay gate.")

    def power(self, _):
        raise ValueError("Cannot raise a Delay gate to any power.")

    def _decompose(self, circ: Circuit, qubits, bitsm, zvars):
        """Decompose the Delay gate into a GateID for the given circuit."""
        for qubit in qubits:
            circ.push(GateID(), [qubit])
        return circ


__all__ = ["Delay"]
