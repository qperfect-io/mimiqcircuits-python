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

import mimiqcircuits.operations.gates.gate as mcg
from mimiqcircuits.operations.gates.standard.phase import GateP
from symengine import pi


class PhaseGradient(mcg.Gate):
    """
    Phase Gradient gate

    A phase gradient gate applies a phase shift to a quantum register of `n` qubits,
    where each computational basis state |k⟩ experiences a phase
    proportional to its integer value `k`:

    Args:
        n (int): The number of qubits in the quantum register.

    Returns:
        PhaseGradient: The PhaseGradient gate.

    Attributes:
        name (str): The name of the operation.
        num_qubits (int): The number of qubits in the quantum register.
        qregsizes (list of int): The sizes of the quantum registers.

    Examples:
        >>> from mimiqcircuits import *
        >>> c=Circuit()
        >>> c.push(PhaseGradient(2),9,8)
            10-qubit circuit with 1 instructions:
            └── PhaseGradient @ q9, q8

    """
    _name = "PhaseGradient"
    _num_qregs = 1

    def __init__(self, num_qubits):
        super().__init__()

        self._num_qubits = num_qubits
        self._qregsizes = [num_qubits,]

        self._num_bits = 0

    def _decompose(self, circ, qubits, bits):
        q = qubits[::-1]
        for i in q:
            phase = 1 * pi / 2.0**(i - 1)
            circ.push(GateP(phase), q[i])
        return circ
    
    def matrix(self):
        raise NotImplementedError(
            "Matrix representation for PhaseGradient is not implemented.")


__all__ = ['PhaseGradient']
