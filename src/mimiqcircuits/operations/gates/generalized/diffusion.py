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

import mimiqcircuits as mc
from symengine import pi


class Diffusion(mc.Gate):
    """
    Grover's diffusion operator.

    Args:
        num_qubits (int): The number of qubits.

    Raises:
        ValueError: If the number of qubits is not an integer or less than 1.

    Returns:
        Diffusion: Grover's diffusion operator.

    Attributes:
        num_qubits (int): The number of qubits for the diffusion operator.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Diffusion(2), 1, 2)
        3-qubit circuit with 1 instructions:
        └── Diffusion @ q[1,2]
        <BLANKLINE>

    """

    _name = "Diffusion"
    _num_qregs = 1
    _num_qubits = None

    def __init__(self, num_qubits):
        if not isinstance(num_qubits, int):
            raise ValueError("Number of qubits must be an integer.")

        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")

        super().__init__()

        self._num_qubits = num_qubits
        self._qregsizes = [num_qubits]
        self._params = [num_qubits]
        self._params = [num_qubits]

    def __new__(cls, *args):
        if len(args) == 0:
            return mc.LazyExpr(Diffusion, mc.LazyArg())
        elif len(args) == 1:
            return object.__new__(cls)
        else:
            raise ValueError("Invalid number of arguments.")

    def __str__(self):
        return "Diffusion"

    def _matrix(self):
        raise NotImplementedError(
            "Matrix representation for Diffusion operator is not implemented."
        )

    def _decompose(self, circ, qubits, bits, zvars):
        circ.push(mc.GateRY(pi / 2), qubits)
        circ.push(mc.control(self.num_qubits - 1, mc.GateZ()), *qubits)
        circ.push(mc.GateRY(-pi / 2), qubits)
        return circ


__all__ = ["Diffusion"]
