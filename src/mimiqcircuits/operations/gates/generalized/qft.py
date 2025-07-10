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

import mimiqcircuits.operations.gates.gate as mcg
from mimiqcircuits.operations.gates.standard.cphase import GateCP
from mimiqcircuits.operations.gates.standard.hadamard import GateH
import mimiqcircuits.lazy as lz
from symengine import pi


class QFT(mcg.Gate):
    """
    Quantum Fourier transform.

    Performs the quantum Fourier transform on a register of `n` qubits.

    Args:
        n (int): The number of qubits in the quantum register.

    Raises:
        ValueError: If the number of qubits is less than 1.

    Returns:
        QFT: The Quantum Fourier Transform operation.

    Attributes:
        name (str): The name of the operation.
        num_qubits (int): The number of qubits in the quantum register.
        qregsizes (list of int): The sizes of the quantum registers.

    Examples:

        >>> from mimiqcircuits import *
        >>> c=Circuit()
        >>> c.push(QFT(2),1,2)
        3-qubit circuit with 1 instructions:
        └── QFT @ q[1,2]
        <BLANKLINE>

    """

    _name = "QFT"
    _num_qregs = 1
    _num_qubits = None

    def __init__(self, num_qubits):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be greater than 0")

        super().__init__()

        self._num_qubits = num_qubits
        self._qregsizes = [num_qubits]
        self._params = [num_qubits]

    def __new__(cls, *args):
        if len(args) == 0:
            return lz.LazyExpr(QFT, lz.LazyArg())
        elif len(args) == 1:
            return object.__new__(cls)
        else:
            raise ValueError("Invalid number of arguments.")

    def _matrix(self):
        raise NotImplementedError(
            "Matrix representation for Quantum Fourier Transform is not implemented."
        )

    def _decompose(self, circ, qubits, bits, _):
        q = qubits[::-1]

        circ.push(GateH(), q[0])

        for i in range(1, self.num_qubits):
            for j in range(i):
                angle = pi / (2.0 ** (i - j))
                circ.push(GateCP(angle), q[i], q[j])

            circ.push(GateH(), q[i])

        return circ


__all__ = ["QFT"]
