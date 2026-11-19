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
"""Grover diffusion operator."""

from mimiqcircuits.lazy import LazyArg, LazyExpr, control
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.gates.standard.pauli import GateZ
from mimiqcircuits.operations.gates.standard.rotations import GateRY
from symengine import pi


class Diffusion(Gate):
    r"""Grover's diffusion operator on ``num_qubits`` qubits.

    Implements the unitary

    .. math::
        H^{\otimes n} \left(1 - 2 |0^n\rangle\langle 0^n|\right) H^{\otimes n}

    ``Diffusion()`` with no arguments returns a lazy expression that takes
    the number of qubits later. The gate has no matrix representation:
    :meth:`matrix` raises ``NotImplementedError``.

    Args:
        num_qubits (int): Number of qubits, at least 1.

    Raises:
        ValueError: If ``num_qubits`` is not an integer or is less than 1, or
            more than one argument is given.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Diffusion(2), 1, 2)
        3-qubit circuit with 1 instruction:
        └── Diffusion @ q[1:2]
        <BLANKLINE>
        >>> Diffusion(2).decompose()
        2-qubit circuit with 5 instructions:
        ├── RY((1/2)*pi) @ q[0]
        ├── RY((1/2)*pi) @ q[1]
        ├── CZ @ q[0], q[1]
        ├── RY((-1/2)*pi) @ q[0]
        └── RY((-1/2)*pi) @ q[1]
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
            return LazyExpr(Diffusion, LazyArg())
        elif len(args) == 1:
            return object.__new__(cls)
        else:
            raise ValueError("Invalid number of arguments.")

    def __str__(self):
        return "Diffusion"

    def inverse(self):
        # the diffusion operator is Hermitian and unitary, so its own inverse
        return self

    def _matrix(self):
        raise NotImplementedError(
            "Matrix representation for Diffusion operator is not implemented."
        )

    def _decompose(self, circ, qubits, bits, zvars):
        circ.push(GateRY(pi / 2), qubits)
        if self.num_qubits == 1:
            circ.push(GateZ(), *qubits)
        else:
            circ.push(control(self.num_qubits - 1, GateZ()), *qubits)
        circ.push(GateRY(-pi / 2), qubits)
        return circ


__all__ = ["Diffusion"]
