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


import mimiqcircuits.operations.gates.gate as mcg
import mimiqcircuits as mc
from mimiqcircuits.lazy import LazyArg, LazyExpr
from symengine import eye
from mimiqcircuits.circuit import Circuit


class GateRNZ(mcg.Gate):
    r"""
    Multi-qubit parity-phase rotation gate along the Z-axis.

    Applies a phase of ``e^{±i θ/2}`` depending on the parity (number of 1s) in the computational basis state.

    Args:
        n (int): Number of qubits
        theta (float or symbolic): Rotation angle

    Examples:

    >>> from mimiqcircuits import *
    >>> from symengine import pi
    >>> GateRNZ(2, pi/2)
    RNZ((1/2)*pi)

    >>> GateRNZ()
    lazy GateRNZ(?, ?)

    >>> GateRNZ(2)
    lazy GateRNZ(2, ?)

    >>> GateRNZ(2, pi/2).decompose()
    2-qubit circuit with 3 instructions:
    ├── CX @ q[0], q[1]
    ├── RZ((1/2)*pi) @ q[1]
    └── CX @ q[0], q[1]
    <BLANKLINE>

    >>> c = Circuit()
    >>> c.push(GateRNZ(3, pi/2), 1, 2, 3)
    4-qubit circuit with 1 instructions:
    └── RNZ((1/2)*pi) @ q[1,2,3]
    <BLANKLINE>
    """

    _name = "RNZ"
    _num_bits = 0
    _num_cregs = 0
    _num_zregs = 0
    _num_zvars = 0
    _num_qregs = 1
    _zregsizes = []
    _cregsizes = []
    _parnames = ("theta",)

    def __new__(cls, *args):
        if len(args) == 0:
            return LazyExpr(cls, LazyArg(), LazyArg())
        elif len(args) == 1:
            return LazyExpr(cls, args[0], LazyArg())
        elif len(args) == 2:
            n, theta = args
            if isinstance(n, int):
                return object.__new__(cls)
        raise ValueError("Invalid arguments for GateRNZ. Expected, (n, theta).")

    def __init__(self, n=None, theta=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("GateRNZ expects a positive integer for number of qubits.")

        self._num_qubits = n
        self._num_qregs = 1
        self._qregsizes = [n]
        self.theta = theta

    def __str__(self):
        return f"RNZ({self.theta})"

    def inverse(self):
        return type(self)(self.num_qubits, -self.theta)

    def power(self, pwr):
        return type(self)(self.num_qubits, self.theta * pwr)

    def matrix(self):
        from mimiqcircuits.matrices import cis

        θ = self.theta
        N = self.num_qubits
        dim = 1 << N
        mat = eye(dim)
        for i in range(dim):
            parity = bin(i).count("1")
            mat[i, i] = cis((-((-1) ** parity)) * θ / 2)
        return mat

    def _decompose(self, circ: Circuit, qtargets, ctargets=None, ztargets=None):
        ancilla = qtargets[-1]
        data = qtargets[:-1]

        for q in data:
            circ.push(mc.GateCX(), q, ancilla)

        circ.push(mc.GateRZ(self.theta), ancilla)

        for q in reversed(data):
            circ.push(mc.GateCX(), q, ancilla)

        return circ

    def evaluate(self, d):
        """
        Evaluate symbolic parameters with substitution dictionary d.

        Args:
            d (dict): Mapping of symbolic variables to values.

        Returns:
            GateRNZ: A new evaluated GateRNZ instance.
        """
        theta = self.theta
        if hasattr(theta, "subs"):
            theta = theta.subs(d)
        return type(self)(self.num_qubits, theta)


__all__ = ["GateRNZ"]
