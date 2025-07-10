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
from mimiqcircuits.operations.gates.gate import Gate
import mimiqcircuits as mc
from scipy.linalg import expm
import symengine as se
import sympy as sp
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString


class RPauli(Gate):
    r"""
    Apply a Pauli string rotation gate of the form:

    .. math::
        e^{-i \frac{\theta}{2} (P_1 \otimes P_2 \otimes \dots)}

    where each :math:`P_i` is a Pauli operator from ``"I"``, ``"X"``, ``"Y"``, or ``"Z"``, acting on a separate qubit.

    This gate represents the time evolution under a tensor product of Pauli operators,
    which is useful for simulating Hamiltonian dynamics, variational ansätze, and Trotter steps.

    Identity-only strings or zero rotation angles are optimized away as no-ops.

    Args:
        pauli (PauliString): A tensor product of Pauli operators such as ``"X"``, ``"ZIZ"``, or ``"IXY"``.
        theta (float or symengine.Basic): The rotation angle.

    .. note::
        The operator acts on all qubits where the Pauli string is non-identity. It decomposes into native gates including Hadamards, RZs, and CNOTs.

    Examples:

    >>> from mimiqcircuits import *
    >>> from symengine import pi
    >>> RPauli(PauliString("X"), pi/2)
    R("X", (1/2)*pi)

    >>> RPauli(PauliString("ZIZ"), 2.0)
    R("ZIZ", 2.0)

    >>> c = Circuit()
    >>> c.push(RPauli(PauliString("YXI"), 2.0), 1, 2, 3)
    4-qubit circuit with 1 instructions:
    └── R("YXI", 2.0) @ q[1,2,3]
    <BLANKLINE>


    >>> c.push(RPauli(PauliString("III"), pi), 1, 2, 3)
    4-qubit circuit with 2 instructions:
    ├── R("YXI", 2.0) @ q[1,2,3]
    └── R("III", pi) @ q[1,2,3]
    <BLANKLINE>


    >>> c.push(RPauli(PauliString("IXY"), pi), 1, 2, 3)
    4-qubit circuit with 3 instructions:
    ├── R("YXI", 2.0) @ q[1,2,3]
    ├── R("III", pi) @ q[1,2,3]
    └── R("IXY", pi) @ q[1,2,3]
    <BLANKLINE>


    See Also:
        :class:`PauliString`, :class:`GateRZ`

    """

    _name = "RPauli"
    _num_bits = 0
    _num_cregs = 0
    _num_zregs = 0
    _num_zvars = 0
    _zregsizes = []
    _cregsizes = []
    _parnames = ("theta",)

    def __init__(self, pauli: PauliString, theta):

        self.pauli = pauli
        self.theta = theta
        super().__init__()
        self._num_qregs = 1
        self._num_qubits = len(pauli.pauli)
        self._qregsizes = [len(pauli.pauli)]

    def isidentity(self):
        if all(p == "I" for p in str(self.pauli)):
            return True

        theta = self.theta
        if isinstance(theta, (int, float, complex)):
            return theta == 0
        elif hasattr(theta, "evalf") and theta == 0:
            return True

        return False

    def isunitary(self):
        return True

    def iswrapper(self):
        return True

    def _decompose(self, circ: mc.Circuit, qtargets, ctargets=None, ztargets=None):
        s = str(self.pauli)

        if len(s) != len(qtargets):
            raise ValueError("Length of Pauli string and qubit list must match.")

        active = [(p, q) for p, q in zip(s, qtargets) if p != "I"]
        active_qubits = [q for _, q in active]

        if not active:
            circ.push(mc.GateU(0, 0, 0, -self.theta / 2), max(qtargets))
            return circ

        if max(qtargets) not in active_qubits:
            circ.push(mc.GateID(), max(qtargets))

        for p, q in active:
            if p == "X":
                circ.push(mc.GateH(), q)
            elif p == "Y":
                circ.push(mc.GateHYZ(), q)

        # Apply RNZ on active qubits
        circ.push(mc.GateRNZ(len(active_qubits), self.theta), *active_qubits)


        for p, q in reversed(active):
            if p == "X":
                circ.push(mc.GateH(), q)
            elif p == "Y":
                circ.push(mc.GateHYZ(), q)

        return circ

    def __str__(self):
        return f'R("{str(self.pauli)}", {self.theta})'

    def matrix(self):
        from mimiqcircuits.matrices import symbolic_matrix_exponential

        theta = self.theta
        mat_sym = self.pauli.matrix()

        # Use symbolic exponential if theta is symbolic
        if isinstance(theta, (se.Basic, sp.Basic)):
            return symbolic_matrix_exponential(mat_sym, theta)

        # Use numerical exponential otherwise
        mat_np = np.array(mat_sym.tolist(), dtype=complex)
        mat_exp = expm(-1j * theta / 2 * mat_np)
        return se.Matrix(mat_exp.tolist())

    def evaluate(self, d):
        """
        Substitute values into theta if symbolic. Return new RPauli with updated parameter.

        Parameters:
        - d (dict): Dictionary of substitutions {symbol: value}

        Returns:
        - RPauli: Evaluated RPauli gate
        """
        theta = self.theta
        if hasattr(theta, "subs"):
            theta = theta.subs(d)

        return type(self)(self.pauli, theta)


__all__ = ["RPauli"]
