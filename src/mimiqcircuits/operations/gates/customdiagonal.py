#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
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
"""Custom diagonal gate definition."""

import numpy as np
import symengine as se
import sympy as sp

import mimiqcircuits as mc
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.decompositions.matrix_decompositions.utils import (
    _as_numpy_numeric,
    _to_symengine_matrix,
)


def _fwht(a):
    """Fast Walsh-Hadamard transform, unnormalized."""
    a = np.array(a, dtype=float)
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2 * h):
            x = a[i : i + h].copy()
            y = a[i + h : i + 2 * h].copy()
            a[i : i + h] = x + y
            a[i + h : i + 2 * h] = x - y
        h *= 2
    return a


class GateCustomDiagonal(Gate):
    """Gate diagonal in the computational basis, given by its diagonal.

    Same role as :class:`GateCustom`, but only the ``2**n`` diagonal entries
    are stored, so a wide diagonal block costs ``2**n`` numbers instead of
    ``4**n``. Every entry must be a phase, ``abs(d) == 1``.

    The entries follow the diagonal of the dense matrix, in the basis
    ``|0...0>, ..., |1...1>`` with the first target qubit as the most
    significant bit — the same convention as :class:`GateCustom`.

    Examples:
        >>> from mimiqcircuits import Circuit, GateCustomDiagonal
        >>> c = Circuit()
        >>> c.push(GateCustomDiagonal([1, 1, 1, -1]), 0, 1)
        2-qubit circuit with 1 instruction:
        └── CustomDiagonal(...) @ q[0:1]
        <BLANKLINE>
    """

    _name = "CustomDiagonal"
    _num_qubits = None
    _qregsizes = None

    def __init__(self, diagonal):
        super().__init__()

        if isinstance(diagonal, se.Matrix):
            entries = list(diagonal)
        elif isinstance(diagonal, (np.ndarray, list, tuple)):
            entries = list(np.asarray(diagonal, dtype=object).ravel())
        else:
            raise TypeError(
                f"{type(diagonal)} not supported in GateCustomDiagonal, "
                "use a sequence, numpy.ndarray or symengine.Matrix."
            )

        num_qubits = len(entries).bit_length() - 1
        if num_qubits < 1 or len(entries) != 2**num_qubits:
            raise ValueError("Wrong number of entries for the diagonal")

        d = se.Matrix(len(entries), 1, entries)

        try:
            numeric = _as_numpy_numeric(d)
        except TypeError:
            numeric = None
        if numeric is not None:
            if not np.allclose(np.abs(numeric), 1.0, rtol=0, atol=1e-8):
                raise ValueError("Diagonal is not unitary")
            # Store numeric entries as floats, so that a diagonal given as
            # integers compares equal to the same one given as floats.
            d = _to_symengine_matrix(numeric.reshape(-1, 1))

        self._d = d
        self._num_qubits = num_qubits
        self._qregsizes = [
            num_qubits,
        ]

    def _matrix(self):
        return se.diag(*list(self._d))

    def matrix(self):
        """Dense matrix, entries evaluated numerically where possible.

        Not inherited from :class:`AbstractOperator` for the same reason as
        :meth:`GateCustom.matrix`: that version memoises on the *class* for
        parameter-free operators, which would make every instance share
        whichever matrix was built first.
        """
        return se.diag(*self.diagonal())

    def diagonal(self):
        """The stored diagonal entries, evaluated numerically where possible."""
        out = []
        for x in self._d:
            try:
                out.append(complex(x))
            except Exception:
                out.append(x)
        return out

    def unwrappeddiagonal(self):
        """The diagonal as a ``numpy.ndarray[complex128]``.

        Raises:
            TypeError: if any entry is still symbolic.
        """
        return _as_numpy_numeric(self._d).ravel()

    def unwrappedmatrix(self):
        return np.diag(self.unwrappeddiagonal())

    @property
    def num_qubits(self):
        return self._num_qubits

    def inverse(self):
        return GateCustomDiagonal([se.conjugate(x) for x in self._d])

    def __repr__(self):
        return self.pretty_print()

    def __str__(self):
        return f"{self._name}(...)"

    def pretty_print(self):
        result = f"{self._num_qubits}-qubit GateCustomDiagonal:\n"
        entries = list(self._d)
        for i, entry in enumerate(entries):
            result += "├── " if i < len(entries) - 1 else "└── "
            result += str(entry)
            if i < len(entries) - 1:
                result += "\n"
        return result

    def evaluate(self, d):
        entries = [
            (
                entry
                if isinstance(entry, (float, int))
                else sp.sympify(entry).subs(d)
            )
            for entry in self._d
        ]
        return GateCustomDiagonal([se.sympify(entry) for entry in entries])

    def _decompose(self, circ, qubits, bits, zvars):
        """Expand the phases in the Walsh basis.

        With ``phi(x) = sum_S a_S (-1)^{|x & S|}``, the gate is
        ``e^{i a_0} prod_{S != 0} exp(i a_S Z_S)``, and each parity term is one
        ``GateRNZ`` on the qubits of ``S`` (a ``GateRZ`` when ``S`` is a single
        qubit).

        Not routed through :class:`GateCustom`: that would build a ``4**n``-entry
        matrix for a gate that needs ``2**n``, and QSD would then spend a
        multiple of that many gates rediscovering a structure this expansion
        writes down directly.
        """
        n = self.num_qubits
        if len(qubits) != n:
            raise ValueError(
                f"GateCustomDiagonal expects {n} qubits, got {len(qubits)}"
            )

        alpha = _fwht(np.angle(self.unwrappeddiagonal())) / (1 << n)

        for s in range(1, 1 << n):
            if abs(alpha[s]) < 1e-14:
                continue
            # bit `n - 1 - p` of the mask selects the p-th target, matching the
            # most-significant-first order of the diagonal itself
            qs = [qubits[p] for p in range(n) if (s >> (n - 1 - p)) & 1]
            if len(qs) == 1:
                circ.push(mc.GateRZ(-2 * alpha[s]), qs[0])
            else:
                circ.push(mc.GateRNZ(len(qs), -2 * alpha[s]), *qs)

        # alpha[0] is the global phase left over
        if abs(alpha[0]) >= 1e-14:
            circ.push(mc.GateU(0, 0, 0, alpha[0]), qubits[0])

        return circ


__all__ = ["GateCustomDiagonal"]
