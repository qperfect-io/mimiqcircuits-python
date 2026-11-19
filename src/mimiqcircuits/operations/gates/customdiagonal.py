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

from mimiqcircuits.operations.gates.generalized.rnz import GateRNZ
from mimiqcircuits.operations.gates.standard.rotations import GateRZ
from mimiqcircuits.operations.gates.standard.u import GateU
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
    r"""Gate diagonal in the computational basis, given by its diagonal.

    Same role as :class:`GateCustom`, but only the :math:`2^n` diagonal
    entries are stored, so a wide diagonal gate costs :math:`2^n` numbers
    instead of :math:`4^n`. The entries of a numeric diagonal must be phases,
    ``abs(d) == 1`` within ``1e-8``; a diagonal with a symbolic entry is not
    checked.

    The entries follow the diagonal of the dense matrix, in the basis
    ``|0...0>, ..., |1...1>`` with the first target qubit as the most
    significant bit, the same convention as :class:`GateCustom`.

    Args:
        diagonal (list, tuple, numpy.ndarray or symengine.Matrix): The
            :math:`2^n` diagonal entries, :math:`n \geq 1`.

    Raises:
        TypeError: If ``diagonal`` is not one of the accepted types.
        ValueError: If the number of entries is not a power of two greater
            than one, or a numeric entry does not have modulus one.

    Examples:
        >>> from mimiqcircuits import *
        >>> # the diagonal of a CZ gate
        >>> g = GateCustomDiagonal([1, 1, 1, -1])
        >>> g.num_qubits
        2
        >>> c = Circuit()
        >>> c.push(g, 0, 1)
        2-qubit circuit with 1 instruction:
        └── CustomDiagonal(...) @ q[0:1]
        <BLANKLINE>
        >>> # the decomposition uses one Z-parity rotation per Walsh term
        >>> g.decompose()
        2-qubit circuit with 4 instructions:
        ├── RZ(1.5707963267948966) @ q[1]
        ├── RZ(1.5707963267948966) @ q[0]
        ├── RNZ(-1.5707963267948966) @ q[0:1]
        └── U(0, 0, 0, 0.7853981633974483) @ q[0]
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
        fixed = numeric
        if fixed is None:
            fixed = np.array([complex(e) for e in d if not e.free_symbols])
        if not np.allclose(np.abs(fixed), 1.0, rtol=0, atol=1e-8):
            raise ValueError("Diagonal is not unitary")
        if numeric is not None:
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

    # Overridden for the same reason as GateCustom.matrix: the inherited
    # version caches on the class, so every instance would share one matrix.
    def matrix(self):
        r"""Return the dense diagonal matrix as a SymEngine matrix.

        Entries that can be converted to a complex number are returned as
        complex numbers; symbolic entries are kept as they are.

        Returns:
            symengine.Matrix: A new :math:`2^n \times 2^n` matrix on each call.
        """
        return se.diag(*self.diagonal())

    def diagonal(self):
        """Return the diagonal entries as a list.

        Entries that can be converted to a complex number are returned as
        ``complex``; symbolic entries are kept as they are.
        """
        out = []
        for x in self._d:
            try:
                out.append(complex(x))
            except Exception:
                out.append(x)
        return out

    def unwrappeddiagonal(self):
        """Return the diagonal as a ``complex128`` NumPy array.

        Raises:
            TypeError: If any entry is still symbolic.
        """
        return _as_numpy_numeric(self._d).ravel()

    def unwrappedmatrix(self):
        """Return the dense matrix as a ``complex128`` NumPy array.

        Raises:
            TypeError: If any entry is still symbolic.
        """
        return np.diag(self.unwrappeddiagonal())

    @property
    def num_qubits(self):
        r"""Number of qubits the gate acts on, :math:`\log_2` of the diagonal length."""
        return self._num_qubits

    # The diagonal entries are the parameters, named by their index.
    # `_parnames` stays empty for the same reason as in
    # `GateCustom`.
    @property
    def parnames(self):
        """Parameter names: the 1-based indices of the diagonal entries."""
        return tuple(range(1, len(self._d) + 1))

    def getparams(self):
        """Return the stored diagonal entries as a list."""
        return list(self._d)

    def getparam(self, pn):
        """Return the diagonal entry with 1-based index ``pn``.

        Raises:
            IndexError: If ``pn`` is outside ``1..2**n``.
        """
        # 1-based; a plain index would let 0 and
        # negative values wrap around to the last entries.
        if not 1 <= pn <= len(self._d):
            raise IndexError(f"parameter index must be in 1..{len(self._d)}, not {pn}")
        return self._d[pn - 1]

    def inverse(self):
        return GateCustomDiagonal([se.conjugate(x) for x in self._d])

    def __repr__(self):
        return self.pretty_print()

    def __str__(self):
        return f"{self._name}(...)"

    def pretty_print(self):
        """Return a multi-line string listing the diagonal entries."""
        result = f"{self._num_qubits}-qubit GateCustomDiagonal:\n"
        entries = list(self._d)
        for i, entry in enumerate(entries):
            result += "├── " if i < len(entries) - 1 else "└── "
            result += str(entry)
            if i < len(entries) - 1:
                result += "\n"
        return result

    def evaluate(self, d):
        """Substitute symbols in the diagonal entries and return a new gate.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            GateCustomDiagonal: ``self`` if no entry is symbolic, otherwise a
            new gate built from the substituted entries.
        """
        if not self.is_symbolic():
            return self
        import sympy as sp

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
                circ.push(GateRZ(-2 * alpha[s]), qs[0])
            else:
                circ.push(GateRNZ(len(qs), -2 * alpha[s]), *qs)

        # alpha[0] is the global phase left over
        if abs(alpha[0]) >= 1e-14:
            circ.push(GateU(0, 0, 0, alpha[0]), qubits[0])

        return circ


__all__ = ["GateCustomDiagonal"]
