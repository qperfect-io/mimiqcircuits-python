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
"""Custom gate definition."""


from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.gate import Gate
import symengine as se
import numpy as np
from mimiqcircuits.operations.decompositions.matrix_decompositions.utils import  _as_numpy_numeric
from mimiqcircuits.operations.decompositions.matrix_decompositions.qsd import  _qsd_decomposition
from mimiqcircuits.operations.decompositions.matrix_decompositions.zyz import  _zyz_decomposition




class GateCustom(Gate):
    r"""Custom gate defined by a unitary matrix.

    The matrix must be square with :math:`2^n` rows, :math:`n \geq 1`; the gate
    then acts on :math:`n` qubits. Qubits are ordered big-endian: the first
    target qubit is the most significant bit of the row index. Numeric
    matrices are checked for unitarity with an absolute tolerance of
    ``1e-8``. The check is skipped when an entry is a plain symbol, so a
    symbolic matrix can be filled in later with :meth:`evaluate`.

    Args:
        matrix (numpy.ndarray or symengine.Matrix): The unitary matrix.

    Raises:
        TypeError: If ``matrix`` is neither a NumPy array nor a SymEngine
            matrix.
        ValueError: If the matrix is not square, not unitary, or its size is
            not a power of two greater than one.

    Examples:
        >>> from mimiqcircuits import *
        >>> import numpy as np
        >>> # a controlled-Y written out as a 4x4 matrix
        >>> matrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, -1j],[0, 0, 1j, 0]])
        >>> g = GateCustom(matrix)
        >>> g.num_qubits
        2
        >>> c = Circuit()
        >>> c.push(g, 0, 1)
        2-qubit circuit with 1 instruction:
        └── Custom(...) @ q[0:1]
        <BLANKLINE>
        >>> # a single qubit custom gate decomposes into one GateU
        >>> GateCustom(np.array([[0, 1], [1, 0]])).decompose()
        1-qubit circuit with 1 instruction:
        └── U(3.141592653589793, 0.0, -3.141592653589793, 0.0) @ q[0]
        <BLANKLINE>
    """

    _name = "Custom"
    _num_qubits = None
    _qregsizes = None

    def __init__(self, matrix):
        super().__init__()

        numeric = isinstance(matrix, np.ndarray) and np.issubdtype(
            matrix.dtype, np.number
        )

        if isinstance(matrix, np.ndarray):
            mat = se.Matrix(matrix.tolist())
        elif isinstance(matrix, se.Matrix):
            mat = matrix
        else:
            raise TypeError(
                f"{type(matrix)} not supported in GateCustom, use symengine.Matrix or numpy.ndarray."
            )

        if mat.rows != mat.cols:
            raise ValueError("Matrix is not square")

        tolerance = 1e-8
        # A numeric dtype cannot hold a symbol, so for those the elementwise scan
        # below can only ever fail — 4^n `__getitem__`/`isinstance` pairs for
        # nothing. Unitarity then reads the array we were handed rather than
        # converting the SymEngine copy back to NumPy.
        if numeric:
            if not self.is_unitary(matrix, tol=tolerance):
                raise ValueError("Matrix is not unitary")
        elif not any(
            isinstance(mat[i, j], str) or getattr(mat[i, j], "free_symbols", None)
            for i in range(mat.rows)
            for j in range(mat.cols)
        ):
            if not self.is_unitary(mat, tol=tolerance):
                raise ValueError("Matrix is not unitary")

        num_qubits = mat.rows.bit_length() - 1
        if num_qubits < 1 or mat.rows != 2**num_qubits:
            raise ValueError("Wrong number of the rows for the matrix")

        self._U = mat
        self._num_qubits = num_qubits
        self._qregsizes = [
            num_qubits,
        ]

    def _matrix(self):
        return self._U

    # Not inherited from AbstractOperator: that version caches on the class
    # for parameter-free operators, so every GateCustom would share the first
    # matrix built. `unwrappedmatrix` is overridden for the same reason.
    def matrix(self):
        """Return the gate matrix as a SymEngine matrix.

        Entries that can be converted to a complex number are returned as
        complex numbers; symbolic entries are kept as they are.

        Returns:
            symengine.Matrix: A new matrix on each call.
        """
        out = []
        for x in self._U:
            try:
                out.append(complex(x))
            except Exception:
                out.append(x)
        return se.Matrix(self._U.rows, self._U.cols, out)


    @property
    def num_qubits(self):
        r"""Number of qubits the gate acts on, :math:`\log_2` of the matrix size."""
        return self._num_qubits

    # The matrix entries are the parameters, in column-major order, named by
    # their linear index. `_parnames` stays empty so
    # decomposition does not try to rebuild the gate from one symbol per entry.
    @property
    def parnames(self):
        """Parameter names: the 1-based linear indices of the matrix entries."""
        return tuple(range(1, self._U.rows * self._U.cols + 1))

    def getparams(self):
        """Return the matrix entries as a list, in column-major order."""
        return list(self._U.T)

    def getparam(self, pn):
        """Return the matrix entry with 1-based column-major index ``pn``.

        Raises:
            IndexError: If ``pn`` is outside ``1..rows*cols``.
        """
        # 1-based; a plain index would let 0 and
        # negative values wrap around to the last entries.
        params = self.getparams()
        if not 1 <= pn <= len(params):
            raise IndexError(f"parameter index must be in 1..{len(params)}, not {pn}")
        return params[pn - 1]

    def inverse(self):
        return GateCustom(self._U.inv())

    @staticmethod
    def is_unitary(matrix, tol=1e-8):
        r"""Return whether :math:`U U^\dagger` equals the identity within ``tol``.

        Args:
            matrix (numpy.ndarray or symengine.Matrix): The matrix to check.
            tol (float): Absolute tolerance on each entry. Defaults to ``1e-8``.

        Returns:
            bool: ``True`` if the matrix is unitary.
        """
        # Fully numeric entries go through NumPy. The symbolic path below builds
        # a dense SymEngine product and then walks its 4^n entries in Python, so
        # for a wide gate it dominates construction: at 9 qubits it takes ~13s
        # against ~6ms here.
        try:
            u = _as_numpy_numeric(matrix)
        except TypeError:
            u = None
        if u is not None:
            return bool(
                np.allclose(u @ u.conj().T, np.eye(u.shape[0]), rtol=0, atol=tol)
            )

        conjugate_transpose = matrix.transpose().conjugate()
        product = matrix * conjugate_transpose
        identity_matrix = se.eye(matrix.rows)

        return all(
            abs(product[i, j] - identity_matrix[i, j]) < tol
            for i in range(matrix.rows)
            for j in range(matrix.cols)
        )

    def __repr__(self):
        return self.pretty_print()

    def __str__(self):
        return f"{self._name}(...)"

    def pretty_print(self):
        """Return a multi-line string showing the matrix row by row."""
        if isinstance(self._U, se.Matrix):
            U = np.array(self._U.tolist(), dtype=object)
        else:
            U = np.array(self._U, dtype=object)

        result = f"{self._num_qubits}-qubit GateCustom:\n"
        rows, cols = U.shape
        for i in range(rows):
            if i < rows - 1:
                result += "├── "
            else:
                result += "└── "
            result += " ".join(map(str, U[i, :]))
            if i < rows - 1:
                result += "\n"
        return result

    def evaluate(self, d):
        """Substitute symbols in the matrix entries and return a new gate.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            GateCustom: ``self`` if the matrix has no symbols, otherwise a new
            gate built from the substituted matrix (checked for unitarity
            again).
        """
        if not self.is_symbolic():
            return self
        import sympy as sp

        sympy_matrix = sp.Matrix(self._U)
        matrix = sympy_matrix.applyfunc(
            lambda entry: (
                entry.subs(d) if not isinstance(entry, (float, int)) else entry
            )
        )
        evaluated_matrix = se.Matrix(matrix.tolist())
        return GateCustom(evaluated_matrix)

    def unwrappedmatrix(self):
        """Return the matrix as a ``complex128`` NumPy array.

        Raises:
            TypeError: If an entry is still symbolic.
        """
        return _as_numpy_numeric(self._U)

    def _decompose(self, circ, qubits, bits, zvars):
        N = self.num_qubits
        if len(qubits) != N:
            raise ValueError(f"GateCustom expects {N} qubits, got {len(qubits)}")

        if N == 1:
            U_num = self.unwrappedmatrix()
            theta, phi, lam, gamma = _zyz_decomposition(U_num)
            circ.push(GateU(theta, phi, lam, gamma), qubits[0])
            return circ

        U_num = self.unwrappedmatrix() 
        sub_circ, phase = _qsd_decomposition(U_num)

        # Map local qubits -> actual qubits
        mapping = {i: qubits[i] for i in range(N)}

        for inst in sub_circ:
            op = inst.get_operation()
            local_q = list(inst.get_qubits())
            new_q = [mapping[q] for q in local_q]
            circ.push(op, *new_q)

        # `_qsd_decomposition` builds a circuit for `U * exp(-i phase)`; putting
        # the phase back keeps the decomposition equal to `U` and not merely
        # proportional to it, which matters as soon as the gate sits under a
        # Control or the circuit is read through amplitudes.
        if abs(phase) > 1e-14:
            circ.push(GateU(0, 0, 0, phase), qubits[0])

        return circ



__all__ = ["GateCustom"]
