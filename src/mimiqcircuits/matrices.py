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

from symengine import I, cos, sin, exp, pi, Matrix
import symengine as se
import numpy as np
from functools import reduce
import mimiqcircuits as mc
import sympy as sp


# Writing ``cis(x)`` as ``exp(I*x)`` (rather than ``cos(x) + I*sin(x)``) keeps
# the symbolic display compact — previously ``sp.simplify`` recognized the
# trig form and collapsed it; doing it at construction time lets us keep the
# nice output without paying for SymPy.
def cis(x):
    return exp(I * x)


def cispi(x):
    return exp(I * pi * x)


def cospi(x):
    return cos(pi * x)


def sinpi(x):
    return sin(pi * x)


def pmatrix(lmbda):
    return Matrix([[1, 0], [0, cis(lmbda)]])


def umatrix(theta, phi, lmbda, gamma=0):
    # ``gamma`` defaults to the integer ``0`` (not ``0.0``): ``exp(I*0)``
    # collapses to the integer ``1`` in SymEngine, whereas ``exp(I*0.0)``
    # stays as ``1.0 + 0.0*I`` and litters the symbolic display.
    costheta2 = cos(theta / 2)
    sintheta2 = sin(theta / 2)
    return Matrix(
        [
            [cis(gamma) * costheta2, -cis(lmbda + gamma) * sintheta2],
            [cis(phi + gamma) * sintheta2, cis(phi + lmbda + gamma) * costheta2],
        ]
    )


def umatrixpi(theta, phi, lmbda, gamma=0):
    costheta2 = cospi(theta / 2)
    sintheta2 = sinpi(theta / 2)
    return Matrix(
        [
            [cispi(gamma) * costheta2, -cispi(lmbda + gamma) * sintheta2],
            [cispi(phi + gamma) * sintheta2, cispi(phi + lmbda + gamma) * costheta2],
        ]
    )


def _apply_local(u, g, targets, nq):
    """Left-multiply ``u`` by ``g`` acting on ``targets``, without embedding ``g``.

    Building the embedding costs a chain of ``kron`` products and a full
    ``2**nq`` square product; contracting ``g`` against ``u``'s target axes
    instead is ``O(2**len(targets) * 4**nq)`` and allocates nothing wider than
    ``u``. Axis ``i`` of the reshaped view is qubit ``i`` because MIMIQ orders
    qubit 0 as the most significant index bit — the same convention
    ``reorder_qubits_matrix`` realises through its identity padding.
    """
    ng = len(targets)
    front = list(range(ng))
    t = np.moveaxis(u.reshape((2,) * nq + (-1,)), targets, front)
    shape = t.shape
    t = (g @ t.reshape(2**ng, -1)).reshape(shape)
    return np.moveaxis(t, front, targets).reshape(2**nq, 2**nq)


def _index_permutation(qperm, nq):
    """Permutation of the ``2**nq`` basis indices induced by permuting index bits.

    The map permutes index *bits*, so it is linear over them: the image of an
    index is the sum of the images of its set bits. Evaluating it on the ``nq``
    basis indices costs ``nq`` ``BitString`` round-trips instead of ``2**nq``.
    The images form a bijection on ``range(2**nq)``, so the permutation we want
    is their inverse — a scatter, not an ``argsort``. The per-bit images still go
    through ``BitString``, so the endianness convention is read from there rather
    than re-derived here.
    """
    weights = np.empty(nq, dtype=np.int64)
    for k in range(nq):
        bs = mc.BitString.fromint(nq, 1 << k)
        weights[k] = mc.BitString("".join(str(bs[q]) for q in qperm)).tointeger()

    dim = 1 << nq
    bits = (np.arange(dim, dtype=np.int64)[:, None] >> np.arange(nq)) & 1
    ints = bits @ weights

    perm = np.empty(dim, dtype=np.int64)
    perm[ints] = np.arange(dim, dtype=np.int64)
    return perm


# Entry types for which multiplying by the identity padding (``1.0`` or an
# exact ``0``) gives the same result however many times, and in whatever
# order, it is applied. Exact complex numbers are excluded: ``I*1.0*0`` is a
# ``ComplexDouble`` zero while ``I*0`` is an exact ``0``.
_NUMERIC_SE_TYPES = (se.Rational, se.RealDouble, se.RealMPFR, se.ComplexDouble)


def _isnumeric_se(M):
    if not isinstance(M, se.DenseMatrix):
        return False
    return all(isinstance(x, _NUMERIC_SE_TYPES) for x in M)


def _reorder_numeric_se(M, fullqubits, nq, ident):
    """Numeric-entry counterpart of the SymEngine branch of ``reorder_qubits_matrix``.

    The SymEngine branch pads ``M`` with ``kron(M, ident)`` and then gathers
    every entry through a basis permutation, doing ``4**nq`` symbolic
    multiplications and assignments. For numeric entries each output entry is
    either ``M[a, b] * 1.0`` (where the padding wires agree) or
    ``M[a, b] * 0`` (where they do not), and SymEngine keeps the kind of zero
    that product yields: exact ``0`` for real entries, a signed
    ``ComplexDouble`` zero for complex ones. Precomputing both products for
    the ``4**k`` entries of ``M`` and gathering them with NumPy index arithmetic
    gives the same entries, object for object, at a fraction of the cost.
    """
    k = M.rows.bit_length() - 1
    npad = nq - k
    dim = 1 << nq

    if fullqubits == sorted(fullqubits):
        perm = np.arange(dim, dtype=np.int64)
    else:
        qperm = [(nq - 1) - i for i in reversed(np.argsort(fullqubits))]
        perm = _index_permutation(qperm, nq)

    entries = list(M)
    src = perm >> npad
    idx = src[:, None] * M.cols + src[None, :]
    if npad == 0:
        table = entries
    else:
        # Identity padding entries in the same SymEngine types the kron uses.
        one, zero = ident[0, 0], ident[0, 1]
        table = [x * one for x in entries] + [x * zero for x in entries]
        pad = perm & ((1 << npad) - 1)
        idx = np.where(pad[:, None] == pad[None, :], idx, idx + len(entries))

    return se.DenseMatrix([[table[t] for t in row] for row in idx.tolist()])


def reorder_qubits_matrix(M, qubits, nq=None):
    if nq is None:
        nq = max(qubits) + 1

    # A NumPy matrix is embedded by contracting it against an identity: that is
    # the same result as padding with `kron` and permuting, without building
    # either the padding chain or the index permutation. Every entry of the
    # product is a sum with at most one non-zero term, so it is exact.
    if isinstance(M, np.ndarray):
        return _apply_local(np.eye(2**nq, dtype=M.dtype), M, list(qubits), nq)

    fullqubits = list(qubits) + [q for q in range(nq) if q not in qubits]
    ident = mc.GateID().matrix()

    if _isnumeric_se(M) and not (
        len(qubits) == nq and fullqubits == sorted(fullqubits)
    ):
        return _reorder_numeric_se(M, fullqubits, nq, ident)

    # Symbolic matrices, and numeric ones that need no reordering or
    # padding, take the element-by-element SymEngine path.
    fullM = M
    for _ in range(nq - len(qubits)):
        fullM = kronecker(fullM, ident)

    # The identity padding above places the gate on wires 0..len(qubits)-1, so
    # no permutation is needed only when the full wire list is already ordered
    # (i.e. the gate occupies the leading contiguous wires). Testing `qubits`
    # alone is wrong for non-contiguous targets such as [0, 3].
    if fullqubits == sorted(fullqubits):
        return fullM

    qperm = [(nq - 1) - i for i in reversed(np.argsort(fullqubits))]

    dim = 2**nq

    perm = _index_permutation(qperm, nq)

    reordered = se.Matrix([[se.S(0) for _ in range(dim)] for _ in range(dim)])
    for i in range(dim):
        for j in range(dim):
            reordered[i, j] = fullM[perm[i], perm[j]]

    return reordered


def kronecker(A, B):
    """Kronecker product ``A ⊗ B``.

    Dispatches on the input types: two ``numpy.ndarray`` inputs use
    :func:`numpy.kron` (fast path); otherwise builds a SymEngine matrix.
    """
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.kron(A, B)

    rows_A, cols_A = A.nrows(), A.ncols()
    rows_B, cols_B = B.nrows(), B.ncols()

    result = se.zeros(rows_A * rows_B, cols_A * cols_B)
    for i in range(rows_A):
        for j in range(cols_A):
            for k in range(rows_B):
                for l in range(cols_B):
                    result[i * rows_B + k, j * cols_B + l] = A[i, j] * B[k, l]
    return result


# The single-qubit Pauli matrices as numpy arrays, for numerical code that
# should not go through SymEngine.
PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string_matrix(s):
    """Numerical matrix of the Pauli string ``s``, such as ``"XIZ"``.

    Returns a complex ``numpy.ndarray`` equal to ``PauliString(s).matrix()``:
    the first character is the most significant factor of the Kronecker
    product.
    """
    return reduce(np.kron, (PAULI_MATRICES[p] for p in s))


def symbolic_matrix_exponential(A, theta):
    """
    Use SymPy to compute symbolic matrix exponential: exp(-i * θ/2 * A)
    """
    A_sympy = sp.Matrix(A.tolist())
    expr = -sp.I * theta / 2 * A_sympy
    return se.Matrix(sp.exp(expr).tolist())
