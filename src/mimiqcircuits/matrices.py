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

from symengine import I, cos, sin, exp, pi, eye, Matrix
import symengine as se
import numpy as np
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


def pmatrixpi(lmbda):
    return Matrix([[1, 0], [0, cispi(lmbda)]])


def pmatrix(lmbda):
    return Matrix([[1, 0], [0, cis(lmbda)]])


def ctrl(mat):
    """Single-control version of ``mat``.

    Dispatches on the input type: ``numpy.ndarray`` in, ``numpy.ndarray``
    out (fast path); SymEngine matrix in, SymEngine matrix out.
    """
    if isinstance(mat, np.ndarray):
        dim = mat.shape[0]
        out = np.eye(2 * dim, dtype=mat.dtype)
        out[dim:, dim:] = mat
        return out

    dim = mat.nrows()
    m = eye(2 * dim)
    m[dim:, dim:] = mat
    return m


def gphase(lmbda):
    return cis(lmbda)


def gphasepi(lmbda):
    return cispi(lmbda)


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


def rxmatrixpi(theta):
    return umatrixpi(theta, -1 / 2, 1 / 2)


def rxmatrix(theta):
    return rxmatrixpi(theta / pi)


def rymatrixpi(theta):
    return umatrixpi(theta, 0, 0)


def rymatrix(theta):
    return rymatrixpi(theta / pi)


def rzmatrixpi(lmbda):
    return gphasepi(-lmbda / 2) * umatrixpi(0, 0, lmbda)


def rzmatrix(lmbda):
    return rzmatrixpi(lmbda / pi)


def reorder_qubits_matrix(M, qubits, nq=None):
    if nq is None:
        nq = max(qubits) + 1

    fullqubits = list(qubits) + [q for q in range(nq) if q not in qubits]

    fullM = M

    # Pad with identities (kron) for nq - len(qubits)
    I = mc.GateID().matrix()
    for _ in range(nq - len(qubits)):
        fullM = kronecker(fullM, I)

    # If sorted, no reordering needed
    if sorted(qubits) == list(qubits):
        return fullM

    qperm = [(nq - 1) - i for i in reversed(np.argsort(fullqubits))]

    dim = 2**nq

    ints = []
    for i in range(dim):
        bs = mc.BitString(nq, i)
        permuted = mc.BitString("".join(str(bs[q]) for q in qperm))
        ints.append(permuted.tointeger())

    perm = np.argsort(ints)

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


def symbolic_matrix_exponential(A, theta):
    """
    Use SymPy to compute symbolic matrix exponential: exp(-i * θ/2 * A)
    """
    A_sympy = sp.Matrix(A.tolist())
    expr = -sp.I * theta / 2 * A_sympy
    return se.Matrix(sp.exp(expr).tolist())
