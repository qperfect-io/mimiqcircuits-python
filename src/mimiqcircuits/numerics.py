#
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
"""Fast numeric (NumPy/math) gate-matrix builders.

This module mirrors :mod:`mimiqcircuits.matrices` but builds ``numpy.ndarray``
matrices directly from Python numbers, bypassing SymEngine entirely. It is
the Python analogue of the ``_matrix(::Type{Gate}, params...)`` methods in
``MimiqCircuitsBase.jl/src/matrices.jl``.

The helpers ``sinpi``, ``cospi``, ``cispi`` are the Python equivalents of
Julia's ``sinpi``, ``cospi``, ``cispi`` — they return exact values at
integer and half-integer arguments, which is what lets
``matrix(GateRX(math.pi))`` yield an exactly-unitary result.
"""

from __future__ import annotations

import cmath
import math

import numpy as np


# --- π-aware trig primitives ------------------------------------------------


def sinpi(x: float) -> float:
    """Return ``sin(pi * x)``, exact at integers and half-integers.

    Equivalent to Julia's :func:`sinpi`.
    """
    # Reduce to (-1, 1] modulo 2 to expose exact cases.
    r = x - 2.0 * round(x * 0.5)
    if r == 0.0 or r == 1.0 or r == -1.0:
        return 0.0
    if r == 0.5:
        return 1.0
    if r == -0.5:
        return -1.0
    return math.sin(math.pi * r)


def cospi(x: float) -> float:
    """Return ``cos(pi * x)``, exact at integers and half-integers.

    Equivalent to Julia's :func:`cospi`.
    """
    r = x - 2.0 * round(x * 0.5)
    if r == 0.0:
        return 1.0
    if r == 1.0 or r == -1.0:
        return -1.0
    if r == 0.5 or r == -0.5:
        return 0.0
    return math.cos(math.pi * r)


def cispi(x: float) -> complex:
    """Return ``exp(i * pi * x) == cos(pi*x) + i*sin(pi*x)``.

    Equivalent to Julia's :func:`cispi`.
    """
    return complex(cospi(x), sinpi(x))


def cis(x: float) -> complex:
    """Return ``exp(i * x) == cos(x) + i*sin(x)``.

    Equivalent to Julia's :func:`cis`.
    """
    return complex(math.cos(x), math.sin(x))


# --- Building blocks --------------------------------------------------------


def pmatrix(lmbda: float) -> np.ndarray:
    """Phase matrix ``diag(1, e^{i*lmbda})`` as a NumPy array."""
    return np.array([[1.0, 0.0], [0.0, cis(lmbda)]], dtype=np.complex128)


def pmatrixpi(lmbda: float) -> np.ndarray:
    """Phase matrix ``diag(1, e^{i*pi*lmbda})`` as a NumPy array."""
    return np.array([[1.0, 0.0], [0.0, cispi(lmbda)]], dtype=np.complex128)


def umatrix(theta: float, phi: float, lmbda: float, gamma: float = 0.0) -> np.ndarray:
    """Four-parameter single-qubit unitary — Julia parity with ``umatrix``.

    Uses ``cospi``/``sinpi`` on the half-angle so that rational multiples of
    ``pi`` in ``theta`` yield exact values.
    """
    # Divide-by-pi trick: math.pi / math.pi == 1.0 exactly in IEEE 754, so
    # theta == k * math.pi gives theta / 2 / math.pi == k / 2 exactly.
    half = theta / 2.0 / math.pi
    c = cospi(half)
    s = sinpi(half)
    return np.array(
        [
            [cis(gamma) * c, -cis(lmbda + gamma) * s],
            [cis(phi + gamma) * s, cis(phi + lmbda + gamma) * c],
        ],
        dtype=np.complex128,
    )


def umatrixpi(theta: float, phi: float, lmbda: float, gamma: float = 0.0) -> np.ndarray:
    """Like :func:`umatrix`, but arguments are already pre-divided by ``pi``."""
    c = cospi(theta / 2.0)
    s = sinpi(theta / 2.0)
    return np.array(
        [
            [cispi(gamma) * c, -cispi(lmbda + gamma) * s],
            [cispi(phi + gamma) * s, cispi(phi + lmbda + gamma) * c],
        ],
        dtype=np.complex128,
    )


def rxmatrix(theta: float) -> np.ndarray:
    """RX(theta) as a NumPy array, exact at rational multiples of pi."""
    half = theta / 2.0 / math.pi
    c = cospi(half)
    s = sinpi(half)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def rymatrix(theta: float) -> np.ndarray:
    """RY(theta) as a NumPy array, exact at rational multiples of pi."""
    half = theta / 2.0 / math.pi
    c = cospi(half)
    s = sinpi(half)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rzmatrix(lmbda: float) -> np.ndarray:
    """RZ(lmbda) as a NumPy array, exact at rational multiples of pi."""
    half = lmbda / 2.0 / math.pi
    e = cispi(half)
    return np.array([[e.conjugate(), 0.0], [0.0, e]], dtype=np.complex128)


def rmatrix(theta: float, phi: float) -> np.ndarray:
    """Axis-arbitrary single-qubit rotation ``R(theta, phi)``."""
    half = theta / 2.0 / math.pi
    c = cospi(half)
    s = sinpi(half)
    ep = cis(phi)
    return np.array(
        [[c, -1j * ep.conjugate() * s], [-1j * ep * s, c]],
        dtype=np.complex128,
    )


# --- Composition ------------------------------------------------------------


def ctrl(mat: np.ndarray) -> np.ndarray:
    """Single-control version of ``mat`` as a NumPy array."""
    dim = mat.shape[0]
    out = np.eye(2 * dim, dtype=np.complex128)
    out[dim:, dim:] = mat
    return out


def kron(*mats: np.ndarray) -> np.ndarray:
    """Left-to-right Kronecker product of one or more NumPy matrices."""
    if not mats:
        raise ValueError("kron requires at least one matrix")
    result = np.asarray(mats[0], dtype=np.complex128)
    for m in mats[1:]:
        result = np.kron(result, m)
    return result
