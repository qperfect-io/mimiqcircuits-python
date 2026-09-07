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

import numpy as np
from mimiqcircuits.operations.decompositions.matrix_decompositions.utils import _as_numpy_numeric


def _zyz_decomposition(U):
    """
    Decompose a single-qubit unitary matrix U into angles (theta, phi, lambda, gamma)
    such that:
        U = exp(i*gamma) * Rz(phi) * Ry(theta) * Rz(lambda)

    Returns:
        (theta, phi, lambda, gamma)  as Python floats
    """
    U = _as_numpy_numeric(U)

    if U.shape != (2, 2):
        raise ValueError("_zyz_decomposition expects a 2x2 unitary matrix")

    u00 = U[0, 0]
    u01 = U[0, 1]
    u10 = U[1, 0]
    u11 = U[1, 1]

    # cos(theta/2) and sin(theta/2) read straight off the matrix, and theta from
    # their ratio. Going through `arccos(|u00|)` instead loses half the
    # significant digits whenever `|u00| ~ 1` — `arccos(1 - eps) ~ sqrt(2 eps)`,
    # so a diagonal matrix comes out with theta ~ 1.5e-8 instead of 0, which is
    # both a 1e-8 error in the reconstruction and enough to miss any test for
    # "theta is zero".
    c = abs(u00)
    s = abs(u10)
    theta = 2.0 * np.arctan2(s, c)

    # Diagonal: the off-diagonal entries are exactly zero, so phi is free — only
    # phi + lambda is fixed, and the conventional choice is phi = 0.
    if s == 0.0:
        gamma = np.angle(u00)
        return (0.0, 0.0, float(np.angle(u11) - gamma), float(gamma))

    # Anti-diagonal: both diagonal entries vanish, so gamma and lambda cannot be
    # read from them. `u01` and `u10` carry independent phases here (any
    # `[[0, b], [a, 0]]` with `|a| = |b| = 1` is unitary), so lambda must come
    # from `u01`.
    if c <= 1e-8 * max(c, s):
        gamma = np.angle(u10)
        lam = np.angle(u01) - gamma - np.pi
        return (float(np.pi), 0.0, float(lam), float(gamma))

    # Everywhere else, take every phase from an entry of size cos(theta/2)
    # except phi, whose defining entry is `u10`. Reading lambda from `u11`
    # rather than from `u01` is what keeps a near-diagonal matrix exact: the
    # noisy `angle(u10)` of a vanishing entry then enters phi and lambda with
    # opposite signs, so it cancels in `phi + lambda` — the only combination
    # that multiplies cos(theta/2) — and what it does reach is scaled by
    # sin(theta/2) ~ 0.
    gamma = np.angle(u00)
    phi = np.angle(u10) - gamma
    lam = np.angle(u11) - np.angle(u10)

    return (float(theta), float(phi), float(lam), float(gamma))
