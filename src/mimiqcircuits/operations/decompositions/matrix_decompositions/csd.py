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
from scipy.linalg import svd
import symengine as se
from .utils import _as_numpy_numeric, _to_symengine_matrix


def _csd_decomposition(U, threshold=1e-6):
    """
    Perform Cosine-Sine Decomposition (CSD) on a unitary matrix U.
    """

    U = _as_numpy_numeric(U)

    n = U.shape[0]
    if n % 2 != 0:
        raise ValueError("Matrix dimension must be even for CSD.")

    m = n // 2

    # Partition
    u00 = U[:m, :m]
    u01 = U[:m, m:]
    u10 = U[m:, :m]
    u11 = U[m:, m:]

    # u00 = L0 * C * R0
    U0, S0, Vh0 = svd(u00)

    L0 = U0
    C_diag = np.clip(S0, 0.0, 1.0)
    R0 = Vh0

    # Derivation from u10 = L1 @ S @ R0 and u01 = -L0 @ S @ R1
    X = u10 @ R0.conj().T
    Y = L0.conj().T @ u01

    # `X[:, i] = L1[:, i] * sin(theta_i)` with `L1[:, i]` a unit vector, so the
    # sines are column norms. Taking them as `sin(arccos(sigma_i))` instead
    # loses half the significant digits whenever `sigma_i ~ 1` —
    # `arccos(1 - eps) ~ sqrt(2 eps)` — and a block that is already unitary has
    # every `sigma_i = 1`, so a diagonal `U` came out with `sin(theta) ~ 1.5e-8`
    # where it should be 0.
    sin_theta = np.linalg.norm(X, axis=0)
    theta = np.arctan2(sin_theta, C_diag)

    # Allocate
    L1 = np.zeros((m, m), dtype=complex)
    R1 = np.zeros((m, m), dtype=complex)

    determined = np.where(sin_theta > threshold)[0]
    undetermined = np.where(sin_theta <= threshold)[0]

    # Determined part
    for idx in determined:
        L1[:, idx] = X[:, idx] / sin_theta[idx]
        R1[idx, :] = -Y[idx, :] / sin_theta[idx]

    # Undetermined part
    if len(undetermined) > 0:
        L1_det = L1[:, determined]
        R1_det = R1[determined, :]
        C_det = np.diag(C_diag[determined])

        Rem = u11 - L1_det @ C_det @ R1_det
        Urem, _, Vhrem = svd(Rem)

        k = len(undetermined)
        L1[:, undetermined] = Urem[:, :k]
        R1[undetermined, :] = Vhrem[:k, :]

    # FINAL: convert to SymEngine
    return (
        _to_symengine_matrix(L0),
        _to_symengine_matrix(L1),
        _to_symengine_matrix(R0),
        _to_symengine_matrix(R1),
        se.Matrix(len(theta), 1, [se.RealDouble(float(t)) for t in theta]),
    )
