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

    # Calculate theta from diagonal magnitude
    cos_theta_2 = min(abs(u00), 1.0)
    theta = 2.0 * np.arccos(cos_theta_2)

    atol = 1e-10

    
    # Case 1: theta ~ 0 (Identity-like)
    if np.isclose(theta, 0.0, atol=atol):
        gamma = np.angle(u00)
        # u11 = exp(i*(gamma + phi + lambda))
        # choose phi = 0
        lam = np.angle(u11) - gamma
        return (0.0, 0.0, float(lam), float(gamma))

    
    # Case 2: theta ~ pi (X-like)
    if np.isclose(theta, np.pi, atol=atol):
        # u10 = exp(i*(gamma + phi)), choose phi = 0
        gamma = np.angle(u10)
        # u01 = -exp(i*(gamma + lambda))
        lam = np.angle(u01) - gamma - np.pi
        return (float(theta), 0.0, float(lam), float(gamma))

    
    # General case
    gamma = np.angle(u00)
    phi = np.angle(u10) - gamma
    lam = np.angle(-u01) - gamma

    return (float(theta), float(phi), float(lam), float(gamma))
