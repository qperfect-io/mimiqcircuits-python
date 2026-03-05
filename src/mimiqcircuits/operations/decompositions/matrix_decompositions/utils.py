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
import symengine as se
import sympy as sp


def _as_numpy_numeric(U):
    if isinstance(U, np.ndarray):
        return U.astype(complex)

    try:
        if isinstance(U, sp.MatrixBase):
            if not all(x.is_number for x in U):
                raise TypeError("Symbolic matrices are not supported for CSD")
            return np.array(U.evalf(), dtype=complex)
    except ImportError:
        pass

    try:
        if isinstance(U, se.Matrix):
            try:
                return np.array(
                    [[complex(x) for x in row] for row in U.tolist()], dtype=complex
                )
            except Exception:
                raise TypeError("Symbolic matrices are not supported for CSD")
    except ImportError:
        pass

    raise TypeError(f"Unsupported matrix type: {type(U)}")


def _to_symengine_matrix(A):

    def to_se_number(z):
        z = complex(z)

        if abs(z.imag) < 1e-15:
            return se.RealDouble(float(z.real))
        else:
            return se.ComplexDouble(z)

    return se.Matrix([[to_se_number(x) for x in row] for row in A])



