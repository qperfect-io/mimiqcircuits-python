# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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
"""Readout error model."""

import numpy as np
import symengine as se
import sympy as sp
from mimiqcircuits import Operation


class ReadoutErr(Operation):
    r"""Represents a classical readout error applied immediately after a measurement.

    The readout error is modeled by a 2×2 *confusion matrix*:

    .. math::

        \begin{pmatrix}
            P(\text{report } 0 | \text{true } 0) &
            P(\text{report } 1 | \text{true } 0) = p_0 \\
            P(\text{report } 0 | \text{true } 1) = p_1 &
            P(\text{report } 1 | \text{true } 1)
        \end{pmatrix}

    Can be initialized either from the error probabilities ``p0``, ``p1`` or
    directly from a 2×2 confusion matrix. Accepts NumPy, SymPy, or SymEngine matrices.

    Examples:
        >>> from mimiqcircuits import *
        >>> ReadoutErr(0.02, 0.03)
        RErr(0.02,0.03)

        >>> import numpy as np
        >>> ReadoutErr(np.array([[0.98, 0.02],
        ...                      [0.03, 0.97]]))
        RErr(0.02,0.03)
        
        >>> c = Circuit()
        >>> c.push(ReadoutErr(0.01, 0.02),0)
        1-bit circuit with 1 instruction:
        └── RErr(0.01, 0.02) @ c[0]
        <BLANKLINE>
    """

    _name = "RErr"
    _num_qubits = 0
    _num_bits = 1
    _num_zvars = 0
    _parnames = ("p0", "p1")
    _num_cregs = 1
    _num_qregs = 0
    _num_zregs = 0
    _cregsizes = [1]

    def __init__(self, p0, p1=None):
        # Case 1: confusion matrix
        if p1 is None:
            if isinstance(p0, np.ndarray):
                mat = p0
            elif isinstance(p0, se.Matrix):
                mat = np.array(p0.tolist(), dtype=object)
            elif isinstance(p0, sp.MatrixBase):
                mat = np.array(p0.tolist(), dtype=object)
            else:
                raise TypeError(
                    f"ReadoutErr expects numpy.ndarray, symengine.Matrix, or sympy.Matrix — got {type(p0)}"
                )

            if mat.shape != (2, 2):
                raise ValueError("Confusion matrix must be 2×2.")

            for i in range(2):
                for j in range(2):
                    val = mat[i, j]
                    if isinstance(val, (sp.Basic, se.Basic)) and not val.is_Number:
                        raise TypeError(
                            f"All elements of confusion matrix  must be numeric."
                        )

            if np.any((mat < 0) | (mat > 1)):
                raise ValueError("All entries in confusion matrix must be in [0,1].")

            for i in range(2):
                row_sum = np.sum(mat[i, :])
                if not np.isclose(row_sum, 1.0, atol=1e-8):
                    raise ValueError(
                        f"Row {i + 1} of confusion matrix must sum to 1 (got {row_sum})."
                    )

            p0, p1 = float(mat[0, 1]), float(mat[1, 0])

        # Case 2: direct parameters (numeric or symbolic allowed)
        self.p0 = p0
        self.p1 = p1

        if not isinstance(self.p0, (sp.Basic, se.Basic)) and not isinstance(
            self.p1, (sp.Basic, se.Basic)
        ):
            if not (0.0 <= self.p0 <= 1.0):
                raise ValueError(f"p0 must be in [0,1], got {self.p0}")
            if not (0.0 <= self.p1 <= 1.0):
                raise ValueError(f"p1 must be in [0,1], got {self.p1}")

        super().__init__()

    def evaluate(self, d={}):
        """Evaluate symbolic parameters numerically.
        Returns a new ReadoutErr instance with evaluated parameters.
        """
        p0_eval = (
            float(self.p0.subs(d).evalf()) if hasattr(self.p0, "subs") else self.p0
        )
        p1_eval = (
            float(self.p1.subs(d).evalf()) if hasattr(self.p1, "subs") else self.p1
        )

        if not (0.0 <= p0_eval <= 1.0) or not (0.0 <= p1_eval <= 1.0):
            raise ValueError("Readout probabilities must be in [0,1].")

        return ReadoutErr(p0_eval, p1_eval)

    def matrix(self):
        """Return the confusion matrix as a symengine.Matrix."""
        return se.Matrix(
            [
                [1 - self.p0, self.p0],
                [self.p1, 1 - self.p1],
            ]
        )

    @property
    def opname(self):
        return self._name

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"{self._name}({self.p0}, {self.p1})"

    def __repr__(self):
        return f"{self._name}({self.p0},{self.p1})"

    def iswrapper(self):
        pass


__all__ = ["ReadoutErr"]
