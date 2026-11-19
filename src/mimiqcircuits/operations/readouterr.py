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
from mimiqcircuits._sympycheck import _is_basic, _is_sympy_matrix
from mimiqcircuits.operations.operation import Operation


class ReadoutErr(Operation):
    r"""Classical readout error on one bit, applied after a measurement.

    The operation flips the recorded bit at random: a 0 is reported as 1 with
    probability :math:`p_0` and a 1 is reported as 0 with probability
    :math:`p_1`. This is the confusion matrix

    .. math::

        \begin{pmatrix}
            P(0 | 0) & P(1 | 0) \\
            P(0 | 1) & P(1 | 1)
        \end{pmatrix}
        =
        \begin{pmatrix}
            1 - p_0 & p_0 \\
            p_1 & 1 - p_1
        \end{pmatrix},

    where :math:`P(r | t)` is the probability of reporting :math:`r` when the
    true outcome is :math:`t`. The operation targets one classical bit and no
    qubits.

    Args:
        p0: Probability of reporting 1 when the outcome is 0, in
            :math:`[0, 1]`. It may also be a 2×2 confusion matrix (numpy,
            sympy or symengine) as above, with ``p1`` omitted; its entries
            must be numeric, in :math:`[0, 1]`, with rows adding up to 1.
        p1: Probability of reporting 0 when the outcome is 1, in
            :math:`[0, 1]`.

    A symbolic ``p0`` or ``p1`` is not range checked.

    Raises:
        TypeError: If ``p1`` is omitted and ``p0`` is not a matrix, or the
            matrix has symbolic entries.
        ValueError: If the matrix is not 2×2, has entries outside
            :math:`[0, 1]` or rows not adding up to 1, or if ``p0`` or ``p1``
            is outside :math:`[0, 1]`.

    Examples:
        >>> from mimiqcircuits import *
        >>> # 2% chance to read 1 instead of 0, 3% to read 0 instead of 1
        >>> ReadoutErr(0.02, 0.03)
        RErr(0.02,0.03)

        >>> # the same error from its confusion matrix
        >>> import numpy as np
        >>> ReadoutErr(np.array([[0.98, 0.02],
        ...                      [0.03, 0.97]]))
        RErr(0.02,0.03)

        >>> # apply it to bit 0, where a measurement result is stored
        >>> c = Circuit()
        >>> c.push(Measure(), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> c.push(ReadoutErr(0.01, 0.02), 0)
        1-qubit, 1-bit circuit with 2 instructions:
        ├── M @ q[0], c[0]
        └── RErr(0.01, 0.02) @ c[0]
        <BLANKLINE>

    See Also:
        :meth:`NoiseModel.add_readout_noise`, which adds readout errors after
        measurements.
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
            elif _is_sympy_matrix(p0):
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
                    if _is_basic(val) and not val.is_Number:
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

        if not _is_basic(self.p0) and not (0.0 <= self.p0 <= 1.0):
            raise ValueError(f"p0 must be in [0,1], got {self.p0}")
        if not _is_basic(self.p1) and not (0.0 <= self.p1 <= 1.0):
            raise ValueError(f"p1 must be in [0,1], got {self.p1}")

        super().__init__()

    def evaluate(self, d={}):
        """Return a copy with symbols in ``p0`` and ``p1`` substituted by values.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            ReadoutErr: A new operation; a probability becomes a float once no
            symbol is left in it.

        Raises:
            ValueError: If a substituted probability is outside :math:`[0, 1]`.
        """

        def sub(p):
            if not hasattr(p, "subs"):
                return p
            p = p.subs(d)
            return float(p.evalf()) if p.is_number else p

        # The constructor range checks every numeric probability.
        return ReadoutErr(sub(self.p0), sub(self.p1))

    def matrix(self):
        """Return the confusion matrix ``[[1 - p0, p0], [p1, 1 - p1]]``.

        Returns:
            symengine.Matrix: The 2×2 confusion matrix.

        Examples:
            >>> from mimiqcircuits import *
            >>> ReadoutErr(0.25, 0.5).matrix()
            [0.75, 0.25]
            [0.5, 0.5]
            <BLANKLINE>
        """
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
