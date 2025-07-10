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


import mimiqcircuits as mc
import mimiqcircuits.lazy as lz


class PolynomialOracle(mc.Gate):
    """
    Polynomial Oracle.

    Args:
        a (Num): Coefficient a in the polynomial.
        b (Num): Coefficient b in the polynomial.
        c (Num): Coefficient c in the polynomial.
        d (Num): Coefficient d in the polynomial.

    Raises:
        ValueError: If the input parameters do not satisfy the required conditions.

    Returns:
        PolynomialOracle: The Polynomial Oracle.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(PolynomialOracle(5,5,1, 2, 3, 4), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        11-qubit circuit with 1 instructions:
        └── PolynomialOracle(1, 2, 3, 4) @ q[1,2,3,4,5], q[6,7,8,9,10]
        <BLANKLINE>
    """

    _name = "PolynomialOracle"
    _num_qregs = 2
    _num_qubits = None

    def __init__(self, nx, ny, a, b, c, d):
        if not all(isinstance(x, int) for x in [a, b, c, d]):
            raise ValueError("Coefficients a, b, c, d must be integers.")

        super().__init__()

        self._num_qubits = nx + ny
        self._qregsizes = [nx, ny]
        self._params = [nx, ny, a, b, c, d]
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __new__(cls, *args):
        if len(args) == 4:
            return lz.LazyExpr(PolynomialOracle, lz.LazyArg(), lz.LazyArg(), *args)
        elif len(args) == 6:
            return object.__new__(cls)
        else:
            raise ValueError("Invalid number of arguments.")

    def inverse(self):
        return self

    def _matrix(self):
        raise NotImplementedError(
            "Matrix representation for PolynomialOracle is not implemented."
        )

    def _decompose(self, circ, qubits, bits, _):
        return circ.push(self, *qubits)

    def __str__(self):
        return f"PolynomialOracle{self.a, self.b, self.c, self.d}"


__all__ = ["PolynomialOracle"]
