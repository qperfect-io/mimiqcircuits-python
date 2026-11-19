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
"""Barrier operation."""

from mimiqcircuits.operations.operation import Operation
import mimiqcircuits.lazy as lz


class Barrier(Operation):
    """No-op that stops circuit optimizations from moving gates across it.

    A barrier does not change the quantum state. Compression and
    optimization passes do not merge, cancel or reorder operations across
    it.

    ``Barrier(n)`` covers ``n`` qubits. ``Barrier()`` without arguments
    returns a lazy expression that takes the number of qubits later.

    Args:
        num_qubits: Number of qubits the barrier covers, at least 1.

    Raises:
        ValueError: If ``num_qubits`` is less than 1, or if more than one
            argument is given.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> # a single-qubit barrier on qubit 1
        >>> c.push(Barrier(1), 1)
        2-qubit circuit with 1 instruction:
        └── Barrier @ q[1]
        <BLANKLINE>

        Pushing a single-qubit barrier to a range adds one barrier per qubit:

        >>> c = Circuit()
        >>> c.push(Barrier(1), range(0,4))
        4-qubit circuit with 4 instructions:
        ├── Barrier @ q[0]
        ├── Barrier @ q[1]
        ├── Barrier @ q[2]
        └── Barrier @ q[3]
        <BLANKLINE>

        A multi-qubit barrier is a single instruction across all its qubits:

        >>> c = Circuit()
        >>> c.push(Barrier(5),1,2,3,4,5)
        6-qubit circuit with 1 instruction:
        └── Barrier @ q[1:5]
        <BLANKLINE>
    """

    _name = "Barrier"
    _num_qubits = None
    _num_bits = 0
    _num_zvars = 0

    def __init__(self, num_qubits):
        """Create a barrier on ``num_qubits`` qubits.

        Args:
            num_qubits: Number of qubits the barrier covers.

        Raises:
            ValueError: If ``num_qubits`` is less than 1.
        """
        super().__init__()

        if num_qubits < 1:
            raise ValueError("Number of qubits must be greater than 0")

        self._num_qubits = num_qubits
        self._qregsizes = [num_qubits]

    def __new__(self, *args):
        if len(args) == 0:
            return lz.LazyExpr(Barrier, lz.LazyArg())
        elif len(args) == 1:
            return object.__new__(self)
        else:
            raise ValueError("Invalid number of arguments.")

    # copy, deepcopy and pickle rebuild through __new__, which without
    # arguments returns a LazyExpr instead of a Barrier.
    def __getnewargs__(self):
        return (self._num_qubits,)

    def inverse(self):
        """Return the barrier itself; it is its own inverse."""
        return self

    def power(self, p):
        """Not supported for barriers.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Barrier^p is not defined.")

    def control(self, num_qubits):
        """Not supported for barriers.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Controlled Barrier is not defined.")

    def iswrapper(self):
        return False

    def asciiwidth(self, qubits, bits, zvars):
        return 1

    @staticmethod
    def isunitary():
        return True


# export operations
__all__ = ["Barrier"]
