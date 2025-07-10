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

from mimiqcircuits.operations.operation import Operation
import mimiqcircuits.lazy as lz


class Barrier(Operation):
    """Barrier operation.

    A barrier is a special operation that does not affect the quantum state or
    the execution of a circuit, but it prevents compression or optimization
    operation from being applied across it.

    Examples:

        Adding Barrier operation to the Circuit (The args can be: range, list,
        tuple, set or int)

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Barrier(1), 1)
        2-qubit circuit with 1 instructions:
        └── Barrier @ q[1]
        <BLANKLINE>

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Barrier(1), range(0,4))
        4-qubit circuit with 4 instructions:
        ├── Barrier @ q[0]
        ├── Barrier @ q[1]
        ├── Barrier @ q[2]
        └── Barrier @ q[3]
        <BLANKLINE>

        Adding Barrier to the circuit as a multi-qubits gate

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Barrier(5),1,2,3,4,5)
        6-qubit circuit with 1 instructions:
        └── Barrier @ q[1,2,3,4,5]
        <BLANKLINE>
    """

    _name = "Barrier"
    _num_qubits = None
    _num_bits = 0
    _num_zvars = 0

    def __init__(self, num_qubits):
        """
        Initialize a barrier operation

        Args:
            num_qubits: number of qubits the barrier will cover

        Raises:
            ValueError: if num_qubits is less than 1
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

    def inverse(self):
        return self

    def power(self, p):
        raise TypeError("Barrier^p is not defined.")

    def control(self, num_qubits):
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
