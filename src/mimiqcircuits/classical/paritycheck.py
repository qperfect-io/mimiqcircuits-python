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

from mimiqcircuits.classical.abstract_classical import AbstractClassical


class ParityCheck(AbstractClassical):
    """
    Performs a parity check on N-1 classical bits and stores the result in the first bit.

    It computes the sum modulo 2 of the inputs.

    Examples:
        >>> from mimiqcircuits import *
        >>> ParityCheck()
        c[?0] = ⨊ c[?1, ?2]
        >>> ParityCheck(5)
        c[?0] = ⨊ c[?1, ?2, ?3, ?4]
        >>> c = Circuit()
        >>> c.push(ParityCheck(), 0, 2, 3)
        4-bit circuit with 1 instruction:
        └── c[0] = ⨊ c[2, 3]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(ParityCheck(5), 0, 1, 2, 3, 4)
        5-bit circuit with 1 instruction:
        └── c[0] = ⨊ c[1, 2, 3, 4]
        <BLANKLINE>
    """

    _name = "⨊"
    _num_bits = 0
    _num_qubits = 0
    _num_zvars = 0
    _num_qregs = 0
    _num_cregs = 0
    _num_zregs = 0
    _qregsizes = []
    _cregsizes = []
    _zregsizes = []
    _parnames = ()

    def __init__(self, N=3):
        """
        Initializes a ParityCheck operation.

        Args:
            N: The total number of classical bits (1 destination + N-1 inputs).
               Must be 3 or greater. Defaults to 3.
        """
        if N < 3:
            raise ValueError("ParityCheck must act on at least 3 classical bits")
        super().__init__()
        self._num_bits = N
        self._num_cregs = 1
        self._cregsizes = [N]

    def iswrapper(self):
        return False

    def __repr__(self):
        """
        Returns the string representation of the operation with placeholders.
        """
        N = self._num_bits
        inputs = ", ".join(f"?{i}" for i in range(1, N))
        return f"c[?0] = ⨊ c[{inputs}]"

    def __str__(self):
        """
        Returns the simple string name of the operation.
        """
        return self._name

    def format_with_targets(self, qubits, bits, zvars):
        """
        Formats the operation with its specific target bits.
        """
        creg = bits
        inputs = ", ".join(str(b) for b in creg[1:])
        return f"c[{creg[0]}] = ⨊ c[{inputs}]"


__all__ = ["ParityCheck"]
