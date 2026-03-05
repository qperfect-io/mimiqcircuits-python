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


class Xor(AbstractClassical):
    """
    Computes the bitwise XOR of N-1 classical bits and stores the result in the first given bit.

    Examples:
        >>> from mimiqcircuits import *
        >>> Xor()
        c[?0] = c[?1] ^ c[?2]
        >>> Xor(8)
        c[?0] = ^ @ c[?1:?7]
        >>> c = Circuit()
        >>> c.push(Xor(), 0, 2, 3)
        4-bit circuit with 1 instruction:
        └── c[0] = c[2] ^ c[3]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(Xor(5), 0, 1, 2, 3, 4)
        5-bit circuit with 1 instruction:
        └── c[0] = c[1] ^ c[2] ^ c[3] ^ c[4]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(Xor(8), 0, 1, 2, 3, 4, 5, 6, 7)
        8-bit circuit with 1 instruction:
        └── c[0] = ^ @ c[1, 2, 3, 4, 5, 6, 7]
        <BLANKLINE>
    """

    _name = "^"
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
        Initializes an Xor operation.

        Args:
            N: The total number of classical bits (1 destination + N-1 inputs).
               Must be 3 or greater. Defaults to 3.
        """
        if N < 3:
            raise ValueError("Xor operation requires at least 3 classical bits.")
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
        if N > 6:
            # Compact view for many bits
            return f"c[?0] = ^ @ c[?1:?{N - 1}]"
        else:
            # Expanded view for fewer bits
            parts = [f"c[?{i}]" for i in range(1, N)]
            return f"c[?0] = {' ^ '.join(parts)}"

    def __str__(self):
        """
        Returns the simple string name of the operation.
        """
        return self._name

    def format_with_targets(self, qubits, bits, zvars):
        """
        Formats the operation with its specific target bits.
        """
        N = self._num_bits

        if N > 6:
            # Compact view for many bits
            inputs = ", ".join(str(b) for b in bits[1:])
            return f"c[{bits[0]}] = ^ @ c[{inputs}]"
        else:
            # Expanded view for fewer bits
            inputs = " ^ ".join(f"c[{b}]" for b in bits[1:])
            return f"c[{bits[0]}] = {inputs}"


__all__ = ["Xor"]
