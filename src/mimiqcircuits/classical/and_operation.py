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

from mimiqcircuits.classical.abstract_classical import (
    AbstractClassical,
    _format_logical,
)


class And(AbstractClassical):
    """Bitwise AND of classical bits, written to a classical bit.

    ``And(N)`` acts on ``N`` bits: the first is the destination and the
    other ``N - 1`` are the inputs. The destination is set to 1 only if all inputs are 1. The
    destination may also be one of the inputs, which updates it in place.

    Args:
        N (int): Total number of bits, destination included. At least 3;
            defaults to 3.

    Raises:
        ValueError: If ``N`` is less than 3.

    Examples:
        >>> from mimiqcircuits import *
        >>> # placeholders show the destination ?0 and the inputs
        >>> And()
        c[?0] = c[?1] & c[?2]
        >>> And(8)
        c[?0] = & @ c[?1:?7]
        >>> # c[0] = c[2] & c[3]
        >>> c = Circuit()
        >>> c.push(And(), 0, 2, 3)
        4-bit circuit with 1 instruction:
        └── c[0] = c[2] & c[3]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(And(5), 0, 1, 2, 3, 4)
        5-bit circuit with 1 instruction:
        └── c[0] = c[1] & c[2] & c[3] & c[4]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(And(8), 0, 1, 2, 3, 4, 5, 6, 7)
        8-bit circuit with 1 instruction:
        └── c[0] = & @ c[1, 2, 3, 4, 5, 6, 7]
        <BLANKLINE>
    """

    _name = "&"
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
    _allow_bit_aliasing = True

    def __init__(self, N=3):
        """Create an And on ``N`` bits: one destination and ``N - 1`` inputs.

        Raises:
            ValueError: If ``N`` is less than 3.
        """
        if N < 3:
            raise ValueError("And operation requires at least 3 classical bits.")
        super().__init__()
        self._num_bits = N
        self._num_cregs = 1
        self._cregsizes = [N]

    def iswrapper(self):
        return False

    def __repr__(self):
        """Return the operation written with placeholder bits, such as ``c[?0]``."""
        N = self._num_bits
        if N > 6:
            # Compact view for many bits
            return f"c[?0] = & @ c[?1:?{N - 1}]"
        else:
            # Expanded view for fewer bits
            parts = [f"c[?{i}]" for i in range(1, N)]
            return f"c[?0] = {' & '.join(parts)}"

    def __str__(self):
        """Return the operator symbol, ``"&"``."""
        return self._name

    def format_with_targets(self, qubits, bits, zvars):
        """Return the operation written with the given target bits."""
        return _format_logical("&", bits)


__all__ = ["And"]
