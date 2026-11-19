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


class SetBit0(AbstractClassical):
    r"""Set a classical bit to 0, whatever its value.

    Acts on one classical bit.

    Examples:
        >>> from mimiqcircuits import *
        >>> SetBit0().name
        'set0'
        >>> # set bit 1 to 0
        >>> c = Circuit()
        >>> c.push(SetBit0(), 1)
        2-bit circuit with 1 instruction:
        └── c[1] = 0
        <BLANKLINE>
    """

    _name = "set0"
    _num_bits = 1
    _num_qubits = 0
    _num_cregs = 0
    _num_zvars = 0
    _qregsizes = []
    _cregsizes = [1]

    def inverse(self):
        """Return ``SetBit1()``.

        Setting a bit is not invertible; this pairing only mirrors the
        opposite operation.
        """
        return SetBit1()

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def __repr__(self):
        return f"{self._name}"

    def format_with_targets(self, qubits, bits, zvars):
        return f"c[{bits[0]}] = 0"


class SetBit1(AbstractClassical):
    r"""Set a classical bit to 1, whatever its value.

    Acts on one classical bit.

    Examples:
        >>> from mimiqcircuits import *
        >>> SetBit1().name
        'set1'
        >>> # set bit 1 to 1
        >>> c = Circuit()
        >>> c.push(SetBit1(), 1)
        2-bit circuit with 1 instruction:
        └── c[1] = 1
        <BLANKLINE>
    """

    _name = "set1"
    _num_bits = 1
    _num_qubits = 0
    _num_cregs = 0
    _num_zvars = 0
    _qregsizes = []
    _cregsizes = [1]

    def inverse(self):
        """Return ``SetBit0()``.

        Setting a bit is not invertible; this pairing only mirrors the
        opposite operation.
        """
        return SetBit0()

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def __repr__(self):
        return f"{self._name}"

    def format_with_targets(self, qubits, bits, zvars):
        return f"c[{bits[0]}] = 1"


__all__ = ["SetBit0", "SetBit1"]
