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

from mimiqcircuits import Operation


class Not(Operation):
    r"""
    Not operation.

    Represents a NOT operation that can be added to quantum circuits.
    This operation  inverts a classical bit.

    Examples:

        >>> from mimiqcircuits import *
        >>> not_op = Not()
        >>> not_op.name
        '!'
        >>> c = Circuit()
        >>> c.push(Not(), 1)
        2-bit circuit with 1 instructions:
        └── ! @ c[1]
        <BLANKLINE>
    """

    _name = "!"
    _num_bits = 1
    _num_qubits = 0
    _num_cregs = 0
    _num_zvars = 0
    _qregsizes = []
    _cregsizes = [1]

    def inverse(self):
        return Not()

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def __repr__(self):
        return f"{self._name}"


__all__ = ["Not"]
