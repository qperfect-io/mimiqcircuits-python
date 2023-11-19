#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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


class Reset(Operation):
    """Reset operation.

    Quantum operation that resets the status of one qubit to the :math:`\\ket{0}` state.

    This operation is non-reversible.

    Examples:
        Adding Reset operation to the Circuit (The args can be: range, list, tuple, set or int)

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Reset(), 0)
        1-qubit circuit with 1 instructions:
        └── Reset @ q0
        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Reset(),(0,1,2))
        3-qubit circuit with 3 instructions:
        ├── Reset @ q0
        ├── Reset @ q1
        └── Reset @ q2
    """
    _name = 'Reset'
    _num_qubits = 1
    _num_bits = 0

    def inverse(self):
        raise TypeError('Reset is not inversible')

    def power(self, pwr):
        raise TypeError('Reset^p is not defined.')

    def control(self, num_qubits):
        raise TypeError('Controlled Reset is not defined.')

    def iswrapper(self):
        return False

    @staticmethod
    def from_json(d):
        if d["name"] != "Reset":
            raise ValueError("Invalid json for Reset")

        return Reset()


# export operations
__all__ = ['Reset']
