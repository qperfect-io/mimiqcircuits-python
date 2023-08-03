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

from mimiqcircuits.operation import Operation


class Reset(Operation):
    """Reset operation.
    
    Quantum operation that resets the status of one qubit to the :math:`\\ket{0}` state.

    This operation is non-reversible.

    Examples:
            >>> from  mimiqcircuits import *
            >>> c=Circuit()
            >>> c.push(GateX(),0)
            >>> c.push(Reset(),0)

            >>> 1-qubit circuit with 2 instructions:
                ├── X @ q0
                └── Reset @ q0
    """
    _name = 'Reset'
    _num_qubits = 1
    _num_bits = 0

    def inverse(self):
        raise TypeError('Reset is not inversible')

    @staticmethod
    def from_json(d):
        if d["name"] != "Reset":
            raise ValueError("Invalid json for Reset")

        return Reset()
    
    def __eq__(self, other):
        if not isinstance(other, Reset):
            return False
        return (self.name == other.name)


# export operations
__all__ = ['Reset']
