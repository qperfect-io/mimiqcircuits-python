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


class Measure(Operation):
    """Measure operation.
    
    Single qubit measurement operation in the computational basis.
    
    This operation is non-reversible

    Examples:
        >>> from  mimiqcircuits import *
        >>> c=Circuit()
        >>> c.push(GateX(),0)
        >>> c.push(Measure(),0,0)

        >>> 1-qubit circuit with 2 instructions:
            ├── X @ q0
            └── Measure @ q0, c0
    """
    _name = 'Measure'
    _num_bits = 1
    _num_qubits = 1

    def inverse(self):
        raise TypeError('Measure is not inversible')

    @staticmethod
    def from_json(d):
        if d["name"] != "Measure":
            raise ValueError("Invalid json for Measure")

        return Measure()
    
    def __eq__(self, other):
        if not isinstance(other, Measure):
            return False
        return (self.name == other.name)


# export operations
__all__ = ['Measure']

