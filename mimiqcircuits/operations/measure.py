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


class Measure(Operation):
    """Measure operation.

    Single qubit measurement operation in the computational basis.

    This operation is non-reversible

    Examples:

        Adding Measure operation to the Circuit (The qubits (first arg) and the bits (second arg) can be: range, list, tuple, set or int)

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Measure(),0,0)
        1-qubit circuit with 1 instructions:
        └── Measure @ q[0], c[0]
        <BLANKLINE>

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Measure(), range(0,3), range(0,3))
        3-qubit circuit with 3 instructions:
        ├── Measure @ q[0], c[0]
        ├── Measure @ q[1], c[1]
        └── Measure @ q[2], c[2]
        <BLANKLINE>
    """

    _name = "Measure"
    _num_bits = 1
    _num_qubits = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]

    def inverse(self):
        raise TypeError("Measure is not inversible")

    def power(self, p):
        raise TypeError("Measure^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled Measure is not defined.")

    def iswrapper(self):
        return False

    @staticmethod
    def from_json(d):
        if d["name"] != "Measure":
            raise ValueError("Invalid json for Measure")

        return Measure()

    def asciiwidth(self, _, bits):
        bit = bits[0]
        return max(3, 1 + len(str(bit)))

    def get_operation(self):
        return self


# export operations
__all__ = ["Measure"]
