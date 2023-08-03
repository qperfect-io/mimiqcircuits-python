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


class Barrier(Operation):
    """Barrier operation.

    A barrier is a special operation that does not affect the quantum state or the
    execution of a circuit, but it prevents compression or optimization operation
    from being applied across it.
    
    Examples:
            >>> from  mimiqcircuits import *
            >>> c=Circuit()
            >>> c.push(GateX(),0)
            >>> c.push(Barrier(),0)

            >>> 1-qubit circuit with 2 instructions:
                ├── X @ q0
                └── Barrier @ q0
    """
    _name = 'Barrier'
    _num_qubits = None
    _num_bits = 0

    def __init__(self, num_qubits=1):
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

    def inverse(self):
        return self

    @staticmethod
    def from_json(d):
        if d["name"] != "Barrier":
            raise ValueError("Invalid json for Barrier")

        return Barrier()

    def __eq__(self, other):
        if not isinstance(other, Barrier):
            return False
        return (self.name == other.name)
    
# export operations
__all__ = ['Barrier']
