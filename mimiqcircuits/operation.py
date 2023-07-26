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

from abc import ABC, abstractmethod
import copy


class Operation(ABC):
    _num_qubits = None
    _num_bits = None
    _name = None

    @property
    def num_qubits(self):
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, value):
        raise ValueError('Cannot set num_qubits. Read only parameter.')

    @property
    def num_bits(self):
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value):
        raise ValueError('Cannot set num_qubits. Read only parameter.')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise ValueError('Cannot set name. Read only parameter.')

    @abstractmethod
    def inverse(self):
        pass

    def to_json(self):
        return {'name': self.name, 'N': self.num_qubits, 'M': self.num_bits}

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return False
        return (self.name == other.name)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


# export operations
__all__ = ['Operation']
