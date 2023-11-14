#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use thas file except in compliance with the License.
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
import mimiqcircuits.circuit as mc


class Operation(ABC):
    _name = None

    _num_qubits = None
    _num_qregs = 1
    _qregsizes = None

    _num_bits = None
    _num_cregs = 1
    _cregsizes = None

    _parnames = ()

    @property
    def num_qubits(self):
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, value):
        raise ValueError('Cannot set num_qubits. Read only parameter.')

    @property
    def num_qregs(self):
        return self._num_qregs

    @num_qregs.setter
    def num_qregs(self, value):
        raise ValueError('Cannot set num_qregs. Read only parameter.')

    @property
    def num_bits(self):
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value):
        raise ValueError('Cannot set num_bits. Read only parameter.')

    @property
    def num_cregs(self):
        return self._num_cregs

    @num_cregs.setter
    def num_cregs(self, value):
        raise ValueError('Cannot set num_cregs. Read only parameter.')

    @property
    def qregsizes(self):
        return self._qregsizes

    @qregsizes.setter
    def qregsizes(self, value):
        raise ValueError('Cannot set qregsizes. Read only parameter.')

    @property
    def cregsizes(self):
        return self._cregsizes

    @cregsizes.setter
    def cregsizes(self, value):
        raise ValueError('Cannot set cregsizes. Read only parameter.')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise ValueError('Cannot set name. Read only parameter.')

    @ property
    def parnames(self):
        return self._parnames

    @ parnames.setter
    def parnames(self, value):
        raise ValueError('Cannot set parnames. Read only parameter.')

    def getparams(self):
        return [getattr(self, pn) for pn in self.parnames]

    def getparam(self, pn):
        if pn not in self.parnames:
            raise ValueError(f'Parameter {pn} not found.')
        return getattr(self, pn)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, type(self)) and\
            self.__dict__ == other.__dict__

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    @abstractmethod
    def iswrapper(self):
        pass

    def isopalias(self):
        return False

    def _decompose(self, circ, qubits, bits):
        circ.push(self, *qubits, *bits)

    def decompose(self):
        return self._decompose(mc.Circuit(), range(self.num_qubits), range(self.num_bits))

    def evaluate(self, d):
        return self


# export operations
__all__ = ['Operation']
