
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

import mimiqcircuits.operations.gates.gate as mcg
from mimiqcircuits.matrices import cis
from symengine import eye


class GPhase(mcg.Gate):
    """GPhase
    """
    _name = 'GPhase'
    _num_qregs = 1
    _parnames = ('lmbda',)

    def __init__(self, num, lmbda):
        super().__init__()
        self._num_qubits = num
        self._qregsizes = [num,]

        self.lmbda = lmbda

    def inverse(self):
        return GPhase(self.num_qubits, -self.lmbda)

    def power(self, pwr):
        return GPhase(self.num_qubits, self.lmbda * pwr)

    def matrix(self):
        hdim = 2 ** self.num_qubits
        result = cis(self.lmbda) * eye(hdim)
        return result

    def __str__(self):
        return f'GPhase(lmbda={self.lmbda})'
    
    
__all__ = ['GPhase']
