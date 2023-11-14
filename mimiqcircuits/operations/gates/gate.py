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

from abc import abstractmethod
from mimiqcircuits.operations.operation import (
    Operation
)

import mimiqcircuits as mc


class Gate(Operation):
    _name = None

    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0
    _cregsizes = ()

    @abstractmethod
    def matrix(self):
        pass

    def iswrapper(self):
        return False

    def inverse(self):
        return mc.Inverse(self)

    def power(self, pwr):
        return mc.Power(self, pwr)

    def control(self, num_controls):
        return mc.Control(num_controls, self)

    def __str__(self):
        pars = ''
        if len(self.parnames) != 0:
            pars += '('
            pars += ', '.join([f'{getattr(self, pn)}' for pn in self.parnames])
            pars += ')'
        return self.name + pars

    def __repr__(self):
        return str(self)

    def evaluate(self, d):
        if len(self.parnames) == 0:
            return self

        else:
            params = self.getparams()

            for i in range(len(params)):
                if isinstance(params[i], (int, float)):
                    continue

                params[i] = params[i].subs(d)
                if isinstance(self, mc.GPhase):
                    return  type(self)(self.num_qubits, *params)

            return type(self)(*params)
        
        
__all__ = ['Gate']
