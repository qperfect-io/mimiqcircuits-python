
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
import mimiqcircuits.lazy as lz
from symengine import eye, Symbol


class GPhase(mcg.Gate):
    """GPhase

    Examples:

        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> c = Circuit()
        >>> lmbda = Symbol('lambda')
        >>> GPhase(1,lmbda).matrix()
        [exp(I*lambda), 0]
        [0, exp(I*lambda)]
        <BLANKLINE>
    """
    _name = 'GPhase'
    _num_qregs = 1
    _parnames = ('lmbda',)

    def __init__(self, num, lmbda):
        super().__init__()
        self._num_qubits = num
        self._qregsizes = [num,]
        self._params = [num]

        self.lmbda = lmbda

    def __new__(cls, *args):
        if len(args) == 0:
            return lz.LazyExpr(GPhase, lz.LazyArg(), lz.LazyArg())
        elif len(args) == 1:
            lmbda = args[0]
            return lz.LazyExpr(GPhase, lz.LazyArg(), lmbda)
        elif len(args) == 2:
            return object.__new__(cls)
        else:
            raise ValueError("Invalid number of arguments.")

    def inverse(self):
        return GPhase(self.num_qubits, -self.lmbda)

    def power(self, pwr):
        return GPhase(self.num_qubits, self.lmbda * pwr)

    def _matrix(self):
        hdim = 2 ** self.num_qubits
        result = cis(self.lmbda) * eye(hdim)
        return result

    def __str__(self):
        return f'GPhase({self.lmbda})'

    def evaluate(self, d):
        if len(self.parnames) == 0:
            return self

        evaluated_params = [p.subs(d) if isinstance(
            p, Symbol) else p for p in self.getparams()]

        lmbda = evaluated_params[0] if isinstance(
            self.lmbda, Symbol) else self.lmbda

        return GPhase(self.num_qubits, lmbda)


__all__ = ['GPhase']
