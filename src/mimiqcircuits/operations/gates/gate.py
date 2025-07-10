#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
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

from mimiqcircuits.operations.operator import AbstractOperator
import mimiqcircuits.lazy as lz

import mimiqcircuits as mc
from mimiqcircuits.operations.repeat import repeat


class Gate(AbstractOperator):
    _name = None

    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0
    _cregsizes = ()

    def _matrix(self):
        raise ValueError(
            "This is a parrent class of simple gates and does not have matrix"
        )

    def iswrapper(self):
        return False

    def inverse(self):
        return mc.Inverse(self)

    def _power(self, pwr):
        return mc.Power(self, pwr)

    def power(self, *args):
        if len(args) == 0:
            return lz.power(self)
        elif len(args) == 1:
            pwr = args[0]
            return self._power(pwr)
        else:
            raise ValueError("Invalid number of arguments.")

    def __pow__(self, pwr):
        return self.power(pwr)

    def _control(self, num_controls):
        return mc.Control(num_controls, self)

    def control(self, *args):
        if len(args) == 0:
            return lz.control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return self._control(num_controls)
        else:
            raise ValueError("Invalid number of arguments.")

    def _parallel(self, num_repeats):
        return mc.Parallel(num_repeats, self)

    def parallel(self, *args):
        if len(args) == 0:
            return lz.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return self._parallel(num_repeats)
        else:
            raise ValueError("Invalid number of arguments.")

    def _repeat(self, num_repeats):
        return mc.Repeat(num_repeats, self)

    def repeat(self, *args):
        if len(args) == 0:
            return repeat(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return self._repeat(num_repeats)
        else:
            raise ValueError("Invalid number of arguments.")

    def __str__(self):
        pars = ""
        if len(self.parnames) != 0:
            pars += "("
            pars += ", ".join([f"{getattr(self, pn)}" for pn in self.parnames])
            pars += ")"
        return self.name + pars

    def __repr__(self):
        return str(self)

    def evaluate(self, d):
        if len(self.parnames) == 0:
            return self

        params = self.getparams()

        for i in range(len(params)):
            if isinstance(params[i], (int, float)):
                continue

            params[i] = params[i].subs(d)

        return type(self)(*params)

    def gettypekey(self):
        return type(self)

    @staticmethod
    def isunitary():
        return True


__all__ = ["Gate"]
