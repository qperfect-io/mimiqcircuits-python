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
import mimiqcircuits.operations.gates.gate as mcg
import mimiqcircuits.operations.control as mctrl
import mimiqcircuits.operations.power as mpwr
import mimiqcircuits.circuit as mcc
from mimiqcircuits.printutils import print_wrapped_parens


class Inverse(Operation):
    """Inverse of the wrapped quantum operation.

    The inversion is not performed right away, but only when the circuit is
    cached or executed.

    .. warning::

        User should not use directly `Inverse` but rather the `inverse` method,
        which performs already all the simplifications (e.g. `op.inverse().inverse() == op`)

    Examples:
    """
    _name = 'Inverse'
    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0

    _op = None

    def __init__(self, operation, *args, **kwargs):
        if isinstance(operation, type) and issubclass(operation, Operation):
            op = operation(*args, **kwargs)
        elif isinstance(operation, Operation):
            op = operation
        else:
            raise ValueError("Operation must be an Operation object or type.")

        if op.num_bits != 0:
            raise ValueError("Cannot inverte operation with classical bits.")

        self.op = op
        super().__init__()
        self._num_qubits = op.num_qubits
        self._num_qregs = op._num_qregs
        self._qregsizes = op._qregsizes
        self._parnames = op.parnames

    def __str__(self):
        return f'{print_wrapped_parens(self.op)}†'

    def iswrapper(self):
        return True

    def inverse(self):
        return self.op

    def power(self, pwr):
        return mpwr.Power(self, pwr)

    def control(self, num_controls):
        return mctrl.Control(num_controls, self)

    def matrix(self):
        return self.op.matrix().inv()

    def evaluate(self, d):
        return self.op.evaluate(d).inverse()

    def _decompose(self, circ, qubits, bits):
        newc = self.op._decompose(mcc.Circuit(), qubits, bits).inverse()
        circ.append(newc)
        return circ


__all__ = ['Inverse']
