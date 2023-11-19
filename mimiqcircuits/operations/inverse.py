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

import mimiqcircuits as mc
import sympy as sp
import symengine as se
from mimiqcircuits.printutils import print_wrapped_parens


class Inverse(mc.Operation):
    """Inverse of the wrapped quantum operation.

    The inversion is not performed right away, but only when the circuit is
    cached or executed.

    .. warning::

        Users should not use `Inverse` directly but rather the `inverse` method,
        which performs all the necessary simplifications (e.g., `op.inverse().inverse() == op`)

    Examples:
        >>> from mimiqcircuits import *
        >>> Inverse(GateP(1)).matrix()
        [1, 0]
        [0, exp(-I)]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(Inverse(GateP(1)), 1)
        2-qubit circuit with 1 instructions:
        └── P(1)† @ q1
    """
    _name = 'Inverse'
    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0

    _op = None

    def __init__(self, operation, *args, **kwargs):
        if isinstance(operation, type) and issubclass(operation, mc.Operation):
            op = operation(*args, **kwargs)
        elif isinstance(operation, mc.Operation):
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
        return mc.Power(self, pwr)

    def control(self, num_controls):
        return mc.Control(num_controls, self)

    def matrix(self):
        return se.Matrix(sp.simplify(self.op.matrix().inv()))

    def evaluate(self, d):
        return self.op.evaluate(d).inverse()

    def _decompose(self, circ, qubits, bits):
        newc = self.op._decompose(mc.Circuit(), qubits, bits).inverse()
        circ.append(newc)
        return circ


__all__ = ['Inverse']
