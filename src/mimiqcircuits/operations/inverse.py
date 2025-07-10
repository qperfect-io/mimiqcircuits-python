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

import mimiqcircuits as mc
import mimiqcircuits.lazy as lz
from mimiqcircuits.printutils import print_wrapped_parens
from mimiqcircuits.operations.gates.gate import Gate

_inverse_decomposition_registry = {}
_inverse_aliases_registry = {}


def register_inverse_alias(exponent, gate_type, name):
    """Register an alias for an inverse gate definition"""
    key = (exponent, gate_type)
    _inverse_aliases_registry[key] = name


def register_inverse_decomposition(gate_type):
    """Decorator to register a decomposition function for a gate type"""

    def decorator(decomp_func):
        key = (gate_type,)
        _inverse_decomposition_registry[key] = decomp_func
        return decomp_func

    return decorator


class Inverse(Gate):
    """Inverse of the wrapped quantum operation.

    The inversion is not performed right away, but only when the circuit is
    cached or executed.

    .. warning::

        Users should not use `Inverse` directly but rather the `inverse` method,
        which performs all the necessary simplifications (e.g., `op.inverse().inverse() == op`)

    Examples:
        >>> from mimiqcircuits import *
        >>> Inverse(GateP(1)).matrix()
        [1.0, 0]
        [0, 0.54030230586814 - 0.841470984807897*I]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(Inverse(GateP(1)), 1)
        2-qubit circuit with 1 instructions:
        └── P(1)† @ q[1]
        <BLANKLINE>
    """

    _name = "Inverse"
    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0

    _op = None

    def __init__(self, operation, *args, **kwargs):
        if isinstance(operation, type) and issubclass(operation, mc.Gate):
            op = operation(*args, **kwargs)
        elif isinstance(operation, mc.Gate):
            op = operation
        else:
            raise ValueError("Operation must be an Gate object or type.")

        if op.num_bits != 0:
            raise ValueError("Cannot inverte operation with classical bits.")

        self.op = op
        super().__init__()
        self._num_qubits = op.num_qubits
        self._num_qregs = op._num_qregs
        self._qregsizes = op._qregsizes
        self._parnames = op.parnames

    def isopalias(self):
        key = self.gettypekey()[1:]
        return key in _inverse_aliases_registry

    def __str__(self):
        key = self.gettypekey()[1:]
        if key in _inverse_aliases_registry:
            return _inverse_aliases_registry[key]

        return f"{print_wrapped_parens(self.op)}†"

    def iswrapper(self):
        return True

    def inverse(self):
        return self.op

    def getparams(self):
        return self.op.getparams()

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

    def control(self, *args):
        if len(args) == 0:
            return lz.control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return mc.Control(num_controls, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def parallel(self, *args):
        if len(args) == 0:
            return lz.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return mc.Parallel(num_repeats, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def _matrix(self):
        return self.op.matrix().inv()

    def evaluate(self, d):
        return self.op.evaluate(d).inverse()

    def gettypekey(self):
        return (Inverse, self.op.gettypekey())

    def _decompose(self, circ, qubits, bits, zvars):
        key = self.gettypekey()[1:]
        if key in _inverse_decomposition_registry:
            return _inverse_decomposition_registry[key](self, circ, qubits, bits, zvars)

        newc = self.op._decompose(mc.Circuit(), qubits, bits, zvars).inverse()
        circ.append(newc)
        return circ


__all__ = ["Inverse"]
