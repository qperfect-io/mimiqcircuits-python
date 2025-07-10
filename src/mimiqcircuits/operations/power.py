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

from fractions import Fraction
from symengine import pi
from mimiqcircuits.printutils import print_wrapped_parens
import sympy as sp
import numpy as np
import mimiqcircuits as mc
from mimiqcircuits.operations.gates.gate import Gate

_power_decomposition_registry = {}
_power_aliases_registry = {}


def register_power_alias(exponent, gate_type, name):
    """Register an alias for a power gate definition"""
    key = (exponent, gate_type)
    _power_aliases_registry[key] = name


def register_power_decomposition(exponent, gate_type):
    """Decorator to register a decomposition function for a gate type"""

    def decorator(decomp_func):
        key = (exponent, gate_type)
        _power_decomposition_registry[key] = decomp_func
        return decomp_func

    return decorator


class Power(Gate):
    """Power operation.

    Represents a Power operation raised to a specified exponent.

    Examples:
        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Power(GateX(),1/2),1)
        2-qubit circuit with 1 instructions:
        └── SX @ q[1]
        <BLANKLINE>
        >>> c.push(Power(GateX(),5),1)
        2-qubit circuit with 2 instructions:
        ├── SX @ q[1]
        └── X**5 @ q[1]
        <BLANKLINE>
        >>> c.decompose()
        2-qubit circuit with 9 instructions:
        ├── S† @ q[1]
        ├── H @ q[1]
        ├── S† @ q[1]
        ├── U(0, 0, 0, (1/4)*pi) @ q[1]
        ├── X @ q[1]
        ├── X @ q[1]
        ├── X @ q[1]
        ├── X @ q[1]
        └── X @ q[1]
        <BLANKLINE>
    """

    _name = "Power"

    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0

    _op = None
    _parnames = ("exponent",)

    def __init__(self, operation, exponent, *args, **kwargs):
        if isinstance(operation, type) and issubclass(operation, mc.Gate):
            op = operation(*args, **kwargs)
        elif isinstance(operation, mc.Gate):
            op = operation
        else:
            raise ValueError("Operation must be an Gate object or type.")

        if self.num_bits != 0:
            raise ValueError("Power operation cannot act on classical bits.")

        super().__init__()

        self._exponent = exponent
        self._op = op
        self._num_qubits = op.num_qubits
        self._num_qregs = op.num_qregs
        self._qregsizes = op.qregsizes
        self._parnames = op.parnames
        if isinstance(op, mc.Power):
            self._op = op.op
            self._exponent = exponent * op.exponent

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    @property
    def exponent(self):
        return self._exponent

    @exponent.setter
    def exponent(self, power):
        raise ValueError("Cannot set exponent. Read only parameter.")

    def iswrapper(self):
        return True

    def _power(self, pwr):
        return self.op.power(pwr * self._exponent)

    def power(self, *args):
        if len(args) == 0:
            return mc.power(self)
        elif len(args) == 1:
            exponent = args[0]
            return self._power(exponent)
        else:
            raise ValueError("Invalid number of arguments.")

    def __pow__(self, exponent):
        return self.power(exponent)

    def inverse(self):
        return mc.Inverse(self)

    def control(self, *args):
        if len(args) == 0:
            return mc.control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return mc.Control(num_controls, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def parallel(self, *args):
        if len(args) == 0:
            return mc.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return mc.Parallel(num_repeats, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def _matrix(self):
        matrix = sp.Matrix(self.op.matrix().tolist())
        pow_matrix = matrix ** (self.exponent)
        return pow_matrix

    def getparams(self):
        return self.op.getparams()

    def isopalias(self):
        key = self.gettypekey()[1:]
        return key in _power_aliases_registry

    def __str__(self):
        key = self.gettypekey()[1:]
        if key in _power_aliases_registry:
            return _power_aliases_registry[key]

        fraction = Fraction(self.exponent).limit_denominator(100)

        if float(fraction) == self.exponent and int(fraction) != self.exponent:
            return f"{print_wrapped_parens(self.op)}**({fraction})"

        if self.exponent == pi:
            return f"{print_wrapped_parens(self.op)}**π"
        if self.exponent == -pi:
            return f"{print_wrapped_parens(self.op)}**(-π)"

        divpi = Fraction(self.exponent / np.pi).limit_denominator(100)

        if float(divpi) == self.exponent / np.pi:
            if divpi == 1:
                return f"{print_wrapped_parens(self.op)}**π"

            if divpi == -1:
                return f"{print_wrapped_parens(self.op)}**(-1)"

            if divpi > 0:
                return f"{print_wrapped_parens(self.op)}**({divpi} * π)"

            return f"{print_wrapped_parens(self.op)}**(({divpi}) * π)"

        if self.exponent < 0:
            return f"{print_wrapped_parens(self.op)}**({self.exponent})"

        return f"{print_wrapped_parens(self.op)}**{self.exponent}"

    def __repr__(self):
        return self.__str__()

    def evaluate(self, d):
        exponent = self.exponent
        return self.op.evaluate(d).power(exponent)

    def gettypekey(self):
        return (Power, self.op.gettypekey(), self.exponent)

    def _decompose(self, circ, qubits, bits, zvars):
        key = self.gettypekey()[1:]
        if key in _power_decomposition_registry:
            return _power_decomposition_registry[key](self, circ, qubits, bits, zvars)

        if isinstance(self.exponent, int) and self.exponent >= 1:
            for _ in range(self.exponent):
                circ.push(self.op, *qubits)
            return circ

        # try to decompose,
        # if there is only a gate, maybe it is ok
        # if the gates are all diagonal then we can continue
        # otherwise just do nothing and push the same thing
        cop = self.op._decompose(mc.Circuit(), qubits, bits, zvars)

        if len(cop) == 1:
            circ.push(cop.instructions[0].operation._power(self.exponent), *qubits)
            return circ

        circ.push(self, *qubits)

        return circ

    def decompose(self):
        return self._decompose(
            mc.Circuit(),
            range(self.num_qubits),
            range(self.num_bits),
            range(self.num_zvars),
        )


# export operation
__all__ = ["Power"]
