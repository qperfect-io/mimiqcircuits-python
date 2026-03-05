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
"""Power operation."""

from fractions import Fraction
from symengine import pi
from mimiqcircuits.printutils import print_wrapped_parens
import sympy as sp
import numpy as np
import mimiqcircuits as mc
from mimiqcircuits.operations.gates.gate import Gate
from typing import Type, Dict, Tuple, Any, Union

_power_decomposition_registry = {}
_power_aliases_registry = {}

# Registry for canonical Power subclasses: (inner_gate_type, exponent) -> subclass
_power_canonical_types: Dict[Tuple[Type, Any], Type] = {}


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


def canonical_power(inner_gate_type: Type, exponent: Any):
    """Decorator to register a canonical Power subclass.

    When Power(inner_gate_type(), exponent) is called, it will
    return an instance of the decorated subclass instead.

    Args:
        inner_gate_type: Type of the inner gate
        exponent: The exponent value

    Returns:
        Decorator that registers the class

    Example:
        >>> from mimiqcircuits import *
        >>> @canonical_power(GateZ, Fraction(1, 2))
        ... class GateS(Power):
        ...     pass
        >>> isinstance(Power(GateZ(), 0.5), GateS)
        True
    """

    def decorator(cls: Type) -> Type:
        _power_canonical_types[(inner_gate_type, exponent)] = cls
        return cls

    return decorator


def _normalize_exponent(exponent: Any) -> Any:
    """Normalize exponent to a canonical form for comparison.

    Converts floats that are exact fractions to Fraction objects.
    """
    if isinstance(exponent, float):
        frac = Fraction(exponent).limit_denominator(1000)
        if float(frac) == exponent:
            return frac
    return exponent


class Power(Gate):
    """Power operation.

    Represents a Power operation raised to a specified exponent.

    When a canonical subclass is registered (e.g., GateS for Power(GateZ(), 1/2)),
    constructing Power with matching arguments will return an instance of that
    subclass. This enables isinstance() checks to work correctly:


    Examples:
        >>> from mimiqcircuits import *
        >>> isinstance(Power(GateZ(), 0.5), GateS)
        True
        >>> c= Circuit()
        >>> c.push(Power(GateX(),1/2),1)
        2-qubit circuit with 1 instruction:
        └── SX @ q[1]
        <BLANKLINE>
        >>> c.push(Power(GateX(),5),1)
        2-qubit circuit with 2 instructions:
        ├── SX @ q[1]
        └── X**5 @ q[1]
        <BLANKLINE>
        >>> c.decompose()
        2-qubit circuit with 9 instructions:
        ├── U(0, 0, (-1/2)*pi, 0.0) @ q[1]
        ├── U((1/2)*pi, 0, pi, 0.0) @ q[1]
        ├── U(0, 0, (-1/2)*pi, 0.0) @ q[1]
        ├── U(0, 0, 0, (1/4)*pi) @ q[1]
        ├── U(pi, 0, pi, 0.0) @ q[1]
        ├── U(pi, 0, pi, 0.0) @ q[1]
        ├── U(pi, 0, pi, 0.0) @ q[1]
        ├── U(pi, 0, pi, 0.0) @ q[1]
        └── U(pi, 0, pi, 0.0) @ q[1]
        <BLANKLINE>
    """

    _name = "Power"

    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0

    _op = None
    _parnames = ("exponent",)

    def __new__(cls, operation: Union[Type[Gate], Gate] = None, exponent: Any = None, *args, **kwargs):
        """Create a Power instance, returning canonical subclass if registered.

        If a canonical subclass is registered for (type(operation), exponent),
        an instance of that subclass is returned instead of a plain Power.

        Args:
            operation: Gate operation or gate class to raise to a power
            exponent: The exponent value
            *args: Arguments to pass to operation constructor if a class is provided
            **kwargs: Keyword arguments to pass to operation constructor

        Returns:
            Instance of Power or a registered canonical subclass
        """
        # Only intercept direct Power() calls, not subclass calls
        if cls is Power:
            if operation is None or exponent is None:
                # Let __init__ handle the error
                return object.__new__(cls)

            # Resolve the operation to get its type
            if isinstance(operation, type) and issubclass(operation, mc.Gate):
                inner_type = operation
            elif isinstance(operation, mc.Gate):
                inner_type = type(operation)
            else:
                # Let __init__ handle the error
                return object.__new__(cls)

            # Normalize exponent for comparison
            norm_exp = _normalize_exponent(exponent)

            # Check for canonical subclass
            key = (inner_type, norm_exp)
            canonical_cls = _power_canonical_types.get(key)
            if canonical_cls is not None:
                return object.__new__(canonical_cls)

        return object.__new__(cls)

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

        if isinstance(self.op, mc.Parallel):
            nq = self.op.op.num_qubits
            for i in range(self.op.num_repeats):
                q = [qubits[j] for j in range(i * nq, (i + 1) * nq)]
                circ.push(self.op.op.power(self.exponent), *q)
            return circ

        if isinstance(self.op, mc.Inverse) and isinstance(self.op.op, mc.Parallel):
            base = self.op.op.op.inverse()
            nq = base.num_qubits
            for i in range(self.op.op.num_repeats):
                q = [qubits[j] for j in range(i * nq, (i + 1) * nq)]
                circ.push(base.power(self.exponent), *q)
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
__all__ = ["Power", "canonical_power"]
