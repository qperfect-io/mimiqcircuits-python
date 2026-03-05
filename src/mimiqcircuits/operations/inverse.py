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
"""Inverse operation."""

import mimiqcircuits as mc
import mimiqcircuits.lazy as lz
from mimiqcircuits.printutils import print_wrapped_parens
from mimiqcircuits.operations.gates.gate import Gate
from typing import Type, Dict, Tuple, Any, Union

_inverse_decomposition_registry = {}
_inverse_aliases_registry = {}

# Registry for canonical Inverse subclasses: inner_gate_type -> subclass
_inverse_canonical_types: Dict[Type, Type] = {}


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


def canonical_inverse(inner_gate_type: Type):
    """Decorator to register a canonical Inverse subclass.

    When Inverse(inner_gate_type()) is called, it will
    return an instance of the decorated subclass instead.

    Args:
        inner_gate_type: Type of the inner gate

    Returns:
        Decorator that registers the class

    Example:
        >>> from mimiqcircuits import *
        >>> @canonical_inverse(GateS)
        ... class GateSDG(Inverse):
        ...     pass
        >>> isinstance(Inverse(GateS()), GateSDG)
        True
    """

    def decorator(cls: Type) -> Type:
        _inverse_canonical_types[inner_gate_type] = cls
        return cls

    return decorator


class Inverse(Gate):
    """Inverse of the wrapped quantum operation.

    The inversion is not performed right away, but only when the circuit is
    cached or executed.

    When a canonical subclass is registered (e.g., GateSDG for Inverse(GateS())),
    constructing Inverse with matching arguments will return an instance of that
    subclass. This enables isinstance() checks to work correctly:


    .. warning::

        Users should not use `Inverse` directly but rather the `inverse` method,
        which performs all the necessary simplifications (e.g., `op.inverse().inverse() == op`)

    Examples:
        >>> from mimiqcircuits import *
        >>> isinstance(Inverse(GateS()), GateSDG)
        True
        >>> Inverse(GateP(1)).matrix()
        [1.0, 0]
        [0, 0.54030230586814 - 0.841470984807897*I]
        <BLANKLINE>
        >>> c = Circuit()
        >>> c.push(Inverse(GateP(1)), 1)
        2-qubit circuit with 1 instruction:
        └── P(1)† @ q[1]
        <BLANKLINE>
    """

    _name = "Inverse"
    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0

    _op = None

    def __new__(cls, operation=None, *args, **kwargs):
        """Create an Inverse instance, returning canonical subclass if registered.

        If a canonical subclass is registered for type(operation),
        an instance of that subclass is returned instead of a plain Inverse.

        Args:
            operation: Gate operation or gate class to invert
            *args: Arguments to pass to operation constructor if a class is provided
            **kwargs: Keyword arguments to pass to operation constructor

        Returns:
            Instance of Inverse or a registered canonical subclass
        """
        # Only intercept direct Inverse() calls, not subclass calls
        if cls is Inverse:
            if operation is None:
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

            # Check for canonical subclass
            canonical_cls = _inverse_canonical_types.get(inner_type)
            if canonical_cls is not None:
                return object.__new__(canonical_cls)

        return object.__new__(cls)

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


__all__ = ["Inverse", "canonical_inverse"]
