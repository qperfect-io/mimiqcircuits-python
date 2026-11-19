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

from mimiqcircuits.operations.control import Control
import mimiqcircuits.lazy as lz
from mimiqcircuits.printutils import print_wrapped_parens
from mimiqcircuits.operations.gates.gate import Gate
from typing import Type, Dict

_inverse_decomposition_registry = {}
_inverse_aliases_registry = {}

# Registry for canonical Inverse subclasses: inner_gate_type -> subclass
_inverse_canonical_types: Dict[Type, Type] = {}


def register_inverse_alias(gate_type, name):
    """Register a display name for the inverse of a gate.

    ``Inverse(gate_type(...))`` then prints as ``name``.

    Args:
        gate_type: The gate class.
        name: The name to print.
    """
    _inverse_aliases_registry[(gate_type,)] = name


def register_inverse_decomposition(gate_type):
    """Decorator that registers a custom decomposition for the inverse of a gate.

    The decorated function is used to decompose every :class:`Inverse`
    whose wrapped gate has type key ``gate_type``. It is called as
    ``func(op, circ, qubits, bits, zvars)``, must push the decomposition of
    ``op`` on ``qubits`` to ``circ``, and must return ``circ``.

    Args:
        gate_type: Type key of the wrapped gate. For plain gates this is the
            gate class; for wrapped gates it is the value of their
            ``gettypekey()``, for example ``(Power, GateZ, 1/4)``.

    Returns:
        The decorator. It registers the function and returns it unchanged.
    """

    def decorator(decomp_func):
        key = (gate_type,)
        _inverse_decomposition_registry[key] = decomp_func
        return decomp_func

    return decorator


def canonical_inverse(inner_gate_type: Type):
    """Decorator that registers a named subclass of :class:`Inverse`.

    After registration, ``Inverse(g)`` with ``g`` an instance of
    ``inner_gate_type`` builds an instance of the decorated class instead of
    a plain ``Inverse``. This is how named gates such as :class:`GateSDG`
    (``Inverse(GateS())``) are defined.

    Registering a class for a type that already has one replaces it for the
    rest of the session.

    Args:
        inner_gate_type: Class of the wrapped gate.

    Returns:
        The decorator. It registers the class and returns it unchanged.

    Examples:
        >>> from mimiqcircuits import *
        >>> # GateSDG is registered as the inverse of GateS
        >>> isinstance(Inverse(GateS()), GateSDG)
        True
    """

    def decorator(cls: Type) -> Type:
        _inverse_canonical_types[inner_gate_type] = cls
        return cls

    return decorator


class Inverse(Gate):
    r"""Inverse of a gate.

    ``Inverse(gate)`` is the unitary :math:`U^\dagger`, where :math:`U` is the
    matrix of ``gate``. The inverse is kept as a wrapper: it is not
    rewritten into another gate when built.

    The constructor takes the gate, or a gate class followed by its
    constructor arguments.

    When a named subclass is registered for the gate type (see
    :func:`canonical_inverse`), the constructor returns an instance of that
    subclass: ``Inverse(GateS())`` is a :class:`GateSDG`.

    Args:
        operation: The gate to invert, or a gate class.
        *args: Constructor arguments when ``operation`` is a class.
        **kwargs: Constructor keyword arguments when ``operation`` is a class.

    Raises:
        ValueError: If ``operation`` is not a gate or gate class.

    Warning:
        Prefer the ``inverse`` method of a gate to building ``Inverse``
        directly. The method simplifies where it can, for example
        ``GateH().inverse()`` is ``GateH()`` and
        ``op.inverse().inverse()`` is ``op``.

    Examples:
        >>> from mimiqcircuits import *
        >>> # the generic form returns the named gate when one exists
        >>> isinstance(Inverse(GateS()), GateSDG)
        True
        >>> # the matrix is the conjugate transpose of the gate's matrix
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
            if isinstance(operation, type) and issubclass(operation, Gate):
                inner_type = operation
            elif isinstance(operation, Gate):
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
        if isinstance(operation, type) and issubclass(operation, Gate):
            op = operation(*args, **kwargs)
        elif isinstance(operation, Gate):
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

    def get_operation(self):
        return self.op

    def inverse(self):
        """Return the wrapped gate, undoing the inversion."""
        return self.op

    def getparams(self):
        return self.op.getparams()

    def _power(self, pwr):
        from mimiqcircuits.operations.power import Power
        return Power(self, pwr)

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
            return Control(num_controls, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def parallel(self, *args):
        from mimiqcircuits.operations.parallel import Parallel
        if len(args) == 0:
            return lz.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return Parallel(num_repeats, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def _matrix(self):
        return self.op.matrix().inv()

    def _matrix_numeric(self, *params):
        import numpy as np

        return np.linalg.inv(self.op.unwrappedmatrix())

    def evaluate(self, d):
        if not self.is_symbolic():
            return self
        return self.op.evaluate(d).inverse()

    def gettypekey(self):
        return (Inverse, self.op.gettypekey())

    def _decompose(self, circ, qubits, bits, zvars):
        from mimiqcircuits.circuit import Circuit
        key = self.gettypekey()[1:]
        if key in _inverse_decomposition_registry:
            return _inverse_decomposition_registry[key](self, circ, qubits, bits, zvars)

        newc = self.op._decompose(Circuit(), qubits, bits, zvars).inverse()
        circ.append(newc)
        return circ


__all__ = ["Inverse", "canonical_inverse"]
