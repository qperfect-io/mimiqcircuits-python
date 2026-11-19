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
import math
import numbers
from symengine import pi
from mimiqcircuits._sympycheck import _is_basic
import symengine as se
from mimiqcircuits.printutils import print_wrapped_parens
import numpy as np
from mimiqcircuits.lazy import control, parallel, power
from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.inverse import Inverse
from mimiqcircuits.operations.gates.gate import Gate
from typing import Type, Dict, Tuple, Any, Union

_power_decomposition_registry = {}
_power_aliases_registry = {}

# Registry for canonical Power subclasses: (inner_gate_type, exponent) -> subclass
_power_canonical_types: Dict[Tuple[Type, Any], Type] = {}


def register_power_alias(exponent, gate_type, name):
    """Register a display name for a gate raised to a given exponent.

    A :class:`Power` whose wrapped gate and exponent match the registered
    pair prints as ``name`` and reports ``isopalias() == True``.

    The first two arguments form the lookup key ``(gate type key,
    exponent)``: despite the parameter names, pass the gate type first and
    the exponent second, as in ``register_power_alias(GateZ, 1/4, "T")``.

    Args:
        exponent: The gate type key (see above).
        gate_type: The exponent (see above).
        name: The name to print.
    """
    key = (exponent, gate_type)
    _power_aliases_registry[key] = name


def register_power_decomposition(exponent, gate_type):
    """Decorator that registers a custom decomposition for a gate power.

    The decorated function is used to decompose every :class:`Power` whose
    wrapped gate and exponent match the key. It is called as
    ``func(op, circ, qubits, bits, zvars)``, must push the decomposition of
    ``op`` on ``qubits`` to ``circ``, and must return ``circ``.

    The two arguments form the lookup key ``(gate type key, exponent)``:
    despite the parameter names, pass the gate type first and the exponent
    second, as in ``@register_power_decomposition(GateZ, 1/4)``.

    Args:
        exponent: The gate type key (see above).
        gate_type: The exponent (see above).

    Returns:
        The decorator. It registers the function and returns it unchanged.
    """

    def decorator(decomp_func):
        key = (exponent, gate_type)
        _power_decomposition_registry[key] = decomp_func
        return decomp_func

    return decorator


def canonical_power(inner_gate_type: Type, exponent: Any):
    """Decorator that registers a named subclass of :class:`Power`.

    After registration, ``Power(g, exponent)`` with ``g`` an instance of
    ``inner_gate_type`` builds an instance of the decorated class instead of
    a plain ``Power``. This is how named gates such as :class:`GateS`
    (``Power(GateZ(), 1/2)``) are defined. The exponent is compared after
    normalization, so ``0.5`` and ``Fraction(1, 2)`` match the same entry.

    Registering a class for a pair that already has one replaces it for the
    rest of the session.

    Args:
        inner_gate_type: Class of the wrapped gate.
        exponent: The exponent, in normalized form (an ``int`` or a
            ``Fraction`` when the value is a simple fraction).

    Returns:
        The decorator. It registers the class and returns it unchanged.

    Examples:
        >>> from mimiqcircuits import *
        >>> # GateS is registered for GateZ to the power 1/2
        >>> isinstance(Power(GateZ(), 0.5), GateS)
        True
    """

    def decorator(cls: Type) -> Type:
        _power_canonical_types[(inner_gate_type, exponent)] = cls
        return cls

    return decorator


def _normalize_exponent(exponent: Any) -> Any:
    """Normalize an exponent to its canonical form.

    A float that is exactly a fraction with denominator at most 1000 becomes
    that fraction, or an integer when the denominator is 1 and it fits in
    int64, so that ``0.5`` and ``Fraction(1, 2)`` build the same operation and
    serialize to the same bytes.
    Other real values are returned unchanged.

    Raises:
        TypeError: if the exponent is not a real number, for example a
            symbolic or a complex value. A power must be known to build its
            matrix and to fold nested powers.
    """
    if _is_basic(exponent):
        if exponent.free_symbols or not exponent.is_real:
            raise TypeError(f"the exponent of a power must be a real number, not {exponent}")
        exponent = float(exponent)
    elif not isinstance(exponent, numbers.Real):
        raise TypeError(
            f"the exponent of a power must be a real number, not {exponent!r}"
        )
    if isinstance(exponent, float) and math.isfinite(exponent):
        frac = Fraction(exponent).limit_denominator(1000)
        if float(frac) == exponent:
            if frac.denominator != 1:
                return frac
            # Kept as a float past int64, which the protobuf integer field holds.
            if abs(frac.numerator) < 2**63 - 1:
                return frac.numerator
    return exponent


def _scale_by_exponent(value, exponent):
    """Return ``value * exponent`` for a parameter scaled by a power.

    SymEngine does not accept a ``Fraction`` operand, so a ``Fraction``
    exponent is turned into the equivalent SymEngine rational when ``value`` is
    symbolic. The result stays exact.
    """
    if isinstance(exponent, Fraction) and isinstance(value, se.Basic):
        exponent = se.Rational(exponent.numerator, exponent.denominator)
    return value * exponent


class Power(Gate):
    """Gate raised to a real power.

    ``Power(gate, p)`` is the unitary :math:`U^p`, where :math:`U` is the
    matrix of ``gate``. For non-integer ``p`` the principal matrix power is
    used.

    The constructor takes the gate (or a gate class followed by its
    constructor arguments) and the exponent. The exponent must be a real
    number: a float that is a simple fraction (denominator up to 1000) is
    stored as a ``Fraction`` or ``int``, so ``Power(g, 0.5)`` and
    ``Power(g, Fraction(1, 2))`` are the same operation. A power of a
    ``Power`` is folded into one: ``Power(Power(g, a), b)`` is
    ``Power(g, a * b)``.

    When a named subclass is registered for the combination (see
    :func:`canonical_power`), the constructor returns an instance of that
    subclass: ``Power(GateZ(), 0.5)`` is a :class:`GateS`. A folded power
    stays a plain ``Power``, even when a subclass is registered for the
    folded exponent.

    The ``power`` method of a gate, or the ``**`` operator, is the usual way
    to build one; it may simplify the result further.

    Args:
        operation: The gate to raise, or a gate class.
        exponent: The exponent, a real number.
        *args: Constructor arguments when ``operation`` is a class.
        **kwargs: Constructor keyword arguments when ``operation`` is a class.

    Raises:
        ValueError: If ``operation`` is not a gate or gate class.
        TypeError: If ``exponent`` is not a real number, for example a
            symbol or a complex number.

    Examples:
        >>> from mimiqcircuits import *
        >>> # the generic form returns the named gate when one exists
        >>> isinstance(Power(GateZ(), 0.5), GateS)
        True
        >>> c = Circuit()
        >>> # the square root of X prints with its own name, SX
        >>> c.push(Power(GateX(), 1/2), 1)
        2-qubit circuit with 1 instruction:
        └── SX @ q[1]
        <BLANKLINE>
        >>> c.push(Power(GateX(), 5), 1)
        2-qubit circuit with 2 instructions:
        ├── SX @ q[1]
        └── X**5 @ q[1]
        <BLANKLINE>
        >>> # an integer power decomposes into repeated applications
        >>> Power(GateX(), 3).decompose()
        1-qubit circuit with 3 instructions:
        ├── X @ q[0]
        ├── X @ q[0]
        └── X @ q[0]
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
            if isinstance(operation, type) and issubclass(operation, Gate):
                inner_type = operation
            elif isinstance(operation, Gate):
                inner_type = type(operation)
            else:
                # Let __init__ handle the error
                return object.__new__(cls)

            # __init__ folds Power(Power(g, a), b) into Power(g, a * b), so
            # look the canonical class up with the folded pair.
            if isinstance(operation, type) and issubclass(operation, Power):
                operation = operation(*args, **kwargs)
            if isinstance(operation, Power):
                inner_type = type(operation.op)
                exponent = _normalize_exponent(exponent) * operation.exponent

            # Normalize exponent for comparison
            norm_exp = _normalize_exponent(exponent)

            # Check for canonical subclass
            key = (inner_type, norm_exp)
            canonical_cls = _power_canonical_types.get(key)
            if canonical_cls is not None:
                return object.__new__(canonical_cls)

        return object.__new__(cls)

    def __init__(self, operation, exponent, *args, **kwargs):
        if isinstance(operation, type) and issubclass(operation, Gate):
            op = operation(*args, **kwargs)
        elif isinstance(operation, Gate):
            op = operation
        else:
            raise ValueError("Operation must be an Gate object or type.")

        super().__init__()

        self._exponent = _normalize_exponent(exponent)
        self._op = op
        self._num_qubits = op.num_qubits
        self._num_qregs = op.num_qregs
        self._qregsizes = op.qregsizes
        self._parnames = op.parnames
        if isinstance(op, Power):
            self._op = op.op
            self._exponent = _normalize_exponent(exponent * op.exponent)

    @property
    def op(self):
        """The gate being raised to a power. Read-only."""
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    @property
    def exponent(self):
        """The exponent, normalized to an ``int`` or ``Fraction`` when possible. Read-only."""
        return self._exponent

    @exponent.setter
    def exponent(self, power):
        raise ValueError("Cannot set exponent. Read only parameter.")

    def iswrapper(self):
        return True

    def get_operation(self):
        return self.op

    def _power(self, pwr):
        return self.op.power(pwr * self._exponent)

    def power(self, *args):
        """Raise to a further power, multiplying the exponents.

        Args:
            *args: The exponent. When omitted, a lazy expression is returned
                that takes the exponent later.

        Returns:
            Gate: ``op.power(exponent * p)``, or a lazy expression when
            called without arguments.

        Raises:
            ValueError: If more than one argument is given.

        Examples:
            >>> from mimiqcircuits import *
            >>> # (X^(1/2))^2 is X
            >>> Power(GateX(), 0.5).power(2)
            X
        """
        if len(args) == 0:
            return power(self)
        elif len(args) == 1:
            return self._power(_normalize_exponent(args[0]))
        else:
            raise ValueError("Invalid number of arguments.")

    def __pow__(self, exponent):
        return self.power(exponent)

    def inverse(self):
        """Return the inverse, ``Inverse(self)``."""
        return Inverse(self)

    def control(self, *args):
        if len(args) == 0:
            return control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return Control(num_controls, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def parallel(self, *args):
        from mimiqcircuits.operations.parallel import Parallel
        if len(args) == 0:
            return parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return Parallel(num_repeats, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def _matrix(self):
        import sympy as sp

        matrix = sp.Matrix(self.op.matrix().tolist())
        pow_matrix = matrix ** (self.exponent)
        return pow_matrix

    def _matrix_numeric(self, *params):
        from scipy.linalg import fractional_matrix_power

        inner = self.op.unwrappedmatrix()
        exp = self.exponent
        if isinstance(exp, (int, np.integer)) or (
            isinstance(exp, float) and exp.is_integer()
        ):
            return np.linalg.matrix_power(inner, int(exp))
        return np.asarray(fractional_matrix_power(inner, float(exp)), dtype=np.complex128)

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
        if not self.is_symbolic():
            return self
        return self.op.evaluate(d).power(self.exponent)

    def gettypekey(self):
        return (Power, self.op.gettypekey(), self.exponent)

    def _decompose(self, circ, qubits, bits, zvars):
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.parallel import Parallel
        from mimiqcircuits.operations.gates.generalized.phasegradient import (
            PhaseGradient,
        )
        key = self.gettypekey()[1:]
        if key in _power_decomposition_registry:
            return _power_decomposition_registry[key](self, circ, qubits, bits, zvars)

        if isinstance(self.exponent, int) and self.exponent >= 1:
            for _ in range(self.exponent):
                circ.push(self.op, *qubits)
            return circ

        if isinstance(self.op, Parallel):
            nq = self.op.op.num_qubits
            for i in range(self.op.num_repeats):
                q = [qubits[j] for j in range(i * nq, (i + 1) * nq)]
                circ.push(self.op.op.power(self.exponent), *q)
            return circ

        if isinstance(self.op, Inverse) and isinstance(self.op.op, Parallel):
            base = self.op.op.op.inverse()
            nq = base.num_qubits
            for i in range(self.op.op.num_repeats):
                q = [qubits[j] for j in range(i * nq, (i + 1) * nq)]
                circ.push(base.power(self.exponent), *q)
            return circ

        # PhaseGradient is a product of commuting phase gates, so its power
        # scales every phase.
        if isinstance(self.op, PhaseGradient):
            return self.op._decompose_power(circ, qubits, self.exponent)

        # (C-U)^p = C-(U^p): the controls pass through the power.
        if isinstance(self.op, Control):
            circ.push(
                Control(self.op.num_controls, self.op.op.power(self.exponent)),
                *qubits,
            )
            return circ

        # A single-gate decomposition can be raised to the power directly.
        # Anything longer would need the matrix power of the whole circuit.
        cop = self.op._decompose(Circuit(), qubits, bits, zvars)

        if len(cop) == 1:
            inst = cop.instructions[0]
            circ.push(inst.operation._power(self.exponent), *inst.qubits)
            return circ

        from mimiqcircuits.decomposition.abstract import DecompositionError

        raise DecompositionError(
            f"Cannot decompose {self}: the decomposition of its base has more than one gate."
        )

    def decompose(self):
        from mimiqcircuits.circuit import Circuit
        return self._decompose(
            Circuit(),
            range(self.num_qubits),
            range(self.num_bits),
            range(self.num_zvars),
        )


# export operation
__all__ = ["Power", "canonical_power"]
