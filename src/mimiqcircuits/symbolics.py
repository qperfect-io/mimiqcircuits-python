#
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

import symengine as se
from symengine import Symbol, Number, sympify
from mimiqcircuits.operations.operation import Operation


class UndefinedValue(Exception):
    """Raised when a symbolic expression cannot be converted to a numeric value."""

    def __init__(self, expr):
        self.expr = expr
        super().__init__(f"Cannot convert symbolic expression to numeric value: {expr}")


def unwrapvalue(g):
    """Convert a number or a SymEngine expression to a Python number.

    Python ``int``, ``float`` and ``complex`` values are returned unchanged.
    A SymEngine expression without free symbols is evaluated: a real result
    is returned as ``int`` when it is a whole number and as ``float``
    otherwise, a complex result as ``complex``. Other objects are passed to
    ``float``.

    Args:
        g: A number or a SymEngine expression.

    Returns:
        int, float or complex: The numeric value.

    Raises:
        UndefinedValue: If ``g`` has free symbols or cannot be converted.

    Examples:
        >>> from symengine import pi, I, Integer, Symbol
        >>> from mimiqcircuits import *
        >>> unwrapvalue(pi)
        3.141592653589793
        >>> unwrapvalue(Integer(2))
        2
        >>> unwrapvalue(I)
        1j
        >>> unwrapvalue(Symbol('x'))
        Traceback (most recent call last):
            ...
        mimiqcircuits.symbolics.UndefinedValue: Cannot convert symbolic expression to numeric value: x
    """
    # Plain Python numeric types pass through
    if isinstance(g, (int, float)):
        return g
    if isinstance(g, complex):
        return g

    # symengine numeric types
    if isinstance(g, se.Basic):
        # Check for free symbols — if any, it's truly symbolic
        if hasattr(g, "free_symbols") and len(g.free_symbols) > 0:
            raise UndefinedValue(g)

        # Try to evaluate to a number
        try:
            v = complex(g)
            if v.imag == 0:
                fv = v.real
                # Return int if it's an integer value
                if fv == int(fv):
                    return int(fv)
                return fv
            return v
        except (TypeError, ValueError):
            pass

        raise UndefinedValue(g)

    # Fallback: try direct conversion
    try:
        return float(g)
    except (TypeError, ValueError):
        pass

    raise UndefinedValue(g)


def listsymbols(g):
    """Return the free symbols of an expression or an operation.

    Args:
        g: A number, a SymEngine expression or an :class:`Operation`.

    Returns:
        list: For an expression, its free symbols sorted by name. For an
        operation, the result of its ``listvars()``. For a number or any
        other object, an empty list.

    Examples:
        >>> from symengine import symbols
        >>> from mimiqcircuits import *
        >>> x, y = symbols('x y')
        >>> listsymbols(y + x)
        [x, y]
        >>> listsymbols(GateRX(x))
        [x]
        >>> listsymbols(1.5)
        []
    """
    if isinstance(g, (int, float, complex)):
        return []

    if isinstance(g, se.Basic):
        if hasattr(g, "free_symbols"):
            return sorted(g.free_symbols, key=str)
        return []

    # For operations, delegate to listvars
    if isinstance(g, Operation):
        return g.listvars()

    return []


def _validate_rule_gate_params(gate):
    params = gate.getparams()

    for i, p in enumerate(params, start=1):
        # numeric constants
        if isinstance(p, (int, float, complex)):
            continue

        # symengine numeric constants
        if isinstance(p, Number):
            continue

        # convert to expr
        try:
            expr = sympify(p)
        except Exception:
            raise ValueError(f"Gate parameter {i} ({p}) invalid")

        # simple variable
        if isinstance(expr, Symbol):
            continue

        # numeric symbolic expression (e.g. pi/2): allow when there are no symbols
        if hasattr(expr, "free_symbols") and len(expr.free_symbols) == 0:
            continue

        # reject expressions
        raise ValueError(
            f"Gate parameter {i} ({p}) must be numeric or a single symbolic variable"
        )


def _extract_variables(gate):
    """Return the gate parameters, with ``None`` in place of non-symbol ones."""
    params = gate.getparams()

    # Check if any parameter is symbolic
    any_symbolic = False
    for p in params:
        if isinstance(p, Symbol):
            any_symbolic = True
            break

    # If no symbolic variables return tuple of None
    if not any_symbolic:
        return tuple(None for _ in params)

    # Otherwise return tuple with symbolic vars and None for constants
    variables = tuple(p if isinstance(p, Symbol) else None for p in params)

    return variables


def applyparams(source: Operation, relation):
    """Evaluate ``target`` with values taken from the parameters of ``source``.

    ``relation`` is a pair ``(variables, target)``. The ``k``-th entry of
    ``variables`` names the ``k``-th parameter of ``source``; ``None``
    skips a parameter. The named symbols are replaced in ``target`` by the
    matching parameter values. A single symbol may be given in place of a
    tuple.

    Args:
        source (Operation): An operation with numeric parameters.
        relation (tuple): ``(variables, target)``, where ``variables`` is a
            symbol or a tuple of symbols and ``None``, and ``target`` is an
            operation written with those symbols.

    Returns:
        Operation: ``target`` evaluated with the substituted values, or
        ``target`` itself when ``variables`` is empty.

    Raises:
        ValueError: If ``source`` has symbolic parameters, an entry of
            ``variables`` is not a symbol or ``None``, or ``variables`` is
            longer than the parameter list of ``source``.

    Examples:
        >>> from symengine import symbols
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.symbolics import applyparams
        >>> x = symbols('x')
        >>> applyparams(GateRX(0.3), (x, GateRY(x)))
        RY(0.3)
        >>> # skip the first parameter of the source
        >>> applyparams(GateU(0.1, 0.2, 0.3), ((None, x), GateRZ(x)))
        RZ(0.2)
    """
    if source.is_symbolic():
        raise ValueError("source must be fully evaluated, got symbolic parameters")

    # Unpack relation
    lhs, target = relation

    # Normalize single variable case
    if not isinstance(lhs, (tuple, list)):
        variables = (lhs,)
    else:
        variables = tuple(lhs)

    # If no variables, return target as-is
    if len(variables) == 0:
        return target

    # Validate variables
    for i, var in enumerate(variables):
        if var is None:
            continue

        # Must be a simple symengine Symbol
        if not (hasattr(var, "is_symbol") and var.is_symbol):
            raise ValueError(
                f"Left side element #{i + 1} must be a simple symbolic variable, got {var}"
            )

    # Extract concrete parameters
    params = list(source.getparams())

    if len(variables) > len(params):
        raise ValueError(
            f"Relation has {len(variables)} variables but source only has {len(params)} parameters"
        )

    # Build substitution dictionary (position → value)
    subs = {}
    for i, var in enumerate(variables):
        if var is None:
            continue
        subs[var] = params[i]

    # Evaluate target with substitution
    evaluated = target.evaluate(subs)

    return evaluated
