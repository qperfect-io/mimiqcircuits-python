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
import mimiqcircuits as mc


class UndefinedValue(Exception):
    """Raised when a symbolic expression cannot be converted to a numeric value."""

    def __init__(self, expr):
        self.expr = expr
        super().__init__(f"Cannot convert symbolic expression to numeric value: {expr}")


def unwrapvalue(g):
    """Convert a symengine expression to a numeric value.

    Attempts to extract a concrete numeric value (int, float, or complex)
    from a symengine expression. If the expression contains free symbolic
    variables, raises UndefinedValue.

    This is the Python equivalent of Julia's ``unwrapvalue``.

    Args:
        g: A numeric value or symengine expression.

    Returns:
        A Python numeric value (int, float, or complex).

    Raises:
        UndefinedValue: If the expression contains unevaluated symbolic variables.
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
    """Extract all symbolic variables from an expression.

    This is the Python equivalent of Julia's ``listsymbols``.

    Args:
        g: A symengine expression, operation, or iterable.

    Returns:
        list: A sorted list of symengine Symbol objects.
    """
    if isinstance(g, (int, float, complex)):
        return []

    if isinstance(g, se.Basic):
        if hasattr(g, "free_symbols"):
            return sorted(g.free_symbols, key=str)
        return []

    # For operations, delegate to listvars
    if isinstance(g, mc.Operation):
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
    """
    Return tuple of symbolic variables (symengine.Symbol) or None for concrete.
    Concrete parameters become None.
    """
    params = gate.getparams()

    # Check if anu parameter is symbolic
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


def applyparams(source: mc.Operation, relation):
    """
    applyparams
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
