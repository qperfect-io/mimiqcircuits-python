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
"""Lazy evaluation and composition functions."""

from mimiqcircuits.operations.operation import Operation


class LazyArg:
    """Placeholder for an argument of a ``LazyExpr`` that is not known yet. It prints as ``?``."""

    def __str__(self):
        return "?"

    def __repr__(self):
        return self.__str__()


class LazyExpr:
    """Operation whose construction waits for some of its arguments.

    Calling ``control``, ``parallel``, ``power`` or ``repeat`` on a gate
    without the count, or a size-dependent gate such as ``QFT()`` without
    its size, returns a ``LazyExpr``. A ``LazyExpr`` can be wrapped further
    with :meth:`control`, :meth:`parallel`, :meth:`power` and
    :meth:`inverse`. Calling it fills the ``LazyArg`` placeholders, in the
    order they appear in the printed expression (left to right), and builds
    the operation. Passing more arguments than placeholders raises
    ``TypeError``.

    Args:
        obj: Callable that builds the operation.
        *args: Its arguments; ``LazyArg()`` marks the ones still missing.

    Examples:
        >>> from mimiqcircuits import *
        >>> # leave the number of controls open
        >>> lazy = control(GateX())
        >>> lazy
        lazy control(?, X)
        >>> lazy(2)
        C₂X
        >>> # nest lazy operations: the placeholders are filled left to right
        >>> lazy = GateX().power().control()
        >>> lazy
        lazy control(?, lazy power(X, ?))
        >>> lazy(2, 0.5)
        C₂SX
    """

    def __init__(self, obj, *args):
        self.obj = obj
        self.args = list(args)

    def _lazy_recursive_evaluate(self, args):
        actual = []
        for arg in self.args:
            if isinstance(arg, LazyArg):
                new_arg = args.pop()
            elif isinstance(arg, LazyExpr):
                new_arg = arg._lazy_recursive_evaluate(args)
            else:
                new_arg = arg
            actual.append(new_arg)
        return self.obj(*actual)

    def __call__(self, *args):
        remaining = list(reversed(args))
        result = self._lazy_recursive_evaluate(remaining)
        if remaining:
            raise TypeError(
                f"Lazy expression got {len(args)} arguments, "
                f"expected {len(args) - len(remaining)}."
            )
        return result

    def inverse(self):
        """Return a lazy expression for the inverse of this expression."""
        return LazyExpr(inverse, self)

    def parallel(self, *args):
        """Return a lazy expression for the parallel repetition of this expression.

        Args:
            *args: Optional number of repetitions. Without it, the number
                becomes a new placeholder.

        Raises:
            TypeError: If more than one argument is given.
        """
        if len(args) == 0:
            return LazyExpr(parallel, LazyArg(), self)
        elif len(args) == 1:
            num_repeats = args[0]
            return LazyExpr(parallel, num_repeats, self)
        else:
            raise TypeError("Invalid number of arguments.")

    def control(self, *args):
        """Return a lazy expression for the controlled version of this expression.

        Args:
            *args: Optional number of controls. Without it, the number
                becomes a new placeholder.

        Raises:
            TypeError: If more than one argument is given.
        """
        if len(args) == 0:
            return LazyExpr(control, LazyArg(), self)
        elif len(args) == 1:
            num_controls = args[0]
            return LazyExpr(control, num_controls, self)
        else:
            raise TypeError("Invalid number of arguments.")

    def power(self, *args):
        """Return a lazy expression for a power of this expression.

        Args:
            *args: Optional exponent. Without it, the exponent becomes a new
                placeholder.
        """
        if len(args) == 0:
            return LazyExpr(power, self, LazyArg())
        elif len(args) == 1:
            p = args[0]
            return LazyExpr(power, self, p)
        else:
            raise TypeError("Invalid number of arguments.")

    def __str__(self):
        return f"lazy {self.obj.__name__}({', '.join(map(str, self.args))})"

    def __repr__(self):
        return str(self)


def control(*args):
    """Return a controlled operation, or a lazy expression for one.

    Args:
        *args: One of the following.

            - ``(num_controls, op)``: return ``op.control(num_controls)``.
            - ``(op,)``: return the lazy expression ``control(?, op)``.
            - ``(num_controls,)`` with an ``int``: return the lazy expression
              ``control(num_controls, ?)``.

    Returns:
        Operation or LazyExpr: The controlled operation or a lazy expression.

    Raises:
        TypeError: If a single argument is neither an :class:`Operation` nor
            an ``int``, or the number of arguments is not 1 or 2.

    Examples:
        >>> from mimiqcircuits import *
        >>> control(2, GateX())
        C₂X
        >>> # fix the number of controls, give the operation later
        >>> lazy_ctrl = control(2)
        >>> lazy_ctrl(GateX())
        C₂X
    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, Operation):
            return LazyExpr(control, LazyArg(), arg)
        elif isinstance(arg, int):
            return LazyExpr(control, arg, LazyArg())
        else:
            raise TypeError("Invalid argument type.")
    elif len(args) == 2:
        numcontrols, gate = args
        return gate.control(numcontrols)
    else:
        raise TypeError("Invalid number of arguments.")


def parallel(*args):
    """Return an operation repeated in parallel on separate qubits, or a lazy expression for one.

    Args:
        *args: One of the following.

            - ``(num_repeats, op)``: return ``op.parallel(num_repeats)``.
            - ``(op,)``: return the lazy expression ``parallel(?, op)``.
            - ``(num_repeats,)`` with an ``int``: return the lazy expression
              ``parallel(num_repeats, ?)``.

    Returns:
        Operation or LazyExpr: The parallel operation or a lazy expression.

    Raises:
        TypeError: If a single argument is neither an :class:`Operation` nor
            an ``int``, or the number of arguments is not 1 or 2.

    Examples:
        >>> from mimiqcircuits import *
        >>> parallel(3, GateH())
        ⨷ ³ H
        >>> # fix the number of repetitions, give the operation later
        >>> parallel(3)(GateH())
        ⨷ ³ H
    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, Operation):
            return LazyExpr(parallel, LazyArg(), arg)
        elif isinstance(arg, int):
            return LazyExpr(parallel, arg, LazyArg())
        else:
            raise TypeError("Invalid argument type.")
    elif len(args) == 2:
        num_repeats, gate = args
        return gate.parallel(num_repeats)
    else:
        raise TypeError("Invalid number of arguments.")


def inverse(op):
    """Return the inverse of an operation; same as ``op.inverse()``.

    Args:
        op (Operation or LazyExpr): The operation to invert.

    Returns:
        Operation or LazyExpr: The inverse.

    Examples:
        >>> from mimiqcircuits import *
        >>> op = inverse(GateS())
        >>> op
        S†
    """
    return op.inverse()


def power(*args):
    """Return a power of an operation, or a lazy expression for one.

    Args:
        *args: One of the following.

            - ``(op, exponent)``: return ``op.power(exponent)``.
            - ``(op,)``: return the lazy expression ``power(op, ?)``.
            - ``(exponent,)`` with an ``int``: return the lazy expression
              ``power(?, exponent)``. A non-integer exponent cannot be given
              alone.

    Returns:
        Operation or LazyExpr: The powered operation or a lazy expression.

    Raises:
        TypeError: If a single argument is neither an :class:`Operation` nor
            an ``int``, or the number of arguments is not 1 or 2.

    Examples:
        >>> from mimiqcircuits import *
        >>> power(GateX(), 0.5)
        SX
        >>> # fix the exponent, give the operation later
        >>> power(2)(GateS())
        Z
    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, Operation):
            return LazyExpr(power, arg, LazyArg())
        elif isinstance(arg, int):
            return LazyExpr(power, LazyArg(), arg)
        else:
            raise TypeError("Invalid argument type.")
    elif len(args) == 2:
        gate, p = args
        return gate.power(p)
    else:
        raise TypeError("Invalid number of arguments.")


__all__ = ["LazyArg", "LazyExpr", "control", "parallel", "inverse", "power"]
