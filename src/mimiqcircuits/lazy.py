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

import mimiqcircuits as mc


class LazyArg:
    def __str__(self):
        return "?"

    def __repr__(self):
        return self.__str__()


class LazyExpr:
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
        return self._lazy_recursive_evaluate(list(reversed(args)))

    def inverse(self):
        return LazyExpr(inverse, self)

    def parallel(self, *args):
        if len(args) == 0:
            return LazyExpr(parallel, LazyArg(), self)
        elif len(args) == 1:
            num_repeats = args[0]
            return LazyExpr(parallel, num_repeats, self)
        else:
            raise TypeError("Invalid number of arguments.")

    def control(self, *args):
        if len(args) == 0:
            return LazyExpr(control, LazyArg(), self)
        elif len(args) == 1:
            num_controls = args[0]
            return LazyExpr(control, num_controls, self)
        else:
            raise TypeError("Invalid number of arguments.")

    def power(self, *args):
        if len(args) == 0:
            return LazyExpr(power, self, LazyArg())
        elif len(args) == 1:
            p = args[0]
            return LazyExpr(power, self, p)
        else:
            return TypeError("Invalid number of arguments.")

    def __str__(self):
        return f"lazy {self.obj.__name__}({', '.join(map(str, self.args))})"

    def __repr__(self):
        return str(self)


def control(*args):
    """
    Apply a control to a quantum operation or creating a lazy expression.

    This function can be used in two ways:
    1. To create a controlled version of a `Gate` or `Operation`.
    2. To create a lazy expression that will apply a control to a future argument.

    Args:
        *args: Variable length argument list.
            - If one argument is provided:
              - If it's an operation, returns a lazy expression `control(?, op)`.
              - If it's an integer (num_controls), returns a lazy expression `control(n, ?)`.
            - If two arguments are provided (num_controls, gate):
              - Returns the controlled operation `gate.control(num_controls)`.

    Returns:
        Union[Operation, LazyExpr]: The controlled operation or a lazy expression.

    Raises:
        TypeError: If the arguments are of invalid types or count.

    Examples:
        >>> from mimiqcircuits import *
        >>> op = control(2, GateX())
        >>> op
        C₂X
        >>> lazy_ctrl = control(2)
        >>> lazy_ctrl(GateX())
        C₂X
    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, mc.Operation):
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
    """
    Create a parallel execution of a quantum operation or a lazy expression.

    This function can be used to apply an operation multiple times in parallel across different qubits.

    Args:
        *args: Variable length argument list.
            - If one argument is provided:
                - If it's an operation, returns a lazy expression `parallel(?, op)`.
                - If it's an integer (num_repeats), returns a lazy expression `parallel(n, ?)`.

            - If two arguments are provided (num_repeats, gate):
                - Returns the parallel operation `gate.parallel(num_repeats)`.

    Returns:
        Union[Operation, LazyExpr]: The parallel operation or a lazy expression.

    Raises:
        TypeError: If the arguments are of invalid types or count.

    Examples:
        >>> from mimiqcircuits import *
        >>> op = parallel(3, GateH())
        >>> op
        ⨷ ³ H
    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, mc.Operation):
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
    """
    Compute the inverse of a quantum operation.

    Args:
        op (Operation): The operation to invert.

    Returns:
        Operation: The inverse of the operation.

    Examples:
        >>> from mimiqcircuits import *
        >>> op = inverse(GateS())
        >>> op
        S†
    """
    return op.inverse()


def power(*args):
    """
    Raise a quantum operation to a power or create a lazy expression.

    Args:
        *args: Variable length argument list.
            - If one argument is provided:
              - If it's an operation, returns a lazy expression `power(op, ?)`.
              - If it's a number (exponent), returns a lazy expression `power(?, exponent)`.
            - If two arguments are provided (gate, exponent):
              - Returns the powered operation `gate.power(exponent)`.

    Returns:
        Union[Operation, LazyExpr]: The powered operation or a lazy expression.

    Raises:
        TypeError: If the arguments are of invalid types or count.

    Examples:
        >>> from mimiqcircuits import *
        >>> op = power(GateX(), 0.5)
        >>> op
        SX
    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, mc.Operation):
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
