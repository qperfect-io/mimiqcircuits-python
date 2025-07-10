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
    return op.inverse()


def power(*args):
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
