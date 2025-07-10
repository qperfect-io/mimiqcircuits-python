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

from mimiqcircuits import Operation
import warnings


# raises a z-variable to a given power (passwd as a parameter of the operation)
class Pow(Operation):
    """
    Raises a z-variable to a given power (passed as a parameter of the operation).

    Examples:
        >>> from mimiqcircuits import *
        >>> Pow(-1)
        z[?] = z[?]^(-1)
        >>> c = Circuit()
        >>> c.push(Pow(2), 1)
        2-zvar circuit with 1 instructions:
        └── z[1] = z[1]^2
        <BLANKLINE>
    """

    _name = "Pow"
    _num_bits = 0
    _num_qubits = 0
    _num_cregs = 0
    _num_zvars = 1
    _num_zregs = 1
    _qregsizes = []
    _cregsizes = []
    _zregsizes = [1]
    _parnames = ("exponent",)

    def __init__(self, exponent):
        if exponent == 1:
            warnings.warn("Pow(1) will be equivalent to a no-op.", stacklevel=2)

        super().__init__()
        self.exponent = exponent

    def inverse(self):
        return Pow(-self.exponent)

    def iswrapper(self):
        return False

    def __repr__(self):
        if self.exponent >= 0:
            return f"z[?] = z[?]^{self.exponent}"
        return f"z[?] = z[?]^({self.exponent})"

    def __str__(self):
        return f"{self._name}({self.exponent})"

    def format_with_targets(self, qubits, bits, zvars):
        idx = zvars[0]
        if self.exponent >= 0:
            return f"z[{idx}] = z[{idx}]^{self.exponent}"
        else:
            return f"z[{idx}] = z[{idx}]^({self.exponent})"


class Add(Operation):
    """
    Add several z-register variables between them and optionally a constant.
    The result is strored in the first z-register variable given.

    Examples:
        >>> from mimiqcircuits import *
        >>> Add(1)
        z[?0] += 0.0
        >>> Add(1,c=1)
        z[?0] += 1
        >>> Add(3,1)
        z[?0] += 1 + z[?1] + z[?2]
        >>> Add(4, c=3)
        z[?0] += 3 + z[?1] + z[?2] + z[?3]
        >>> c = Circuit()
        >>> c.push(Add(2), 1, 2)
        3-zvar circuit with 1 instructions:
        └── z[1] += 0.0 + z[2]
        <BLANKLINE>
    """

    _name = "Add"
    _num_bits = 0
    _num_qubits = 0
    _num_qregs = 0
    _num_cregs = 0
    _qregsizes = []
    _cregsizes = []
    _parnames = ("term",)

    def __init__(self, N=1, c=0.0):
        if N < 1:
            raise ValueError("Add requires at least one z-variable.")
        if N == 1 and c == 0.0:
            warnings.warn("Add(1; c=0.0) will be equivalent to a no-op.", stacklevel=2)

        super().__init__()
        self._num_zvars = N
        self._num_zregs = 1
        self._zregsizes = [N]
        self.term = c

    def iswrapper(self):
        return False

    def __repr__(self):
        repr = "z[?0] +="
        parts = [str(self.term)]
        for i in range(1, self._num_zvars):
            parts.append(f"z[?{i}]")
        repr += " " + " + ".join(parts)
        return repr

    def __str__(self):
        return f"{self._name}({self._num_zvars}, c={self.term})"

    def format_with_targets(self, qubits, bits, zvars):
        head = f"z[{zvars[0]}] +="
        terms = [str(self.term)]
        terms.extend(f"z[{z}]" for z in zvars[1:])
        return f"{head} {' + '.join(terms)}"


class Multiply(Operation):
    """
    Multiply several z-register variables between them and optionally a constant.
    The result is strored in the first z-register variable given.

    Examples:
        >>> from mimiqcircuits import *
        >>> Multiply(1)
        z[?0] *= 1.0
        >>> Multiply(3)
        z[?0] *= 1.0 * z[?1] * z[?2]
        >>> Multiply(3, c=2)
        z[?0] *= 2 * z[?1] * z[?2]
        >>> Multiply(2,2)
        z[?0] *= 2 * z[?1]
        >>> c = Circuit()
        >>> c.push(Multiply(2), 1, 2)
        3-zvar circuit with 1 instructions:
        └── z[1] *= 1.0 * z[2]
        <BLANKLINE>
    """

    _name = "Multiply"
    _num_bits = 0
    _num_qubits = 0
    _num_qregs = 0
    _num_cregs = 0
    _qregsizes = []
    _cregsizes = []
    _parnames = ("factor",)

    def __init__(self, N=1, c=1.0):
        if N < 1:
            raise ValueError("Multiply requires at least one z-variable.")
        if N == 1 and c == 1.0:
            warnings.warn(
                "Multiply(1; c=1.0) will be equivalent to a no-op.", stacklevel=2
            )

    def __init__(self, N=1, c=1.0):
        if N < 1:
            raise ValueError("Multiply requires at least one z-variable.")
        if N == 1 and c == 1.0:
            warnings.warn(
                "Multiply(1; c=1.0) will be equivalent to a no-op.", stacklevel=2
            )

        super().__init__()
        self._num_zvars = N
        self._num_zregs = 1
        self._zregsizes = [N]
        self.factor = c

    def iswrapper(self):
        return False

    def __repr__(self):
        repr = "z[?0] *="
        parts = [str(self.factor)]
        for i in range(1, self._num_zvars):
            parts.append(f"z[?{i}]")
        repr += " " + " * ".join(parts)
        return repr

    def __str__(self):
        return f"{self._name}({self.num_zvars}, c={self.factor})"

    def format_with_targets(self, qubits, bits, zvars):
        head = f"z[{zvars[0]}] *="
        terms = [str(self.factor)]
        terms.extend(f"z[{z}]" for z in zvars[1:])
        return f"{head} {' * '.join(terms)}"


__all__ = ["Pow", "Add", "Multiply"]
