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


import symengine as se
from typing import TypeVar, Generic
from mimiqcircuits.operations.operator import AbstractOperator
from mimiqcircuits.operations.gates.gate import Gate
import sympy as sp

T = TypeVar("T", bound=Gate)  # T is a type variable bound to Gate


class RescaledGate(AbstractOperator, Generic[T]):
    r"""RescaledGate operation.

    The `RescaledGate` represents an operation where a quantum gate is rescaled by a factor `p`,
    typically between 0 and 1. This rescaling modifies the action of the gate, multiplying its
    matrix representation by `p`. The gate to be rescaled must be an instance of the `Gate` class,
    and `p` must be a valid scalar (real or symbolic) in the range [0, 1].

    Args:
        - gate (Gate): The quantum gate to be rescaled.
        - p (float or symbolic): The rescaling factor, must be between 0 and 1.

    Raises:
        - TypeError: If the provided gate is not an instance of `Gate`.
        - ValueError: If the rescaling factor `p` is not between 0 and 1.

    Methods:
        - get_operation(): Returns the underlying gate.
        - get_param(name): Retrieves the value of a parameter (such as `p`).
        - _matrix(*args): Returns the matrix representation of the rescaled gate.
        - get_scale(): Returns the scaling factor `p`.
        - rescale(p): Returns a new `RescaledGate` with a different scaling factor.
        - rescale_in_place(p): Rescales the gate in place by multiplying the current `p` with the new value.
        - evaluate(d): Substitutes values in the parameters (useful when symbolic values are used).

    Examples:

        Creating and rescaling a gate:

        >>> from mimiqcircuits import *
        >>> g = GateH()
        >>> rg = RescaledGate(g, 0.5)

        >>> rg.get_scale()
        0.5

        >>> rg.rescale(0.8)
        0.4*H
    """

    _name = "RescaledGate"

    def __init__(self, gate: T, p):
        if not isinstance(gate, Gate):
            raise TypeError(
                f"Expected gate to be an instance of Gate, got {type(gate).__name__}"
            )

        super().__init__()
        self._num_bits = gate.num_bits
        self._num_qubits = gate.num_qubits
        self._num_zvars = gate.num_zvars
        self._num_cregs = gate._num_cregs
        self._num_qregs = gate._num_qregs
        self._num_zregs = gate._num_zregs
        self._cregsizes = gate._cregsizes
        self._qregsizes = gate._qregsizes
        self._zregsizes = gate._zregsizes

        if not isinstance(p, (se.Basic, sp.Basic)) and (float(p) < 0 or float(p) > 1):
            raise ValueError("Value of p must be between 0 and 1.")
        self.gate = gate
        self.p = p

    def __repr__(self):
        return self.__str__()

    def get_operation(self):
        return self.gate

    def get_param(self, name):
        if name == "p":
            return self.p
        return self.gate.get_param(name)

    def _matrix(self, *args):
        return self.p * self.gate._matrix(*args)

    def get_scale(self):
        return self.p

    def rescale(self, p):
        return RescaledGate(self.gate, self.p * p)

    def rescale_in_place(self, p):
        self.p *= p
        return self

    def evaluate(self, d):
        return RescaledGate(self.gate.evaluate(d), evaluate(self.p, d))

    def __str__(self):
        mult = "*" if getattr(self, "compact", True) else " * "
        return f"{self.p}{mult}{self.gate}"


def evaluate(param, d):
    # Replace symbols in param according to dictionary d, this is just a placeholder
    return param.subs(d) if isinstance(param, (se.Basic, sp.Basic)) else param
