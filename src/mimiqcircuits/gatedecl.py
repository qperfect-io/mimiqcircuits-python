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
from typing import Tuple
import mimiqcircuits as mc
from symengine import Symbol
import symengine as se
import numpy as np
import sympy as sp


class GateDecl:
    """
    Simple declaration of gates using the @gatedecl decorator.

    Examples:

        First Way

        Import necessary libaries

        >>> from symengine import symbols
        >>> from mimiqcircuits import *

        >>> x, y = symbols('x y')

        Declare a gate using the @gatedecl decorator

        >>> @gatedecl("ansatz")
        ... def ansatz(x):
        ...     c = Circuit()
        ...     c.push(GateX(), 0)
        ...     c.push(GateRX(x), 1)
        ...     return c

        Create calls to the gate declariation

        >>> ansatz(x)
        ansatz(x)

        >>> ansatz(y)
        ansatz(y)

        Decompose

        >>> ansatz(x).decompose()
        2-qubit circuit with 2 instructions:
        ├── X @ q[0]
        └── RX(x) @ q[1]
        <BLANKLINE>

        >>> ansatz(2)
        ansatz(2)

        >>> ansatz(2).decompose()
        2-qubit circuit with 2 instructions:
        ├── X @ q[0]
        └── RX(2) @ q[1]
        <BLANKLINE>

        Second Way

        >>> from symengine import *
        >>> from mimiqcircuits import *

        Define symbols for the gate arguments

        >>> x, y = symbols('x y')

        From a circuit, create a GateDecl object directly

        >>> c = Circuit()
        >>> c.push(GateXXplusYY(x,y), 0,1)
        2-qubit circuit with 1 instructions:
        └── XXplusYY(x, y) @ q[0,1]
        <BLANKLINE>
        >>> c.push(GateRX(x), 1)
        2-qubit circuit with 2 instructions:
        ├── XXplusYY(x, y) @ q[0,1]
        └── RX(x) @ q[1]
        <BLANKLINE>
        >>> gate_decl = GateDecl("ansatz", (x,y), c)
        >>> gate_decl
        gate ansatz(x, y) =
        ├── XXplusYY(x, y) @ q[0,1]
        └── RX(x) @ q[1]
        <BLANKLINE>
        >>> GateCall(gate_decl, (2,4))
        ansatz(2, 4)

        Decompose the GateCall into a quantum circuit

        >>> GateCall(gate_decl, (2,4)).decompose()
        2-qubit circuit with 2 instructions:
        ├── XXplusYY(2, 4) @ q[0,1]
        └── RX(2) @ q[1]
        <BLANKLINE>

        Add to Circuit

        >>> g = GateCall(gate_decl, (2,4))
        >>> c = Circuit()
        >>> c.push(g,10,22)
        23-qubit circuit with 1 instructions:
        └── ansatz(2, 4) @ q[10,22]
        <BLANKLINE>
    """

    _name = "GateDecl"

    def __init__(self, name: str, arguments: Tuple[Symbol, ...], circuit: mc.Circuit):
        if not all(isinstance(arg, Symbol) for arg in arguments):
            raise TypeError("All GateDecl arguments must be symbols.")

        if len(circuit) == 0:
            raise ValueError("GateDecl cannot be defined from empty circuits.")

        for inst in circuit:
            # check if the instruction operation is from the Gate operation
            if not isinstance(inst.get_operation(), mc.Gate):
                raise ValueError("GateDecl instructions must be gates.")
            if inst.num_bits() != 0 or inst.num_zvars() != 0:
                raise ValueError("GateDecl instructions cannot have bits or zvars.")
            if inst.num_qubits() == 0:
                raise ValueError("GateDecl instructions must have qubits.")

        nq = circuit.num_qubits()
        np = len(arguments)
        super().__init__()

        self.name = name
        self.arguments = arguments
        self.circuit = circuit

    @property
    def num_qubits(self):
        return self.circuit.num_qubits()

    @num_qubits.setter
    def num_qubits(self, _):
        raise ValueError("Cannot set num_qubits. Read only parameter.")

    @property
    def num_bits(self):
        return self.circuit.num_bits()

    @num_bits.setter
    def num_bits(self, _):
        raise ValueError("Cannot set num_bits. Read only parameter.")

    @property
    def num_zvars(self):
        return self.circuit.num_zvars()

    @num_zvars.setter
    def num_zvars(self, _):
        raise ValueError("Cannot set num_zvars. Read only parameter.")

    def __str__(self):
        result = f"gate {self.name}({', '.join(map(str, self.arguments))}) =\n"
        N = len(self.circuit)

        for i, inst in enumerate(self.circuit):
            if i == N - 1:
                result += f"└── {inst}\n"
            else:
                result += f"├── {inst}\n"

        return result

    def __repr__(self):
        return str(self)

    def __call__(self, *args):
        return mc.GateCall(self, args)


class GateCall(mc.Gate):
    """GateCall"""

    _name = "GateCall"
    _num_bits = 0
    _num_qubits = None

    def __init__(self, decl: GateDecl, args: Tuple[float, ...]):
        if len(args) != len(decl.arguments):
            raise ValueError("Wrong number of arguments for GateCall.")

        self._num_qubits = decl.num_qubits
        self._decl = decl
        self._args = args
        self._qregsizes = [self._num_qubits]

    def _matrix(self):
        pass

    def decompose(self):
        circ = mc.Circuit()
        d = dict(zip(self._decl.arguments, self._args))
        for inst in self._decl.circuit:
            op = inst.operation.evaluate(d)
            qubits = [i for i in inst.get_qubits()]
            bits = [i for i in inst.get_bits()]
            zvars = [i for i in inst.get_zvars()]
            circ.push(op, *qubits, *bits, *zvars)
        return circ

    def _decompose(self, circ, qubits, bits, zvars):
        d = dict(zip(self._decl.arguments, self._args))

        if len(qubits) != self._decl.num_qubits:
            raise ValueError("Wrong number of qubits for GateCall.")

        if len(bits) != self._decl.num_bits:
            raise ValueError("Wrong number of bits for GateCall.")

        if len(zvars) != self._decl.num_zvars:
            raise ValueError("Wrong number of zvars for GateCall.")

        for inst in self._decl.circuit:
            op = inst.operation.evaluate(d)

            # Map the instruction qubits to the actual qubits in the circuit
            inst_qubits = [qubits[q] for q in inst.get_qubits()]

            # Push the operation with the mapped qubits into the circuit
            circ.push(op, *inst_qubits, *bits, *zvars)

        return circ

    def __str__(self):
        return f"{self._decl.name}"

    def evaluate(self, d):
        new_args = [
            arg.subs(d) if hasattr(arg, "subs") else arg
            for arg in self._args
        ]
        return type(self)(self._decl, tuple(new_args))

    def matrix(self):
        from functools import reduce

        N = self._num_qubits
        dim = 2**N
        identity = se.Matrix(np.eye(dim, dtype=complex).tolist())

        iter = map(lambda inst: inst.matrix(N), self.decompose())
        return reduce(lambda a, b: a * b, reversed(list(iter)), identity)


def gatedecl(name):
    def decorator(func):
        def get_arg_names(func):
            import inspect

            sig = inspect.signature(func)
            return tuple(sig.parameters.keys())

        arguments = [Symbol(arg) for arg in get_arg_names(func)]

        circ = func(*arguments)

        if not isinstance(circ, mc.Circuit):
            raise ValueError("GateDecl must return a Circuit object.")

        return GateDecl(name, tuple(arguments), circ)

    return decorator


__all__ = ["GateCall", "GateDecl", "gatedecl"]
