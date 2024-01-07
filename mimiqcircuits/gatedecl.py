#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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
from typing import List, Tuple
import mimiqcircuits as mc


class GateDecl:
    """
Simple declaration of gates using the @gatedecl decorator.

Examples:

    **First way**

    >>> from symengine import symbols
    >>> from mimiqcircuits import *

    Define symbols for the gate arguments

    >>> x, y = symbols('x y')

    Declare a gate using the @gatedecl decorator

    >>> @gatedecl("ansatz")
    ... def ansatz(x):
    ...     insts = [
    ...         Instruction(GateX(), (1,)),
    ...         Instruction(GateRX(x), (2,))
    ...     ]
    ...     return insts

    Create a GateDecl object using the decorator

    >>> ansatz(x)
    gate ansatz(x) =
    ├── X @ q[1]
    └── RX(x) @ q[2]
    <BLANKLINE>

    >>> ansatz(y)
    gate ansatz(y) =
    ├── X @ q[1]
    └── RX(y) @ q[2]
    <BLANKLINE>

    Decompose the GateCall into a quantum circuit

    >>> GateCall(ansatz(x), (1.57,)).decompose()
    3-qubit circuit with 2 instructions:
    ├── X @ q[1]
    └── RX(1.57) @ q[2]
    <BLANKLINE>

    >>> GateCall(ansatz(y), (1.57,)).decompose()
    3-qubit circuit with 2 instructions:
    ├── X @ q[1]
    └── RX(1.57) @ q[2]
    <BLANKLINE>

    **Second Way**

    >>> from symengine import *
    >>> from mimiqcircuits import *

    Define symbols for the gate arguments

    >>> x, y = symbols('x y')

    Create a GateDecl object using the GateDecl class

    >>> gate_decl = GateDecl("ansatz", ('x','y'), [Instruction(GateXXplusYY(x,y), (1,2)), Instruction(GateRX(x),(2,))])
    >>> GateCall(gate_decl, (2,4))
    ansatz(2, 4)

    Decompose the GateCall into a quantum circuit

    >>> GateCall(gate_decl, (2,4)).decompose()
    3-qubit circuit with 2 instructions:
    ├── XXplusYY(2, 4) @ q[1,2]
    └── RX(2) @ q[2]
    <BLANKLINE>

    Add to Circuit

    >>> g = GateCall(gate_decl, (2,4))
    >>> c = Circuit()
    >>> c.push(g,10,22)
    23-qubit circuit with 1 instructions:
    └── ansatz(2, 4) @ q[10,22]
    <BLANKLINE>
"""
    _num_qubits = None
    _num_bits = None
    _name = 'GateDecl'

    def __init__(self, name: str, arguments: Tuple[str, ...], instructions: List[mc.Instruction]):
        if not all(isinstance(arg, str) for arg in arguments):
            raise TypeError("All GateDecl arguments must be strings.")

        if not instructions:
            raise ValueError("GateDecl instructions cannot be empty.")

        unique_qubits = set()
        for inst in instructions:
            unique_qubits.update(inst.get_qubits())

        nq = len(unique_qubits)
        np = len(arguments)
        super().__init__()

        self.name = name
        self.arguments = arguments
        self.instructions = instructions
        self.num_qubits = nq
        self.num_params = np

    def __str__(self):
        result = f"gate {self.name}({', '.join(map(str, self.arguments))}) =\n"

        for i, inst in enumerate(self.instructions):
            if i == len(self.instructions) - 1:
                result += f"└── {inst}\n"
            else:
                result += f"├── {inst}\n"

        return result

    def __repr__(self):
        return str(self)


class GateCall(mc.Gate):
    """GateCall
    """
    _name = 'GateCall'
    _num_bits = 0
    _num_qubits = None

    def __init__(self, decl: GateDecl, args: Tuple[float, ...]):
        if len(args) != len(decl.arguments):
            raise ValueError("Wrong number of arguments for GateCall.")

        self._num_qubits = decl.num_qubits
        self._decl = decl
        self._args = args

    def _matrix(self):
        pass

    def _decompose(self, circ, qubits, bits):
        d = dict(zip(self._decl.arguments, self._args))
        for inst in self._decl.instructions:
            op = inst.operation.evaluate(d)
            qubits = [i for i in inst.get_qubits()]
            bits = [i for i in inst.get_bits()]
            circ.push(op, *qubits, *bits)
        return circ

    def __str__(self):
        return f"{self._decl.name}({', '.join(map(str, self._args))})"

    def __repr__(self):
        return str(self)

    def evaluate(self, d):
        new_args = [arg.subs(d) if not isinstance(
            arg, (int, float)) else arg for arg in self._args]
        return type(self)(self._decl, tuple(new_args))


def gatedecl(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            instructions = func(*args, **kwargs)
            arguments = [str(arg) for arg in args] + \
                [f"{key}={value}" for key, value in kwargs.items()]
            return GateDecl(name, arguments, instructions)
        return wrapper
    return decorator


__all__ = ['GateCall', 'GateDecl', 'gatedecl']
