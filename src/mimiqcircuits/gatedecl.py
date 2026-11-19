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
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.operation import _check_num_ancillas
from symengine import Symbol
import symengine as se
import numpy as np


class GateDecl:
    """Declaration of a named, parametric gate defined by a circuit.

    A ``GateDecl`` stores a name, a tuple of symbolic arguments and a body
    circuit of gates written with those symbols. Calling the declaration with
    values, ``decl(v1, v2, ...)``, returns a :class:`GateCall` that can be
    pushed to circuits like any other gate. The body is copied, so later
    changes to the original circuit do not affect the declaration.

    The :func:`gatedecl` decorator is the usual way to build one.

    Args:
        name (str): Name of the gate, used when printing calls.
        arguments (tuple of symengine.Symbol): The symbolic parameters.
        circuit (Circuit): The body. Its qubits ``0..n-1`` become the
            qubits of the declared gate.

    Raises:
        TypeError: If an argument is not a ``symengine.Symbol``.
        ValueError: If the circuit is empty or contains an operation that is
            not a :class:`Gate`.

    ``GateDecl(name, arguments, circuit, num_ancillas=0)`` declares a gate
    whose body is ``circuit``. ``num_ancillas`` says how many of its last
    qubits are clean ancillas: each call must find them in ``|0>``, and gets
    them back in ``|0>``. On other inputs the call is still a unitary on all
    its qubits, but not the gate it was built for. :func:`decompose` sets it
    when it wraps a decomposition that uses clean ancillas.

    Examples:
        >>> from symengine import symbols
        >>> from mimiqcircuits import *
        >>> x, y = symbols('x y')
        >>> # write the body with symbolic parameters
        >>> c = Circuit()
        >>> c.push(GateXXplusYY(x, y), 0, 1)
        2-qubit circuit with 1 instruction:
        └── XXplusYY(x, y) @ q[0:1]
        <BLANKLINE>
        >>> c.push(GateRX(x), 1)
        2-qubit circuit with 2 instructions:
        ├── XXplusYY(x, y) @ q[0:1]
        └── RX(x) @ q[1]
        <BLANKLINE>
        >>> # declare the gate from the body
        >>> decl = GateDecl("ansatz", (x, y), c)
        >>> decl
        gate ansatz(x, y) =
        ├── XXplusYY(x, y) @ q[0:1]
        └── RX(x) @ q[1]
        <BLANKLINE>
        >>> # calling the declaration gives a gate with concrete parameters
        >>> g = decl(2, 4)
        >>> g
        ansatz(2, 4)
        >>> g.decompose()
        2-qubit circuit with 2 instructions:
        ├── XXplusYY(2, 4) @ q[0:1]
        └── RX(2) @ q[1]
        <BLANKLINE>
        >>> # push the call on any two qubits
        >>> Circuit().push(g, 10, 22)
        23-qubit circuit with 1 instruction:
        └── ansatz(2, 4) @ q[10,22]
        <BLANKLINE>

    See Also:
        :func:`gatedecl`, :class:`GateCall`, :class:`BlockDecl`
    """

    _name = "GateDecl"

    def __init__(
        self,
        name: str,
        arguments: Tuple[Symbol, ...],
        circuit: Circuit,
        num_ancillas: int = 0,
    ):
        if not all(isinstance(arg, Symbol) for arg in arguments):
            raise TypeError("All GateDecl arguments must be symbols.")

        if len(circuit) == 0:
            raise ValueError("GateDecl cannot be defined from empty circuits.")

        for inst in circuit:
            # check if the instruction operation is from the Gate operation
            if not isinstance(inst.get_operation(), Gate):
                raise ValueError("GateDecl instructions must be gates.")
            if inst.num_bits() != 0 or inst.num_zvars() != 0:
                raise ValueError("GateDecl instructions cannot have bits or zvars.")
            if inst.num_qubits() == 0:
                raise ValueError("GateDecl instructions must have qubits.")

        super().__init__()

        self.name = name
        self.arguments = arguments
        new_circuit = Circuit()
        for inst in circuit:
            new_circuit.push(
                inst.get_operation(),
                *inst.get_qubits(),
                *inst.get_bits(),
                *inst.get_zvars(),
            )

        self.circuit = new_circuit
        _check_num_ancillas(num_ancillas, self.num_qubits)
        self._num_ancillas = num_ancillas

    @property
    def num_qubits(self):
        """Number of qubits of the body circuit. Read-only."""
        return self.circuit.num_qubits()

    @property
    def num_ancillas(self):
        """Number of clean ancillas, the last targets of each call."""
        return self._num_ancillas

    @num_qubits.setter
    def num_qubits(self, _):
        raise ValueError("Cannot set num_qubits. Read only parameter.")

    @property
    def num_bits(self):
        """Number of classical bits of the body circuit, always 0. Read-only."""
        return self.circuit.num_bits()

    @num_bits.setter
    def num_bits(self, _):
        raise ValueError("Cannot set num_bits. Read only parameter.")

    @property
    def num_zvars(self):
        """Number of z-variables of the body circuit, always 0. Read-only."""
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
        return GateCall(self, args)


class GateCall(Gate):
    """Application of a :class:`GateDecl` to concrete parameter values.

    Usually built by calling the declaration, ``decl(*args)``. The gate acts
    on as many qubits as the declaration's body. Its decomposition is the
    body with the declaration's symbols replaced by ``args``. Inverse, power
    and control wrap the call in :class:`Inverse`, :class:`Power` and
    :class:`Control`.

    Args:
        decl (GateDecl): The declaration to call.
        args (tuple): One value per declaration argument. Values may be
            numbers or symbolic expressions.

    Raises:
        ValueError: If ``len(args)`` differs from the number of declaration
            arguments.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import symbols
        >>> x = symbols('x')
        >>> decl = GateDecl("rot", (x,), Circuit().push(GateH(), 0).push(GateRZ(x), 0))
        >>> g = GateCall(decl, (0.5,))
        >>> g
        rot(0.5)
        >>> g.decompose()
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── RZ(0.5) @ q[0]
        <BLANKLINE>
        >>> g.inverse().decompose()
        1-qubit circuit with 2 instructions:
        ├── RZ(-0.5) @ q[0]
        └── H @ q[0]
        <BLANKLINE>
    """

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

    @property
    def decl(self):
        """The :class:`GateDecl` being called."""
        return self._decl

    @property
    def num_ancillas(self):
        return self._decl.num_ancillas

    @property
    def arguments(self):
        """The values passed for the declaration arguments, as a tuple."""
        return self._args

    def _matrix(self):
        pass

    def decompose(self):
        """Return the body circuit with the declaration symbols replaced by the call values."""
        circ = Circuit()
        d = dict(zip(self._decl.arguments, self._args))
        for inst in self._decl.circuit:
            op = inst.operation.evaluate(d)
            qubits = inst.get_qubits()
            circ.push(op, *qubits)
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
        arg_str = ", ".join(map(str, self._args))
        display_args = f"({arg_str})"
        if len(display_args) > 20:
            display_args = "(...)"
        return f"{self._decl.name}{display_args}"

    # The body only holds the declaration's arguments; the call's values are in _args.
    def is_symbolic(self):
        """Return ``True`` if any call value contains a free symbol."""
        return any(isinstance(a, se.Basic) and a.free_symbols for a in self._args)

    def listvars(self):
        """Return the symbols appearing in the call values."""
        return list(
            {v for a in self._args if isinstance(a, se.Basic) for v in a.atoms(se.Symbol)}
        )

    def evaluate(self, d):
        """Substitute symbols in the call values and return a new ``GateCall``.

        Args:
            d (dict): Map from symbols to the values to substitute.
        """
        new_args = [arg.subs(d) if hasattr(arg, "subs") else arg for arg in self._args]
        return type(self)(self._decl, tuple(new_args))

    def matrix(self):
        """Return the unitary of the decomposed body as a SymEngine matrix."""
        from functools import reduce

        N = self._num_qubits
        dim = 2**N
        identity = se.Matrix(np.eye(dim, dtype=complex).tolist())

        iter = map(lambda inst: inst.matrix(N), self.decompose())
        return reduce(lambda a, b: a * b, reversed(list(iter)), identity)


def gatedecl(name):
    """Decorator that turns a function returning a circuit into a :class:`GateDecl`.

    The names of the function's arguments become the symbolic parameters of
    the declaration. The function is called once, with those symbols, to
    build the body. The decorated name is bound to the ``GateDecl``, not to
    the function; calling it with values gives a :class:`GateCall`.

    Args:
        name: Name of the declared gate.

    Returns:
        A decorator that returns the ``GateDecl``.

    Raises:
        ValueError: If the decorated function does not return a
            :class:`Circuit`.

    Examples:
        >>> from mimiqcircuits import *
        >>> # the argument name ``theta`` becomes the symbol of the declaration
        >>> @gatedecl("ansatz")
        ... def ansatz(theta):
        ...     return Circuit().push(GateX(), 0).push(GateRX(theta), 1)
        >>> ansatz
        gate ansatz(theta) =
        ├── X @ q[0]
        └── RX(theta) @ q[1]
        <BLANKLINE>
        >>> # call it with a value to get a gate
        >>> ansatz(0.5)
        ansatz(0.5)
        >>> ansatz(0.5).decompose()
        2-qubit circuit with 2 instructions:
        ├── X @ q[0]
        └── RX(0.5) @ q[1]
        <BLANKLINE>
        >>> # the call is pushed like any gate
        >>> Circuit().push(ansatz(0.5), 3, 4)
        5-qubit circuit with 1 instruction:
        └── ansatz(0.5) @ q[3:4]
        <BLANKLINE>
    """
    def decorator(func):
        def get_arg_names(func):
            import inspect

            sig = inspect.signature(func)
            return tuple(sig.parameters.keys())

        arguments = [Symbol(arg) for arg in get_arg_names(func)]

        circ = func(*arguments)

        if not isinstance(circ, Circuit):
            raise ValueError("GateDecl must return a Circuit object.")

        return GateDecl(name, tuple(arguments), circ)

    return decorator


__all__ = ["GateCall", "GateDecl", "gatedecl"]
