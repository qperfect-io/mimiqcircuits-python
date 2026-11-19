#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
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
"""Named, parametric blocks: ``BlockDecl`` and ``BlockCall``."""

import inspect

import symengine as se

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.operations.block import Block
from mimiqcircuits.operations.operation import Operation, _check_num_ancillas


class BlockDecl:
    """Declaration of a named block with symbolic arguments.

    A ``BlockDecl`` is to :class:`Block` what :class:`GateDecl` is to a gate:
    a reusable body that can hold any instruction (measurements, resets, noise,
    conditionals, z-variable operations) and takes arguments. A declaration
    cannot be pushed to a circuit; calling it with values gives a
    :class:`BlockCall`, which can.

    Every symbol used in the body must be one of the arguments, so evaluating a
    call only changes the call's values and all calls keep sharing the same
    declaration.


    Args:
        name (str): Name of the block.
        arguments (tuple): The symbolic arguments, as ``symengine.Symbol``.
        body: A :class:`Circuit`, a :class:`Block` or a list of
            :class:`Instruction`. A ``Block`` keeps its declared sizes; for the
            others the sizes are those used by the instructions.
        num_ancillas (int): How many of the last qubits of each call are
            clean ancillas, found and returned in ``|0>``. Defaults to that of
            ``body`` when it is a ``Block``, and to 0 otherwise.

    Raises:
        TypeError: If an argument is not a symbol.
        ValueError: If the arguments repeat, the body uses a symbol that is
            not an argument, or ``num_ancillas`` is negative or larger than
            the number of qubits.

    Examples:
        Declare a block that rotates a qubit, entangles it with a second one
        and measures the second one.

        >>> from symengine import symbols
        >>> from mimiqcircuits import *
        >>> theta = symbols("theta")
        >>> body = Circuit()
        >>> _ = body.push(GateRX(theta), 0)
        >>> _ = body.push(GateCX(), 0, 1)
        >>> _ = body.push(Measure(), 1, 0)
        >>> syndrome = BlockDecl("syndrome", (theta,), body)
        >>> syndrome
        block syndrome(theta) =
        ├── RX(theta) @ q[0]
        ├── CX @ q[0], q[1]
        └── M @ q[1], c[0]
        <BLANKLINE>

        Calling the declaration gives a ``BlockCall``, which is pushed like any
        other operation: here on qubits 0 and 1 with bit 0, then on qubits 3
        and 2 with bit 1. Both calls refer to the same declaration.

        >>> c = Circuit()
        >>> _ = c.push(syndrome(0.1), 0, 1, 0)
        >>> _ = c.push(syndrome(0.2), 3, 2, 1)
        >>> c
        4-qubit, 2-bit circuit with 2 instructions:
        ├── syndrome(0.1) @ q[0], q[1], c[0]
        └── syndrome(0.2) @ q[3], q[2], c[1]
        <BLANKLINE>

        One decomposition step replaces each call with its body, with the
        values in place of the arguments.

        >>> decompose_step(c)
        4-qubit, 2-bit circuit with 6 instructions:
        ├── RX(0.1) @ q[0]
        ├── CX @ q[0], q[1]
        ├── M @ q[1], c[0]
        ├── RX(0.2) @ q[3]
        ├── CX @ q[3], q[2]
        └── M @ q[2], c[1]
        <BLANKLINE>
    """

    def __init__(self, name, arguments, body, num_ancillas=None):
        arguments = tuple(arguments)
        if not all(isinstance(arg, se.Symbol) for arg in arguments):
            raise TypeError("BlockDecl arguments must be symbols.")
        if len(set(arguments)) != len(arguments):
            raise ValueError("BlockDecl arguments must be unique.")

        if num_ancillas is None:
            num_ancillas = body.num_ancillas if isinstance(body, Block) else 0
        if isinstance(body, Block):
            sizes = (body.num_qubits, body.num_bits, body.num_zvars)
        else:
            # Block infers the sizes and checks the instructions.
            sizes = None
        instructions = list(body)
        if sizes is None:
            inferred = Block(instructions)
            sizes = (inferred.num_qubits, inferred.num_bits, inferred.num_zvars)

        free = {v for inst in instructions for v in inst.listvars()}
        if not free <= set(arguments):
            missing = ", ".join(sorted(map(str, free - set(arguments))))
            raise ValueError(
                f"BlockDecl body uses symbols that are not arguments: {missing}"
            )

        self.name = name
        self.arguments = arguments
        # Shallow copy: a nested call must keep referring to its declaration.
        self.instructions = instructions
        self._nq, self._nc, self._nz = sizes
        _check_num_ancillas(num_ancillas, self._nq)
        self._num_ancillas = num_ancillas

    @property
    def num_qubits(self):
        """Number of qubits a call to this block acts on."""
        return self._nq

    @property
    def num_ancillas(self):
        """Number of clean ancillas, the last qubit targets of each call."""
        return self._num_ancillas

    @property
    def num_bits(self):
        """Number of classical bits a call to this block acts on."""
        return self._nc

    @property
    def num_zvars(self):
        """Number of z-variables a call to this block acts on."""
        return self._nz

    def __iter__(self):
        return iter(self.instructions)

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        return self.instructions[idx]

    def __eq__(self, other):
        if self is other:
            return True
        return (
            isinstance(other, BlockDecl)
            and self.name == other.name
            and self.arguments == other.arguments
            and (self._nq, self._nc, self._nz) == (other._nq, other._nc, other._nz)
            and self._num_ancillas == other._num_ancillas
            and self.instructions == other.instructions
        )

    def __hash__(self):
        return hash(
            (self.name, self.arguments, self._nq, self._nc, self._nz, self._num_ancillas)
        )

    def __str__(self):
        result = f"block {self.name}({', '.join(map(str, self.arguments))}) =\n"
        N = len(self.instructions)
        for i, inst in enumerate(self.instructions):
            result += f"└── {inst}\n" if i == N - 1 else f"├── {inst}\n"
        return result

    def __repr__(self):
        return str(self)

    def __call__(self, *args):
        return BlockCall(self, args)


class BlockCall(Operation):
    """A call to a :class:`BlockDecl` with values for its arguments.

    The call holds a reference to its declaration, not a copy. Pushing it to a
    circuit takes the declaration's qubits, then its bits, then its
    z-variables. See :class:`BlockDecl` for an example.

    Args:
        decl (BlockDecl): The declaration to call.
        args (tuple): One value (number or symbolic expression) per argument.

    Raises:
        ValueError: If the number of values does not match the arguments.

    Examples:
        >>> from symengine import symbols
        >>> from mimiqcircuits import *
        >>> t = symbols("t")
        >>> decl = BlockDecl("rot", (t,), Circuit().push(GateRX(t), 0))
        >>> call = BlockCall(decl, (0.5,))
        >>> call
        rot(0.5)
        >>> call.decompose()
        1-qubit circuit with 1 instruction:
        └── RX(0.5) @ q[0]
        <BLANKLINE>
    """

    _name = "BlockCall"
    _num_qregs = 1
    _num_cregs = 1
    _num_zregs = 1
    _parnames = ()

    def __init__(self, decl, args):
        args = tuple(args)
        if len(args) != len(decl.arguments):
            raise ValueError(
                f"BlockCall to {decl.name} takes {len(decl.arguments)} values, got {len(args)}."
            )
        super().__init__()
        self._decl = decl
        self._args = args
        self._num_qubits = decl.num_qubits
        self._num_bits = decl.num_bits
        self._num_zvars = decl.num_zvars
        self._qregsizes = [1] * self._num_qubits if self._num_qubits > 0 else []
        self._cregsizes = [1] * self._num_bits if self._num_bits > 0 else []
        self._zregsizes = [1] * self._num_zvars if self._num_zvars > 0 else []

    @property
    def decl(self):
        """The :class:`BlockDecl` being called."""
        return self._decl

    @property
    def num_ancillas(self):
        return self._decl.num_ancillas

    @property
    def arguments(self):
        """The values passed for the declaration arguments, as a tuple."""
        return self._args

    def isunitary(self):
        """Return ``True`` if every operation in the declaration body is unitary."""
        return all(inst.operation.isunitary() for inst in self._decl)

    def is_symbolic(self):
        """Return ``True`` if any call value contains a free symbol."""
        return any(isinstance(a, se.Basic) and a.free_symbols for a in self._args)

    def listvars(self):
        """Return the symbols appearing in the call values."""
        return list(
            {v for a in self._args if isinstance(a, se.Basic) for v in a.atoms(se.Symbol)}
        )

    def evaluate(self, d):
        """Substitute symbols in the call values and return a new ``BlockCall``.

        The declaration is shared, not copied.

        Args:
            d (dict): Map from symbols to the values to substitute.
        """
        args = [a.subs(d) if isinstance(a, se.Basic) else a for a in self._args]
        return BlockCall(self._decl, args)

    def iswrapper(self):
        """Return ``False``: a call is not a wrapper around a single operation."""
        return False

    def supports_canonical_rewrite(self):
        """Return ``True``: a call can be expanded into its body during canonical rewriting."""
        return True

    def _decompose(self, circuit, qtargets, ctargets, ztargets):
        d = dict(zip(self._decl.arguments, self._args))
        for inst in self._decl:
            circuit.push(
                inst.operation.evaluate(d),
                *[qtargets[i] for i in inst.qubits],
                *[ctargets[i] for i in inst.bits],
                *[ztargets[i] for i in inst.zvars],
            )
        return circuit

    def __str__(self):
        if not self._args:
            return self._decl.name
        return f"{self._decl.name}({', '.join(map(str, self._args))})"


def blockdecl(name):
    """Decorator that turns a function returning a circuit into a :class:`BlockDecl`.

    The names of the function's arguments become the symbolic arguments of the
    declaration. The function is called once, with those symbols, to build
    the body. The decorated name is bound to the ``BlockDecl``, not to the
    function; calling it with values gives a :class:`BlockCall`.

    Args:
        name: Name of the declared block.

    Returns:
        A decorator that returns the ``BlockDecl``.

    Raises:
        ValueError: If the decorated function does not return a ``Circuit`` or
            a ``Block``.

    Examples:
        >>> from mimiqcircuits import *
        >>> # the argument name ``phi`` becomes the symbol of the declaration
        >>> @blockdecl("prepare")
        ... def prepare(phi):
        ...     return Circuit().push(GateRY(phi), 0).push(Measure(), 0, 0)
        >>> prepare
        block prepare(phi) =
        ├── RY(phi) @ q[0]
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> # call it with a value and push it on qubit 2 and bit 1
        >>> Circuit().push(prepare(0.5), 2, 1)
        3-qubit, 2-bit circuit with 1 instruction:
        └── prepare(0.5) @ q[2], c[1]
        <BLANKLINE>
    """

    def decorator(func):
        arguments = tuple(se.Symbol(a) for a in inspect.signature(func).parameters)
        body = func(*arguments)
        if not isinstance(body, (Circuit, Block)):
            raise ValueError("A blockdecl function must return a Circuit or a Block.")
        return BlockDecl(name, arguments, body)

    return decorator


__all__ = ["BlockCall", "BlockDecl", "blockdecl"]
