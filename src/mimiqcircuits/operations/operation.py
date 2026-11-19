#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use thas file except in compliance with the License.
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
"""Base Operation class."""

from abc import ABC, abstractmethod
import copy
import symengine as se
from mimiqcircuits._sympycheck import _is_basic


def _check_num_ancillas(num_ancillas, nq):
    if not 0 <= num_ancillas <= nq:
        raise ValueError(
            f"num_ancillas must be between 0 and the number of qubits ({nq}), "
            f"got {num_ancillas}"
        )


class Operation(ABC):
    """Abstract base class for everything that can be pushed to a circuit.

    An operation acts on a fixed number of qubits, classical bits and
    complex-valued z-variables (see :attr:`num_qubits`, :attr:`num_bits` and
    :attr:`num_zvars`). Gates, measurements, resets, noise channels,
    classical operations and annotations are all operations. An operation
    only says *what* is applied; an :class:`Instruction` pairs it with the
    targets it is applied to.

    Examples:
        >>> from mimiqcircuits import *
        >>> # operations know how many targets of each kind they need
        >>> op = Measure()
        >>> op.num_qubits, op.num_bits, op.num_zvars
        (1, 1, 0)
        >>> # pushing an operation to a circuit creates an instruction
        >>> c = Circuit()
        >>> c.push(op, 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── M @ q[0], c[0]
        <BLANKLINE>
    """

    _name = None

    _num_qubits = None
    _num_qregs = 1
    _qregsizes = None

    _num_bits = None
    _num_cregs = 0
    _cregsizes = None
    _num_zvars = None
    _num_zregs = 0
    _zregsizes = None

    _parnames = ()

    # Whether this operation accepts repeated classical-bit targets in a single
    # instruction. Defaults to False. Operations that read and write the same
    # bit (e.g. ``c[1] = c[1] & c[2]``), or wrappers whose layout exposes
    # condition bits alongside body bits (``IfStatement``, ``WhileStatement``),
    # opt in. Qubit uniqueness is a physical constraint (no-cloning) and is
    # never relaxed.
    _allow_bit_aliasing = False

    # Whether this operation accepts repeated z-variable targets in a single
    # instruction. Defaults to False. ``Add``, ``Multiply``, ``Pow`` whose
    # output may also appear as an input opt in.
    _allow_zvar_aliasing = False

    @classmethod
    def allow_bit_aliasing(cls):
        """Whether an instruction may list the same classical bit more than once.

        ``False`` for most operations. Operations that read and write the same
        bit, such as the classical logic operations, return ``True``.
        """
        return cls._allow_bit_aliasing

    @classmethod
    def allow_zvar_aliasing(cls):
        """Whether an instruction may list the same z-variable more than once.

        ``False`` for most operations. Arithmetic operations whose output may
        also be one of their inputs return ``True``.
        """
        return cls._allow_zvar_aliasing

    @property
    def num_qubits(self):
        """Number of qubits the operation acts on.

        Read-only: assigning to it raises ``ValueError``.
        """
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, value):
        raise ValueError("Cannot set num_qubits. Read only parameter.")

    @property
    def num_ancillas(self):
        """Number of clean ancillas of the operation.

        The last ``num_ancillas`` qubit targets must be in ``|0>`` when the
        operation starts, and are returned in ``|0>``. On those inputs the
        operation acts as the one it was built for, which only touches the
        other qubits.

        It is set on :class:`GateDecl`, :class:`BlockDecl`, :class:`Block`
        and calls to them, and is ``0`` for every other operation.
        :func:`decompose` sets it when it wraps a decomposition that uses
        clean ancillas, and marks the ancillas of each call with
        :class:`AncillaAlloc` and :class:`AncillaFree`.
        """
        return 0

    @property
    def num_qregs(self):
        """Number of quantum registers (groups of qubit targets) the operation takes.

        Read-only: assigning to it raises ``ValueError``.
        """
        return self._num_qregs

    @num_qregs.setter
    def num_qregs(self, value):
        raise ValueError("Cannot set num_qregs. Read only parameter.")

    @property
    def num_bits(self):
        """Number of classical bits the operation acts on.

        Read-only: assigning to it raises ``ValueError``.
        """
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value):
        raise ValueError("Cannot set num_bits. Read only parameter.")

    @property
    def num_zvars(self):
        """Number of complex-valued z-variables the operation acts on.

        Read-only: assigning to it raises ``ValueError``.
        """
        return self._num_zvars

    @num_zvars.setter
    def num_zvars(self, value):
        raise ValueError("Cannot set num_zvars. Read only parameter.")

    @property
    def num_cregs(self):
        """Number of classical registers (groups of bit targets) the operation takes.

        Read-only: assigning to it raises ``ValueError``.
        """
        return self._num_cregs

    @num_cregs.setter
    def num_cregs(self, value):
        raise ValueError("Cannot set num_cregs. Read only parameter.")

    @property
    def qregsizes(self):
        """Size of each quantum register, as a list with one entry per register.

        Read-only: assigning to it raises ``ValueError``.
        """
        return self._qregsizes

    @qregsizes.setter
    def qregsizes(self, value):
        raise ValueError("Cannot set qregsizes. Read only parameter.")

    @property
    def cregsizes(self):
        """Size of each classical register, with one entry per register.

        Empty or ``None`` when the operation has no classical register.
        Read-only: assigning to it raises ``ValueError``.
        """
        return self._cregsizes

    @cregsizes.setter
    def cregsizes(self, value):
        raise ValueError("Cannot set cregsizes. Read only parameter.")

    @property
    def zregsizes(self):
        """Size of each z-variable register, with one entry per register.

        Empty or ``None`` when the operation has no z-variable register.
        Read-only: assigning to it raises ``ValueError``.
        """
        return self._zregsizes

    @zregsizes.setter
    def zregsizes(self, value):
        raise ValueError("Cannot set zregsizes. Read only parameter.")

    @property
    def name(self):
        """Short name of the operation type.

        For a wrapper, and for a named gate defined through one, this is the
        name of the wrapper: ``GateCX().name`` is ``"Control"``. ``str(op)``
        gives the name shown when printing a circuit.
        Read-only: assigning to it raises ``ValueError``.
        """
        return self._name

    @name.setter
    def name(self, value):
        raise ValueError("Cannot set name. Read only parameter.")

    @property
    def parnames(self):
        """Names of the operation's parameters, as a tuple of strings.

        Read-only: assigning to it raises ``ValueError``.
        """
        return self._parnames

    @parnames.setter
    def parnames(self, value):
        raise ValueError("Cannot set parnames. Read only parameter.")

    def getparams(self):
        """Return the values of the operation's parameters.

        Returns:
            list: One value per name in :attr:`parnames`, in the same order.
            Values may be numbers or symbolic expressions.

        Examples:
            >>> from mimiqcircuits import *
            >>> # GateU has a fourth parameter, the global phase, defaulting to 0
            >>> g = GateU(0.1, 0.2, 0.3)
            >>> g.parnames
            ('theta', 'phi', 'lmbda', 'gamma')
            >>> g.getparams()
            [0.1, 0.2, 0.3, 0.0]
        """
        return [getattr(self, pn) for pn in self._parnames]

    def is_symbolic(self):
        """Whether any parameter contains a free symbol.

        For wrappers such as :class:`Control` or :class:`Power`, the wrapped
        operation is checked too.

        Returns:
            bool: ``True`` if at least one parameter is symbolic.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import Symbol
            >>> GateRX(Symbol("theta")).is_symbolic()
            True
            >>> GateRX(0.5).is_symbolic()
            False
        """
        for param in self.getparams():
            if isinstance(param, (int, float, complex)):
                continue
            if _is_basic(param) and param.free_symbols:
                return True
        inner = getattr(self, "op", None)
        return isinstance(inner, Operation) and inner.is_symbolic()

    def getparam(self, pn):
        """Return the value of the parameter named ``pn``.

        Args:
            pn (str): Parameter name, one of :attr:`parnames`.

        Returns:
            The parameter value, a number or a symbolic expression.

        Raises:
            ValueError: If the operation has no parameter named ``pn``.

        Examples:
            >>> from mimiqcircuits import *
            >>> GateRX(0.5).getparam("theta")
            0.5
        """
        if pn not in self.parnames:
            raise ValueError(f"Parameter {pn} not found.")
        return getattr(self, pn)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__dict__ == other.__dict__

    def __hash__(self):
        def make_hashable(value):
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            if isinstance(value, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
            return value

        return hash(
            (
                type(self),
                tuple(sorted((k, make_hashable(v)) for k, v in self.__dict__.items())),
            )
        )

    def copy(self):
        """Return a shallow copy of the operation.

        The copy shares its attributes (for example a wrapped operation) with
        the original. Use :meth:`deepcopy` for an independent copy.

        Returns:
            Operation: A new operation of the same type.
        """
        return copy.copy(self)

    def deepcopy(self):
        """Return a deep copy of the operation and all its attributes.

        Returns:
            Operation: A new operation equal to the original that shares no
            mutable state with it.
        """
        return copy.deepcopy(self)

    @abstractmethod
    def iswrapper(self):
        """Whether the operation wraps another operation.

        ``True`` for composite operations such as :class:`Control`,
        :class:`Power`, :class:`Inverse`, :class:`Parallel` or
        :class:`IfStatement`, ``False`` otherwise.

        Returns:
            bool
        """
        pass

    def isopalias(self):
        """Whether the operation is a named alias of a wrapper.

        Some named gates are defined as a wrapper around another operation,
        for example ``GateS`` as ``GateZ`` to the power 1/2. Such gates return
        ``True``. Not every such gate reports it: ``GateSDG`` and ``GateCX``
        return ``False``.

        Returns:
            bool: ``True`` for a named alias of a wrapper that reports it,
            ``False`` otherwise.
        """
        return False

    def supports_canonical_rewrite(self):
        """Whether :class:`CanonicalRewrite` may decompose this operation.

        Operations opt in: only those the canonical rewrite knows how to
        handle return ``True``.

        Returns:
            bool
        """
        return False

    def numparams(self):
        """Return the number of parameters of the operation.

        Returns:
            int: The length of :meth:`getparams`.
        """
        return len(self.getparams())

    def _decompose(self, circ, qubits, bits, zvars):
        return circ.push(self, *qubits, *bits, *zvars)

    def _decompose_ctx(self, circ, qubits, bits, zvars, ctx):
        """Like ``_decompose``, with a ``DecompositionContext``.

        Operations whose decomposition can use the context, for example to
        borrow qubits, override this method. The default drops it.
        """
        return self._decompose(circ, qubits, bits, zvars)

    def decompose(self):
        """Decompose the operation one level into simpler operations.

        The result is a new circuit acting on qubits ``0`` to
        ``num_qubits - 1`` (and likewise for bits and z-variables). Operations
        without a decomposition return a circuit holding only themselves.

        Returns:
            Circuit: The decomposed operation.

        Examples:
            >>> from mimiqcircuits import *
            >>> # GateSWAP decomposes into three CX gates
            >>> GateSWAP().decompose()
            2-qubit circuit with 3 instructions:
            ├── CX @ q[0], q[1]
            ├── CX @ q[1], q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>
        """
        from mimiqcircuits.circuit import Circuit
        return self._decompose(
            Circuit(),
            range(self.num_qubits),
            range(self.num_bits),
            range(self.num_zvars),
        )

    def evaluate(self, d):
        """Substitute values for the symbols in the operation's parameters.

        Operations without symbolic parameters may return themselves.

        Args:
            d (dict): Map from symbols (or their names) to values.

        Returns:
            Operation: The operation with the symbols substituted.
        """
        return self

    def isidentity(self):
        """Whether the operation is known to act as the identity.

        Returns:
            bool: ``True`` only when the operation is known to be the
            identity. ``False`` does not prove the opposite.
        """
        return False

    @classmethod
    def isunitary(self):
        """Whether the operation acts unitarily on the quantum state.

        Returns:
            bool: ``True`` for gates, annotations, :class:`Barrier`, and the
            operations that only read the state into a z-variable, such as
            :class:`ExpectationValue` or :class:`Amplitude`. ``False`` for
            everything else: measurements, resets, noise channels, classical
            operations, and containers such as :class:`Block` or
            :class:`IfStatement`.
        """
        return False

    def asciiwidth(self, qubits, bits, zvars):
        # Calculate padding for qubits, bits, and zvars
        from mimiqcircuits.canvas import _gate_name_padding
        namepadding = _gate_name_padding(qubits, bits, zvars)

        # Calculate the width contribution of zvars
        zvar_width = len(",".join(map(str, zvars))) if zvars else 0

        # Assume operation has a string representation (e.g., __str__)
        operation_repr = str(self)

        # Calculate the width: | + (num + space) + name + zvars (if any) + |
        if zvars and not qubits:
            # If there are zvars but no qubits, ensure width includes zvars fully
            return 1 + max(namepadding + len(operation_repr), zvar_width) + 1
        else:
            # Regular case where zvars and qubits are both present or only qubits
            return 1 + namepadding + len(operation_repr) + 1

    def get_operation(self):
        """Return the operation that this one wraps, or the operation itself.

        Wrappers such as :class:`Control` or :class:`Power` return the
        operation they wrap; other operations return themselves.

        Returns:
            Operation
        """
        return self

    def listvars(self):
        """List the symbolic variables in the parameters of the operation.

        Wrappers such as ``Control``, ``Inverse``, ``Power``, ``Parallel``,
        ``Repeat``, ``IfStatement`` and ``WhileStatement`` give the variables
        of the operation they wrap, and a ``Block`` those of all its
        instructions.

        Returns:
            list: Each ``symengine.Symbol`` once, in no particular order.
            The list is empty when the operation has no symbolic parameters.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import Symbol
            >>> theta = Symbol("theta")
            >>> GateRX(theta).listvars()
            [theta]
            >>> Control(2, GateRY(2 * theta)).listvars()
            [theta]
            >>> GateH().listvars()
            []
        """
        vars = set()
        for param in self.getparams():
            if isinstance(param, se.Basic):
                vars.update(param.atoms(se.Symbol))
        # Wrappers (Control, Inverse, Power, Parallel, Repeat, IfStatement, ...)
        # do not all forward their parameters, so descend into the wrapped
        # operation directly.
        inner = getattr(self, "op", None)
        if isinstance(inner, Operation):
            vars.update(inner.listvars())
        return list(vars)


# export operations
__all__ = ["Operation"]
