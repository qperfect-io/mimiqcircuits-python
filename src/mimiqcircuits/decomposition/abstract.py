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
"""Abstract base classes for the decomposition framework.

This module defines the core abstractions for quantum circuit decomposition:

- :class:`RewriteRule`: Single-step transformation rules
- :class:`DecompositionBasis`: Recursive decomposition to a target gate set
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


class DecompositionError(Exception):
    """Raised when an operation cannot be decomposed.

    A basis raises it from :meth:`DecompositionBasis.decompose` for an
    operation it is not terminal for and has no rewrite for, and most rules
    raise it from :meth:`RewriteRule.decompose_step` for an operation they do
    not match.
    """

    pass


@dataclass(frozen=True)
class _AncillaOptions:
    """Ancilla options of a basis.

    Every basis that supports them stores one and returns it from
    ``_ancillaoptions``; the others use the defaults, which keep every
    decomposition inside the targets of the operation it decomposes.
    """

    borrow: bool = False
    max_ancillas: int = 0
    allow_measurement: bool = False

    def __post_init__(self):
        if self.max_ancillas < 0:
            raise ValueError(
                f"max_ancillas must be non-negative, got {self.max_ancillas}"
            )


class _AncillaAllocator:
    """Clean ancillas handed out while a circuit is decomposed.

    The iterator keeps it up to date as instructions stream out:
    ``AncillaAlloc`` raises the window depth of its qubit and ``AncillaFree``
    lowers it. A qubit whose depth drops to 0 is free, and stays free until an
    instruction touches it. Qubits a rule takes leave the free list at once,
    before their ``AncillaAlloc`` is streamed.
    """

    def __init__(self, next_qubit, remaining):
        self.next = next_qubit
        self.remaining = remaining
        self.depth = {}
        self.free = []

    def available(self):
        """Number of clean ancillas that can be taken now."""
        return len(self.free) + self.remaining

    def take(self, k):
        """Take up to ``k`` clean ancillas.

        Free ones first, lowest index first, then new ones numbered after the
        circuit.
        """
        self.free.sort()
        taken = self.free[:k]
        del self.free[: len(taken)]
        while len(taken) < k and self.remaining > 0:
            taken.append(self.next)
            self.next += 1
            self.remaining -= 1
        return taken

    def track(self, inst):
        from mimiqcircuits.operations.annotations import AncillaAlloc, AncillaFree

        op = inst.operation
        if isinstance(op, AncillaAlloc):
            q = inst.qubits[0]
            self.depth[q] = self.depth.get(q, 0) + 1
            if q in self.free:
                self.free.remove(q)
        elif isinstance(op, AncillaFree):
            q = inst.qubits[0]
            d = self.depth.get(q, 0) - 1
            if d > 0:
                self.depth[q] = d
            else:
                self.depth.pop(q, None)
                if q not in self.free:
                    self.free.append(q)
        else:
            # A qubit touched after it was freed is no longer known to be in |0>.
            for q in inst.qubits:
                if q in self.free:
                    self.free.remove(q)


@dataclass(frozen=True)
class DecompositionContext:
    """State that :func:`decompose` hands to a basis and its rules.

    The context goes next to the operation and its targets, through
    :meth:`DecompositionBasis.decompose_ctx` and
    :meth:`RewriteRule.decompose_step_ctx`.

    Args:
        nqubits: Width of the circuit being decomposed. Qubits
            ``0, ..., nqubits - 1`` that are not targets of the operation may
            be borrowed. ``0`` means the width is unknown, and nothing is
            borrowed.
        borrow: Whether rules may borrow qubits outside the operation's
            targets. A borrowed qubit can be in any state; the decomposition
            acts as the identity on it.
        unitary_only: Set when the result must be unitary, as for the body
            of a wrapped ``GateDecl`` or of an ``IfStatement``.

    :func:`decompose` also hands out clean ancillas through the context when
    the basis has ``max_ancillas > 0``. A rule that takes them brackets their
    use with :class:`AncillaAlloc` and :class:`AncillaFree` and returns them
    in ``|0>``. With ``allow_measurement=True`` on the basis, the context also
    holds a classical bit, numbered after those of the circuit, that rules may
    measure an ancilla into.

    Bases and rules that do not use the context inherit the default
    ``*_ctx`` methods, which drop it and call the method without it.
    """

    nqubits: int = 0
    borrow: bool = False
    unitary_only: bool = False
    allocator: _AncillaAllocator | None = None
    # Bit that measuring rules write to, or None when they may not measure.
    bit: int | None = None

    def __post_init__(self):
        if self.nqubits < 0:
            raise ValueError(f"nqubits must be non-negative, got {self.nqubits}")

    @classmethod
    def from_basis(cls, basis, nqubits, nbits=0):
        """The context for a circuit ``nqubits`` wide with ``nbits`` bits.

        Clean ancillas and the measured bit are numbered after them, so a
        circuit of unknown width gets none.
        """
        opts = basis._ancillaoptions()
        alloc = None
        if opts.max_ancillas > 0 and nqubits > 0:
            alloc = _AncillaAllocator(nqubits, opts.max_ancillas)
        bit = nbits if opts.allow_measurement and nbits > 0 else None
        return cls(nqubits=nqubits, borrow=opts.borrow, allocator=alloc, bit=bit)

    def unitary(self):
        """The same context for a result that must be unitary."""
        return DecompositionContext(
            nqubits=self.nqubits,
            borrow=self.borrow,
            unitary_only=True,
            allocator=self.allocator,
            bit=self.bit,
        )

    def can_measure(self):
        """Whether a rule may measure an ancilla."""
        return self.bit is not None and not self.unitary_only

    def available(self):
        """Number of clean ancillas that can be taken now."""
        return 0 if self.allocator is None else self.allocator.available()

    def take_clean(self, k):
        """Up to ``k`` clean ancillas for the rule that receives this context."""
        if self.allocator is None or k <= 0:
            return []
        return self.allocator.take(k)


_NO_CONTEXT = DecompositionContext()


def _uses_context(op):
    """Whether ``op`` has a rule that uses the decomposition context.

    Bases take the context for these operations alone, so an override of
    ``decompose`` for any other operation keeps being called.
    """
    from mimiqcircuits.operations.control import Control
    from mimiqcircuits.operations.ifstatement import IfStatement

    return isinstance(op, (Control, IfStatement))


def _borrowable(ctx, qubits, k):
    """Qubits a rule may borrow for an operation on ``qubits``, at most ``k``.

    Nearest to the targets first, so that MPS chains stay local.
    """
    if not ctx.borrow or k <= 0:
        return []
    targets = set(qubits)
    free = [q for q in range(ctx.nqubits) if q not in targets]
    free.sort(key=lambda q: (min(abs(q - t) for t in qubits), q))
    return free[:k]


class RewriteRule(ABC):
    """Abstract base class for single-step rewrite rules.

    A rewrite rule says *how* to rewrite an operation into a circuit of other
    operations, but not *when* to stop. Apply it once with
    :func:`decompose_step`, or repeatedly with :func:`decompose`, which wraps
    it in a :class:`RuleBasis` and rewrites until nothing matches. For a
    target gate set, see :class:`DecompositionBasis`.

    A subclass implements :meth:`matches` and :meth:`decompose_step`.

    Examples:
        >>> from mimiqcircuits import *
        >>> # a rule that rewrites a SWAP into three CX gates
        >>> class SwapToCX(RewriteRule):
        ...     def matches(self, op):
        ...         return isinstance(op, GateSWAP)
        ...
        ...     def decompose_step(self, op, qubits, bits, zvars):
        ...         # qubits are the targets of the SWAP in the circuit
        ...         a, b = qubits
        ...         circ = Circuit()
        ...         circ.push(GateCX(), a, b)
        ...         circ.push(GateCX(), b, a)
        ...         circ.push(GateCX(), a, b)
        ...         return circ
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateSWAP(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── SWAP @ q[0:1]
        <BLANKLINE>
        >>> # the SWAP matches and is rewritten; the H does not and is kept
        >>> decompose_step(c, SwapToCX())
        2-qubit circuit with 4 instructions:
        ├── H @ q[0]
        ├── CX @ q[0], q[1]
        ├── CX @ q[1], q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
    """

    @abstractmethod
    def matches(self, op: Operation) -> bool:
        """Check whether this rule can rewrite an operation.

        Args:
            op: The operation to check.

        Returns:
            ``True`` if :meth:`decompose_step` accepts ``op``.
        """
        ...

    @abstractmethod
    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Rewrite an operation one step.

        Call it only on an operation that :meth:`matches` accepts. The
        result acts on the given indices, so it can be appended in place of
        the original instruction. Its operations may themselves match the
        rule again.

        Args:
            op: The operation to rewrite.
            qubits: Qubit indices the operation acts on.
            bits: Classical bit indices the operation acts on.
            zvars: Z-variable indices the operation acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` that implements ``op`` on
            those indices.

        Raises:
            DecompositionError: If the rule cannot rewrite ``op``.
        """
        ...

    def decompose_step_ctx(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
        ctx: DecompositionContext,
    ) -> Circuit:
        """Apply this rule with a :class:`DecompositionContext`.

        Rules that use the context, for example to borrow qubits, override
        this method. The default drops the context and calls
        :meth:`decompose_step`.
        """
        return self.decompose_step(op, qubits, bits, zvars)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DecompositionBasis(ABC):
    """Abstract base class for decomposition targets.

    A basis decides which operations are terminal, that is kept as they are,
    and how to rewrite every other operation. :func:`decompose` rewrites each
    non-terminal operation with :meth:`decompose` and repeats on the result
    until only terminal operations are left.

    A subclass implements :meth:`isterminal` and :meth:`decompose`.

    Examples:
        >>> from mimiqcircuits import *
        >>> # a basis of GateU and GateCZ: CX becomes H, CZ, H and the rest
        >>> # follows the built-in decomposition
        >>> class CZBasis(DecompositionBasis):
        ...     def isterminal(self, op):
        ...         return isinstance(op, (GateU, GateCZ))
        ...
        ...     def decompose(self, op, qubits, bits, zvars):
        ...         if isinstance(op, GateCX):
        ...             control, target = qubits
        ...             circ = Circuit()
        ...             circ.push(GateH(), target)
        ...             circ.push(GateCZ(), control, target)
        ...             circ.push(GateH(), target)
        ...             return circ
        ...         return CanonicalRewrite().decompose_step(op, qubits, bits, zvars)
        >>> c = Circuit()
        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 1 instruction:
        └── CX @ q[0], q[1]
        <BLANKLINE>
        >>> # the H gates are not terminal, so they are decomposed in turn
        >>> decompose(c, CZBasis())
        2-qubit circuit with 3 instructions:
        ├── U((1/2)*pi, 0, pi, 0.0) @ q[1]
        ├── CZ @ q[0], q[1]
        └── U((1/2)*pi, 0, pi, 0.0) @ q[1]
        <BLANKLINE>
    """

    @abstractmethod
    def isterminal(self, op: Operation) -> bool:
        """Check whether an operation is terminal in this basis.

        Terminal operations are the target set: :func:`decompose` keeps them
        as they are.

        Args:
            op: The operation to check.

        Returns:
            ``True`` if ``op`` is kept, ``False`` if it is rewritten.
        """
        ...

    @abstractmethod
    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Rewrite a non-terminal operation one step.

        Called by :func:`decompose` only for operations that
        :meth:`isterminal` rejects. The result does not need to be terminal:
        :func:`decompose` keeps rewriting it.

        Args:
            op: The operation to rewrite.
            qubits: Qubit indices the operation acts on.
            bits: Classical bit indices the operation acts on.
            zvars: Z-variable indices the operation acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` that implements ``op`` on
            those indices.

        Raises:
            DecompositionError: If the basis has no rewrite for ``op``.
        """
        ...

    def decompose_ctx(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
        ctx: DecompositionContext,
    ) -> Circuit:
        """Decompose a non-terminal operation with a :class:`DecompositionContext`.

        :func:`decompose` always calls this method. Bases that pass the
        context on to their rules override it. The default drops the context
        and calls :meth:`decompose`.
        """
        return self.decompose(op, qubits, bits, zvars)

    def _ancillaoptions(self) -> _AncillaOptions:
        return _AncillaOptions()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


__all__ = [
    "RewriteRule",
    "DecompositionBasis",
    "DecompositionContext",
    "DecompositionError",
]
