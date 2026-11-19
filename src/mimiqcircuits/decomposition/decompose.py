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
"""Main decomposition functions for quantum circuits.

This module provides the primary API for decomposing quantum circuits:

- :func:`decompose`: recursively decompose to a target basis.
- :func:`eachdecomposed`: the same, as a lazy iterator.
- :func:`decompose_step`: apply a rewrite rule once.
- :class:`DecomposeIterator`: the iterator behind both.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from mimiqcircuits.decomposition.abstract import (
    DecompositionBasis,
    DecompositionContext,
    RewriteRule,
    _AncillaAllocator,
    _borrowable,
)
from mimiqcircuits.decomposition.basis import CanonicalBasis, RuleBasis
from mimiqcircuits.decomposition.rules import CanonicalRewrite
from mimiqcircuits.operations.gates.gate import Gate

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.instruction import Instruction
    from mimiqcircuits.operations.operation import Operation


def _to_basis(basis: DecompositionBasis | RewriteRule) -> DecompositionBasis:
    """Convert a RewriteRule to a DecompositionBasis if needed."""
    if isinstance(basis, RewriteRule):
        return RuleBasis(basis)
    return basis


def _op_basename(op):
    """Build a descriptive base name from the wrapper chain.

    Examples:
        GateRX(0.5)                  -> "RX"
        Inverse(GateRX(0.5))        -> "RX_inv"
        Control(2, GateRX(0.5))     -> "C2RX"
        Control(1, GateH())         -> "CH"
        Power(GateRX(0.5), 3)       -> "RX_pow"
        Inverse(Control(2, GateH())) -> "C2H_inv"
        GateCall(decl, ...)          -> decl.name
        Inverse(GateCall(decl, ...)) -> decl.name + "_dagger"
    """
    from mimiqcircuits.operations.control import Control
    from mimiqcircuits.operations.inverse import Inverse
    from mimiqcircuits.operations.power import Power
    from mimiqcircuits.gatedecl import GateCall

    if isinstance(op, GateCall):
        return op.decl.name
    if isinstance(op, Control):
        n = op.num_controls
        inner = _op_basename(op.op)
        return f"C{n}{inner}" if n > 1 else f"C{inner}"
    if isinstance(op, Inverse):
        inner = _op_basename(op.op)
        suffix = "_dagger" if isinstance(op.op, GateCall) else "_inv"
        return f"{inner}{suffix}"
    if isinstance(op, Power):
        inner = _op_basename(op.op)
        return f"{inner}_pow"
    return getattr(op, "name", type(op).__name__)


def _haslosses(op):
    from mimiqcircuits.blockdecl import BlockCall
    from mimiqcircuits.operations.block import Block
    from mimiqcircuits.operations.ifstatement import IfStatement
    from mimiqcircuits.operations.losschannel import Loss, Reload
    from mimiqcircuits.operations.repeat import Repeat
    from mimiqcircuits.operations.whilestatement import WhileStatement

    if isinstance(op, (Loss, Reload)):
        return True
    if isinstance(op, (IfStatement, WhileStatement, Repeat)):
        return _haslosses(op.op)
    if isinstance(op, Block):
        return any(_haslosses(i.operation) for i in op.instructions)
    if isinstance(op, BlockCall):
        return any(_haslosses(i.operation) for i in op.decl.instructions)
    return False


def _wrapped_decl_name(op):
    """Generate a GateDecl name for a wrapped (non-terminal) operation."""
    return f"mimiq_{_op_basename(op)}"


def _rebuild_with_symbols(op, symbols):
    """Rebuild a (possibly wrapped) operation with symbolic parameters.

    Traverses the wrapper chain (Control, Inverse, Power) and reconstructs
    with symbolic params at the leaf gate.
    """
    from mimiqcircuits.operations.control import Control
    from mimiqcircuits.operations.inverse import Inverse
    from mimiqcircuits.operations.power import Power
    from mimiqcircuits.gatedecl import GateCall

    if isinstance(op, Control):
        inner = _rebuild_with_symbols(op.op, symbols)
        return Control(op.num_controls, inner)
    if isinstance(op, Inverse):
        inner = _rebuild_with_symbols(op.op, symbols)
        return Inverse(inner)
    if isinstance(op, Power):
        inner = _rebuild_with_symbols(op.op, symbols)
        return Power(inner, op.exponent)
    if isinstance(op, GateCall):
        return GateCall(op.decl, symbols)
    # Leaf gate — reconstruct with symbolic params
    return type(op)(*symbols)


def _get_leaf_parnames(op):
    """Get parnames from the leaf gate in a wrapper chain.

    Wrappers (Control, Inverse, Power) delegate getparams() to their inner op
    but may not set _parnames. This traverses to the leaf to find the actual
    parameter names.
    """
    from mimiqcircuits.operations.control import Control
    from mimiqcircuits.operations.inverse import Inverse
    from mimiqcircuits.operations.power import Power
    from mimiqcircuits.gatedecl import GateCall

    if isinstance(op, (Control, Inverse, Power)):
        return _get_leaf_parnames(op.op)
    if isinstance(op, GateCall):
        # GateCall params come from decl.arguments, not _parnames
        return tuple(str(a) for a in op.decl.arguments)
    return op._parnames


def _compact_extra(circ, nq, nqb):
    """Renumber the extra qubits a body used, keeping their order.

    The borrowed slots (from ``nq`` to ``nqb - 1``) come first, then the clean
    ones (from ``nqb`` on). The declaration is then only as wide as what it
    touches, with its clean ancillas last. Returns the body and the number of
    clean ancillas.
    """
    from mimiqcircuits.circuit import Circuit

    used = sorted({q for inst in circ for q in inst.qubits if q >= nq})
    remap = {u: nq + i for i, u in enumerate(used)}
    out = Circuit()
    for inst in circ:
        qubits = [remap.get(q, q) for q in inst.qubits]
        out.push(inst.operation, *qubits, *inst.bits, *inst.zvars)
    return out, sum(1 for u in used if u >= nqb)


def _wrap_decomposition(inst, basis, cache, ctx):
    """Wrap a non-terminal gate instruction into a GateDecl/GateCall.

    Decomposes the operation one step, recursively wraps the result and
    creates a GateDecl from the wrapped body. Returns the instructions to
    emit: the GateCall, with markers around it for the clean ancillas it
    uses. A body that measures becomes a BlockDecl instead, and the call a
    BlockCall that writes the measured bit.
    """
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.instruction import Instruction
    from mimiqcircuits.operations.inverse import Inverse
    from mimiqcircuits.gatedecl import GateCall, GateDecl
    from mimiqcircuits.operations.annotations import AncillaAlloc, AncillaFree
    from mimiqcircuits.blockdecl import BlockCall, BlockDecl

    op = inst.operation
    qs = inst.qubits
    bs = inst.bits
    zs = inst.zvars
    nq = op.num_qubits

    # Step 1: Canonicalize for parametric reuse.
    # For GateCall/Inverse(GateCall), use the decl's own symbolic arguments.
    # For all other parametric ops (including wrappers like Control(2, GateRX(0.5))),
    # rebuild the op with fresh symbols from the leaf gate's parnames.
    # Note: Control._parnames is () even when inner gate has params,
    # so we use _get_leaf_parnames + getparams() to detect parametric ops.
    if isinstance(op, GateCall):
        # GateCall(decl, args) -> canonical = GateCall(decl, decl.arguments)
        symbols = op.decl.arguments
        concrete_args = op.arguments
        canonical_op = GateCall(op.decl, symbols) if symbols else op
    elif isinstance(op, Inverse) and isinstance(op.op, GateCall):
        # Inverse(GateCall(decl, args)) -> canonical = Inverse(GateCall(decl, decl.arguments))
        symbols = op.op.decl.arguments
        concrete_args = op.op.arguments
        if symbols:
            canonical_gc = GateCall(op.op.decl, symbols)
            canonical_op = Inverse(canonical_gc)
        else:
            canonical_op = op
    else:
        leaf_parnames = _get_leaf_parnames(op)
        if leaf_parnames:
            # Parametric op: plain gates (GateRYY(0.5)) AND wrappers
            # (Control(2, GateRX(0.5)), Inverse(GateRX(0.5)), Power(GateRX(0.5), 3))
            from symengine import Symbol

            symbols = tuple(Symbol(pn) for pn in leaf_parnames)
            concrete_args = tuple(op.getparams())
            try:
                canonical_op = _rebuild_with_symbols(op, symbols)
            except Exception:
                # Fallback: no canonicalization
                canonical_op = op
                concrete_args = ()
                symbols = ()
        else:
            # Non-parametric ops (Inverse(GateH), Power(GateX, 3), etc.)
            canonical_op = op
            concrete_args = ()
            symbols = ()

    # Step 2: Check cache
    # The body is built on targets 0..nq-1, followed by nb slots for the
    # qubits it may borrow and nc slots for clean ancillas. The borrowable
    # qubits are nearest-first in both frames, so slot nq + j stands for
    # borrowable[j] at this call site. Clean ancillas are taken at the call
    # site once the body has said how many it uses.
    borrowable = _borrowable(ctx, qs, ctx.nqubits)
    nborrow = len(borrowable)
    nclean = ctx.allocator.available() if ctx.allocator is not None else 0

    def call(op, nextra, num_ancillas, bits=bs):
        """Call ``op`` on the data qubits, borrowed ones and clean ancillas.

        The clean ancillas are taken now, and the call-site markers bracket
        them.
        """
        clean = ctx.take_clean(num_ancillas)
        borrowed = borrowable[: nextra - num_ancillas]
        qubits = tuple(qs) + tuple(borrowed) + tuple(clean)
        return (
            [Instruction(AncillaAlloc(), (q,)) for q in clean]
            + [Instruction(op, qubits, bits, zs)]
            + [Instruction(AncillaFree(), (q,)) for q in clean]
        )

    def call_decl(decl):
        if isinstance(decl, BlockDecl):
            return call(
                BlockCall(decl, concrete_args),
                decl.num_qubits - nq,
                decl.num_ancillas,
                (ctx.bit,),
            )
        return call(
            GateCall(decl, concrete_args), decl.num_qubits - nq, decl.num_ancillas
        )

    # Only the rule that decomposes the operation itself gets clean ancillas;
    # the operations it produces are wrapped with none. With measurements
    # allowed, its body may measure into bit 0, the bit of the call site.
    # A measuring body must not be reused where only unitaries are allowed,
    # hence `measure` in the key.
    measure = nclean > 0 and ctx.can_measure()
    key = (canonical_op, nborrow, nclean, measure)
    if key in cache:
        return call_decl(cache[key])

    bodyalloc = _AncillaAllocator(nq + nborrow, nclean) if nclean > 0 else None
    bodyctx = DecompositionContext(
        nqubits=nq + nborrow,
        borrow=ctx.borrow,
        unitary_only=not measure,
        allocator=bodyalloc,
        bit=0 if measure else None,
    )
    innerctx = DecompositionContext(
        nqubits=nq + nborrow, borrow=ctx.borrow, unitary_only=True
    )
    one_step = basis.decompose_ctx(canonical_op, tuple(range(nq)), (), (), bodyctx)

    # Declaration bodies carry no ancilla markers: the call site has them.
    wrapped = Circuit()
    for sub_inst in DecomposeIterator(
        one_step, basis, wrap=True, cache=cache, ctx=innerctx
    ):
        if not isinstance(sub_inst.operation, (AncillaAlloc, AncillaFree)):
            wrapped.push(sub_inst)

    body, L = _compact_extra(wrapped, nq, nq + nborrow)

    # A body that measures cannot be a GateDecl; it becomes a BlockDecl whose
    # bit is the measured bit of the call site.
    name = _wrapped_decl_name(op)
    if all(isinstance(i.operation, Gate) for i in body):
        decl = GateDecl(name, symbols, body, num_ancillas=L)
    else:
        decl = BlockDecl(name, symbols, body, num_ancillas=L)

    cache[key] = decl
    return call_decl(decl)


def _wrap_block(inst, basis, cache, ctx):
    """Wrap a non-terminal, non-gate instruction into a Block.

    The Block holds the wrapped decomposition of the operation, so the gates
    in it become GateCalls as usual. Returns the Block instruction in a list,
    as :func:`_wrap_decomposition` does.
    """
    from mimiqcircuits.instruction import Instruction
    from mimiqcircuits.operations.block import Block

    op = inst.operation
    nq, nb, nz = op.num_qubits, op.num_bits, op.num_zvars
    bodyctx = DecompositionContext(nqubits=nq, borrow=ctx.borrow)
    one_step = basis.decompose_ctx(
        op, tuple(range(nq)), tuple(range(nb)), tuple(range(nz)), bodyctx
    )

    # Push one by one: the Block constructor deep-copies its instructions,
    # which would duplicate the cached GateDecls.
    block = Block(nq, nb, nz)
    for sub_inst in DecomposeIterator(
        one_step, basis, wrap=True, cache=cache, ctx=bodyctx
    ):
        block.push(sub_inst)

    return [Instruction(block, inst.qubits, inst.bits, inst.zvars)]


class DecomposeIterator(Iterator["Instruction"]):
    """Iterator over the instructions of a recursive decomposition.

    Decomposes depth-first and yields each terminal instruction as soon as it
    is reached, in circuit order, so the decomposed circuit is never held in
    memory at once. Each non-terminal operation is decomposed only when the
    iteration reaches it. :func:`eachdecomposed` builds one with the default
    basis, and :func:`decompose` collects one into a circuit.

    Args:
        source: The circuit, instruction or operation to decompose. A bare
            operation is placed on qubits, bits and z-variables ``0, 1, ...``.
        basis: The target basis. Unlike :func:`decompose`, a rewrite rule is
            not accepted here; wrap it in a :class:`RuleBasis`.
        wrap: If ``False`` (default), yield only terminal operations. If
            ``True``, yield each non-terminal gate as a ``GateCall`` to a
            ``GateDecl`` holding its decomposition, and each other
            non-terminal operation (such as ``MeasureReset``) as a ``Block``
            holding its decomposition, in both cases wrapped again at every
            level. A gate whose decomposition measures an ancilla becomes a
            ``BlockCall`` to a ``BlockDecl``.
        cache: Dictionary that maps operations to the declaration built for
            them when ``wrap=True``, keyed by the (symbolic) operation, the
            numbers of borrowed and clean qubits its body was offered, and
            whether it may measure. Pass the same dictionary to several
            iterators to share declarations between them. A new one is used
            by default.
        nqubits: Width of the circuit, which bounds the qubits a basis with
            ``borrow=True`` may borrow. Defaults to ``source.num_qubits()``
            for a Circuit and to 0 (borrow nothing) for anything else, so the
            result never depends on the values of the target indices alone.
        ctx: The :class:`DecompositionContext` to use instead of the one
            built from ``basis`` and ``nqubits``.

    Raises:
        ValueError: If the basis borrows qubits and ``source`` contains
            ``Loss`` or ``Reload``: losses act on qubits a borrowing
            decomposition relies on. Resolve them first, for example with
            ``lower_losses``.
        TypeError: If ``source`` is not a circuit, instruction or operation.
        DecompositionError: While iterating, if the basis cannot decompose an
            operation.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> # instructions are produced one at a time
        >>> for inst in DecomposeIterator(c, CanonicalBasis()):
        ...     print(inst)
        U((1/2)*pi, 0, pi, 0.0) @ q[0]
    """

    def __init__(
        self,
        source: Circuit | Instruction | Operation,
        basis: DecompositionBasis,
        wrap: bool = False,
        cache: dict | None = None,
        nqubits: int = 0,
        ctx: DecompositionContext | None = None,
    ):
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.instruction import Instruction
        from mimiqcircuits.operations.operation import Operation

        basis = _to_basis(basis)
        self._basis = basis
        self._wrap = wrap
        self._cache = cache if cache is not None else {}
        self._stack: list[Instruction] = []
        # Wrapped calls come with their ancilla markers, queued here.
        self._ready: list[Instruction] = []

        # Initialize stack based on source type
        if isinstance(source, Circuit):
            # Reverse so we process in order (stack is LIFO)
            self._stack = list(reversed(source.instructions))
        elif isinstance(source, Instruction):
            self._stack = [source]
        elif isinstance(source, Operation):
            # Create instruction with canonical targets
            inst = Instruction(
                source,
                tuple(range(source.num_qubits)),
                tuple(range(source.num_bits)),
                tuple(range(source.num_zvars)),
            )
            self._stack = [inst]
        else:
            raise TypeError(
                f"Cannot decompose {type(source).__name__}. "
                "Expected Circuit, Instruction, or Operation."
            )

        if ctx is None:
            nbits = 0
            if isinstance(source, Circuit):
                nqubits = max(nqubits, source.num_qubits())
                nbits = source.num_bits()
                if basis._ancillaoptions().allow_measurement and nbits == 0:
                    # Such a circuit is sampled on all its qubits; a measured
                    # ancilla bit would make it sampled on that bit alone.
                    raise ValueError(
                        "allow_measurement needs a circuit with classical bits"
                    )
            ctx = DecompositionContext.from_basis(basis, nqubits, nbits)
            if ctx.borrow and any(_haslosses(i.operation) for i in self._stack):
                raise ValueError(
                    "cannot borrow qubits in a circuit with Loss or Reload; "
                    "resolve the losses first, for example with lower_losses"
                )
        self._ctx = ctx

    def __iter__(self) -> DecomposeIterator:
        return self

    def _emit(self, inst):
        if self._ctx.allocator is not None:
            self._ctx.allocator.track(inst)
        return inst

    def __next__(self) -> Instruction:
        while self._ready or self._stack:
            if self._ready:
                return self._emit(self._ready.pop())

            inst = self._stack.pop()
            op = inst.operation

            if self._basis.isterminal(op):
                return self._emit(inst)

            if self._wrap:
                if isinstance(op, Gate):
                    out = _wrap_decomposition(inst, self._basis, self._cache, self._ctx)
                else:
                    out = _wrap_block(inst, self._basis, self._cache, self._ctx)
                self._ready.extend(reversed(out))
                continue

            decomposed = self._basis.decompose_ctx(
                op, inst.qubits, inst.bits, inst.zvars, self._ctx
            )
            self._stack.extend(reversed(decomposed.instructions))

        raise StopIteration


def decompose(
    source: Circuit | Instruction | Operation,
    basis: DecompositionBasis | RewriteRule | None = None,
    wrap: bool = False,
) -> Circuit:
    """Recursively decompose a circuit to a target basis.

    Rewrites every operation of ``source`` that is not terminal in ``basis``,
    and repeats on the result, until only terminal operations are left. The
    source is not modified.

    Args:
        source: The circuit, instruction or operation to decompose. A bare
            operation is placed on qubits, bits and z-variables ``0, 1, ...``.
        basis: The target basis. A :class:`RewriteRule` is wrapped in a
            :class:`RuleBasis`, so it is applied until it no longer matches.
            Defaults to :class:`CanonicalBasis`.
        wrap: If ``False`` (default), the result holds only terminal
            operations. If ``True``, each non-terminal gate becomes a
            ``GateCall`` to a ``GateDecl`` holding its decomposition, and
            each other non-terminal operation (such as ``MeasureReset``) a
            ``Block`` holding its decomposition, both wrapped again at every
            level. Gates that differ only in their parameters share one
            ``GateDecl``. A gate whose decomposition measures an ancilla
            (``allow_measurement=True``) becomes a ``BlockCall`` to a shared
            ``BlockDecl``. This keeps the structure of the original circuit
            visible.

    Returns:
        A new :class:`~mimiqcircuits.Circuit` of terminal operations, plus
        ``GateCall`` and ``Block`` operations when ``wrap=True``.

    Raises:
        TypeError: If ``source`` is not a circuit, instruction or operation.
        DecompositionError: If ``basis`` cannot decompose an operation, for
            example a ``GateT`` in :class:`StimBasis`.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateCCX(), 0, 1, 2)
        3-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── C₂X @ q[0:1], q[2]
        <BLANKLINE>

        >>> # the default basis leaves only GateU and GateCX
        >>> sorted({type(inst.operation).__name__ for inst in decompose(c)})
        ['GateCX', 'GateU']

        >>> # any other basis can be given
        >>> sorted({type(inst.operation).__name__ for inst in decompose(c, CliffordTBasis())})
        ['GateCX', 'GateH', 'GateT', 'GateTDG']

        >>> # with wrap=True each gate becomes a call to a declared gate,
        >>> # so the two CH gates share one declaration
        >>> c = Circuit()
        >>> c.push(GateCH(), 0, 1)
        2-qubit circuit with 1 instruction:
        └── CH @ q[0], q[1]
        <BLANKLINE>
        >>> c.push(GateCH(), 1, 2)
        3-qubit circuit with 2 instructions:
        ├── CH @ q[0], q[1]
        └── CH @ q[1], q[2]
        <BLANKLINE>
        >>> decompose(c, wrap=True)
        3-qubit circuit with 2 instructions:
        ├── mimiq_CH() @ q[0:1]
        └── mimiq_CH() @ q[1:2]
        <BLANKLINE>
    """
    from mimiqcircuits.circuit import Circuit

    if basis is None:
        basis = CanonicalBasis()
    else:
        basis = _to_basis(basis)

    cache = {} if wrap else None
    result = Circuit()
    for inst in DecomposeIterator(source, basis, wrap=wrap, cache=cache):
        result.push(inst)

    return result


def decompose_step(
    source: Circuit | Instruction | Operation,
    rule: RewriteRule | None = None,
) -> Circuit:
    """Apply a rewrite rule once to every instruction of a circuit.

    Each instruction whose operation matches ``rule`` is replaced by the
    result of :meth:`RewriteRule.decompose_step`; the others are copied
    unchanged. Unlike :func:`decompose`, the result is not rewritten again,
    so it can still hold operations that match ``rule``.

    Args:
        source: The circuit, instruction or operation to rewrite. A bare
            operation is placed on qubits, bits and z-variables ``0, 1, ...``.
        rule: The rewrite rule to apply. Defaults to
            :class:`CanonicalRewrite`. A :class:`DecompositionBasis` is not
            accepted.

    Returns:
        A new :class:`~mimiqcircuits.Circuit` with the rule applied once.

    Raises:
        TypeError: If ``source`` is not a circuit, instruction or operation.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateRX(pi/3), 0)
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── RX((1/3)*pi) @ q[0]
        <BLANKLINE>
        >>> # ZYZRewrite matches RX but not H, so only RX is rewritten
        >>> decompose_step(c, ZYZRewrite())
        1-qubit circuit with 4 instructions:
        ├── H @ q[0]
        ├── S @ q[0]
        ├── RY((1/3)*pi) @ q[0]
        └── S† @ q[0]
        <BLANKLINE>

        >>> # the default rule expands one level of the built-in decomposition
        >>> decompose_step(GateSWAP())
        2-qubit circuit with 3 instructions:
        ├── CX @ q[0], q[1]
        ├── CX @ q[1], q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
    """
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.instruction import Instruction
    from mimiqcircuits.operations.operation import Operation

    if rule is None:
        rule = CanonicalRewrite()

    result = Circuit()

    # Handle different source types
    if isinstance(source, Operation):
        qubits = tuple(range(source.num_qubits))
        bits = tuple(range(source.num_bits))
        zvars = tuple(range(source.num_zvars))

        if rule.matches(source):
            decomposed = rule.decompose_step(source, qubits, bits, zvars)
            result.append(decomposed)
        else:
            result.push(source, *qubits, *bits, *zvars)

    elif isinstance(source, Instruction):
        op = source.operation
        if rule.matches(op):
            decomposed = rule.decompose_step(
                op, source.qubits, source.bits, source.zvars
            )
            result.append(decomposed)
        else:
            result.push(source)

    elif isinstance(source, Circuit):
        for inst in source:
            op = inst.operation
            if rule.matches(op):
                decomposed = rule.decompose_step(
                    op, inst.qubits, inst.bits, inst.zvars
                )
                result.append(decomposed)
            else:
                result.push(inst)

    else:
        raise TypeError(
            f"Cannot decompose {type(source).__name__}. "
            "Expected Circuit, Instruction, or Operation."
        )

    return result


def eachdecomposed(
    source: Circuit | Instruction | Operation,
    basis: DecompositionBasis | RewriteRule | None = None,
    wrap: bool = False,
) -> DecomposeIterator:
    """Iterate over the instructions of a recursive decomposition.

    Yields the same instructions as :func:`decompose`, one at a time, without
    building the decomposed circuit. Use it to stream a large decomposition
    or to stop early.

    Args:
        source: The circuit, instruction or operation to decompose. A bare
            operation is placed on qubits, bits and z-variables ``0, 1, ...``.
        basis: The target basis. A :class:`RewriteRule` is wrapped in a
            :class:`RuleBasis`. Defaults to :class:`CanonicalBasis`.
        wrap: Wrap non-terminal operations instead of expanding them, as in
            :func:`decompose`. Defaults to ``False``.

    Returns:
        A :class:`DecomposeIterator` over the decomposed instructions.

    Raises:
        TypeError: If ``source`` is not a circuit, instruction or operation.
        DecompositionError: While iterating, if ``basis`` cannot decompose an
            operation.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GateCCX(), 0, 1, 2)
        3-qubit circuit with 1 instruction:
        └── C₂X @ q[0:1], q[2]
        <BLANKLINE>
        >>> # instructions are computed as they are requested
        >>> it = eachdecomposed(c)
        >>> next(it)
        U((1/2)*pi, 0, pi, 0.0) @ q[2]
    """
    if basis is None:
        basis = CanonicalBasis()
    else:
        basis = _to_basis(basis)

    cache = {} if wrap else None
    return DecomposeIterator(source, basis, wrap=wrap, cache=cache)


__all__ = [
    "decompose",
    "decompose_step",
    "eachdecomposed",
    "DecomposeIterator",
]
