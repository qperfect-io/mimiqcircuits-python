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

- :func:`decompose`: Recursively decompose to a target basis
- :func:`decompose_step`: Single-step decomposition using a rewrite rule
- :class:`DecomposeIterator`: Memory-efficient iterator for decomposition
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from mimiqcircuits.decomposition.abstract import DecompositionBasis, RewriteRule
from mimiqcircuits.decomposition.basis import CanonicalBasis, RuleBasis
from mimiqcircuits.decomposition.rules import CanonicalRewrite

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Instruction, Operation


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
    import mimiqcircuits as mc
    from mimiqcircuits.gatedecl import GateCall

    if isinstance(op, GateCall):
        return op.decl.name
    if isinstance(op, mc.Control):
        n = op.num_controls
        inner = _op_basename(op.op)
        return f"C{n}{inner}" if n > 1 else f"C{inner}"
    if isinstance(op, mc.Inverse):
        inner = _op_basename(op.op)
        suffix = "_dagger" if isinstance(op.op, GateCall) else "_inv"
        return f"{inner}{suffix}"
    if isinstance(op, mc.Power):
        inner = _op_basename(op.op)
        return f"{inner}_pow"
    return getattr(op, "name", type(op).__name__)


def _wrapped_decl_name(op):
    """Generate a GateDecl name for a wrapped (non-terminal) operation."""
    return f"mimiq_{_op_basename(op)}"


def _rebuild_with_symbols(op, symbols):
    """Rebuild a (possibly wrapped) operation with symbolic parameters.

    Traverses the wrapper chain (Control, Inverse, Power) and reconstructs
    with symbolic params at the leaf gate. Mirrors Julia's _relax_wrapper_type
    approach.
    """
    import mimiqcircuits as mc
    from mimiqcircuits.gatedecl import GateCall

    if isinstance(op, mc.Control):
        inner = _rebuild_with_symbols(op.op, symbols)
        return mc.Control(op.num_controls, inner)
    if isinstance(op, mc.Inverse):
        inner = _rebuild_with_symbols(op.op, symbols)
        return mc.Inverse(inner)
    if isinstance(op, mc.Power):
        inner = _rebuild_with_symbols(op.op, symbols)
        return mc.Power(inner, op.exponent)
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
    import mimiqcircuits as mc
    from mimiqcircuits.gatedecl import GateCall

    if isinstance(op, (mc.Control, mc.Inverse, mc.Power)):
        return _get_leaf_parnames(op.op)
    if isinstance(op, GateCall):
        # GateCall params come from decl.arguments, not _parnames
        return tuple(str(a) for a in op.decl.arguments)
    return op._parnames


def _wrap_decomposition(inst, basis, cache):
    """Wrap a non-terminal instruction into a GateDecl/GateCall.

    Mirrors Julia's decompose.jl wrap logic: decomposes the operation one step,
    recursively wraps the result, creates a GateDecl from the wrapped body,
    and returns a GateCall instruction.
    """
    import mimiqcircuits as mc
    from mimiqcircuits.gatedecl import GateCall, GateDecl

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
    elif isinstance(op, mc.Inverse) and isinstance(op.op, GateCall):
        # Inverse(GateCall(decl, args)) -> canonical = Inverse(GateCall(decl, decl.arguments))
        symbols = op.op.decl.arguments
        concrete_args = op.op.arguments
        if symbols:
            canonical_gc = GateCall(op.op.decl, symbols)
            canonical_op = mc.Inverse(canonical_gc)
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
    if canonical_op in cache:
        cached_decl = cache[canonical_op]
        return mc.Instruction(
            GateCall(cached_decl, concrete_args), qs, bs, zs
        )

    # Step 3: Decompose one step
    one_step = basis.decompose(canonical_op, tuple(range(nq)), (), ())

    # Step 4: Recursively wrap results
    wrapped = mc.Circuit()
    for sub_inst in DecomposeIterator(one_step, basis, wrap=True, cache=cache):
        wrapped.push(sub_inst)

    # Step 5: Create GateDecl from wrapped body
    name = _wrapped_decl_name(op)
    decl = GateDecl(name, symbols, wrapped)

    # Step 6: Cache and return
    cache[canonical_op] = decl
    return mc.Instruction(GateCall(decl, concrete_args), qs, bs, zs)


class DecomposeIterator(Iterator["Instruction"]):
    """Iterator that yields instructions from recursive decomposition.

    Performs depth-first decomposition of a circuit to a target basis.
    This is memory-efficient for large circuits as it doesn't materialize
    the full decomposed circuit at once.

    Args:
        source: The source to decompose (Circuit, Instruction, or Operation).
        basis: The target decomposition basis.
        wrap: If True, wrap non-terminal ops into GateDecl/GateCall instead
              of flattening.
        cache: Shared cache dict for wrapped GateDecls (used when wrap=True).

    Example:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
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
    ):
        import mimiqcircuits as mc

        self._basis = basis
        self._wrap = wrap
        self._cache = cache if cache is not None else {}
        self._stack: list[mc.Instruction] = []

        # Initialize stack based on source type
        if isinstance(source, mc.Circuit):
            # Reverse so we process in order (stack is LIFO)
            self._stack = list(reversed(source.instructions))
        elif isinstance(source, mc.Instruction):
            self._stack = [source]
        elif isinstance(source, mc.Operation):
            # Create instruction with canonical targets
            inst = mc.Instruction(
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

    def __iter__(self) -> DecomposeIterator:
        return self

    def __next__(self) -> Instruction:
        while self._stack:
            inst = self._stack.pop()
            op = inst.operation

            if self._basis.isterminal(op):
                return inst

            if self._wrap:
                return _wrap_decomposition(inst, self._basis, self._cache)

            # Decompose and push onto stack (reversed to maintain order)
            decomposed = self._basis.decompose(
                op, inst.qubits, inst.bits, inst.zvars
            )
            self._stack.extend(reversed(decomposed.instructions))

        raise StopIteration


def decompose(
    source: Circuit | Instruction | Operation,
    basis: DecompositionBasis | RewriteRule | None = None,
    wrap: bool = False,
) -> Circuit:
    """Recursively decompose a circuit to a target basis.

    Decomposes all operations in the source until they are terminal in the
    given basis. This is the main entry point for circuit decomposition.

    Args:
        source: The circuit, instruction, or operation to decompose.
        basis: The target decomposition basis. If a RewriteRule is provided,
               it is wrapped in a RuleBasis. Defaults to CanonicalBasis.
        wrap: If True, wrap non-terminal ops into GateDecl/GateCall instead
              of flattening them into primitives. This produces a circuit with
              only terminal ops and GateCalls.

    Returns:
        A new Circuit containing only terminal operations (and GateCalls
        if wrap=True).

    Example:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.decomposition import decompose, CanonicalBasis
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
        >>> decomposed = decompose(c)
        >>> # decomposed contains only GateU and GateCX

        Using a specific basis:

        >>> from mimiqcircuits.decomposition import CliffordTBasis
        >>> decomposed = decompose(c, CliffordTBasis())

        Using a rewrite rule directly:

        >>> from mimiqcircuits.decomposition import ZYZRewrite
        >>> decomposed = decompose(c, ZYZRewrite())
    """
    import mimiqcircuits as mc

    if basis is None:
        basis = CanonicalBasis()
    else:
        basis = _to_basis(basis)

    cache = {} if wrap else None
    result = mc.Circuit()
    for inst in DecomposeIterator(source, basis, wrap=wrap, cache=cache):
        result.push(inst)

    return result


def decompose_step(
    source: Circuit | Instruction | Operation,
    rule: RewriteRule | None = None,
) -> Circuit:
    """Apply a single decomposition step to a circuit.

    Unlike :func:`decompose`, this function applies the rule only once
    (non-recursively). Operations that don't match the rule are left unchanged.

    Args:
        source: The circuit, instruction, or operation to decompose.
        rule: The rewrite rule to apply. Defaults to CanonicalRewrite.

    Returns:
        A new Circuit with the rule applied once.

    Example:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.decomposition import decompose_step, ZYZRewrite
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> # Apply ZYZ rewrite once (non-recursive)
        >>> stepped = decompose_step(c, ZYZRewrite())
    """
    import mimiqcircuits as mc

    if rule is None:
        rule = CanonicalRewrite()

    result = mc.Circuit()

    # Handle different source types
    if isinstance(source, mc.Operation):
        qubits = tuple(range(source.num_qubits))
        bits = tuple(range(source.num_bits))
        zvars = tuple(range(source.num_zvars))

        if rule.matches(source):
            decomposed = rule.decompose_step(source, qubits, bits, zvars)
            result.append(decomposed)
        else:
            result.push(source, *qubits, *bits, *zvars)

    elif isinstance(source, mc.Instruction):
        op = source.operation
        if rule.matches(op):
            decomposed = rule.decompose_step(
                op, source.qubits, source.bits, source.zvars
            )
            result.append(decomposed)
        else:
            result.push(source)

    elif isinstance(source, mc.Circuit):
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
    """Return an iterator over decomposed instructions.

    This is memory-efficient for large circuits as it yields instructions
    one at a time without materializing the full decomposed circuit.

    Args:
        source: The circuit, instruction, or operation to decompose.
        basis: The target decomposition basis. Defaults to CanonicalBasis.
        wrap: If True, wrap non-terminal ops into GateDecl/GateCall.

    Returns:
        An iterator yielding decomposed instructions.

    Example:
        >>> from mimiqcircuits import Circuit, GateH
        >>> from mimiqcircuits.decomposition import eachdecomposed
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> for inst in eachdecomposed(c):
        ...     print(inst)
        U((1/2)*pi, 0, pi, 0.0) @ q[0]
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
