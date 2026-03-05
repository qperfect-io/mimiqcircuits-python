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


class DecomposeIterator(Iterator["Instruction"]):
    """Iterator that yields instructions from recursive decomposition.

    Performs depth-first decomposition of a circuit to a target basis.
    This is memory-efficient for large circuits as it doesn't materialize
    the full decomposed circuit at once.

    Args:
        source: The source to decompose (Circuit, Instruction, or Operation).
        basis: The target decomposition basis.

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
    ):
        import mimiqcircuits as mc

        self._basis = basis
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

            # Decompose and push onto stack (reversed to maintain order)
            decomposed = self._basis.decompose(
                op, inst.qubits, inst.bits, inst.zvars
            )
            self._stack.extend(reversed(decomposed.instructions))

        raise StopIteration


def decompose(
    source: Circuit | Instruction | Operation,
    basis: DecompositionBasis | RewriteRule | None = None,
) -> Circuit:
    """Recursively decompose a circuit to a target basis.

    Decomposes all operations in the source until they are terminal in the
    given basis. This is the main entry point for circuit decomposition.

    Args:
        source: The circuit, instruction, or operation to decompose.
        basis: The target decomposition basis. If a RewriteRule is provided,
               it is wrapped in a RuleBasis. Defaults to CanonicalBasis.

    Returns:
        A new Circuit containing only terminal operations.

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

    result = mc.Circuit()
    for inst in DecomposeIterator(source, basis):
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
) -> DecomposeIterator:
    """Return an iterator over decomposed instructions.

    This is memory-efficient for large circuits as it yields instructions
    one at a time without materializing the full decomposed circuit.

    Args:
        source: The circuit, instruction, or operation to decompose.
        basis: The target decomposition basis. Defaults to CanonicalBasis.

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

    return DecomposeIterator(source, basis)


__all__ = [
    "decompose",
    "decompose_step",
    "eachdecomposed",
    "DecomposeIterator",
]
