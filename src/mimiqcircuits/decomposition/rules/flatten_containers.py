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
"""Rewrite rule for flattening container operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionError, RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


class FlattenContainers(RewriteRule):
    """Rewrite rule that expands container operations into their contents.

    Matches ``GateCall``, ``Block``, ``BlockCall``, and the ``Inverse`` of a
    ``GateCall``, and replaces each by the instructions it holds
    (inverted for an ``Inverse``), remapped onto the targets. One step
    expands one level: nested containers stay as they are. Gates are never
    decomposed. :class:`FlattenedBasis` repeats this rule until no container
    is left.

    Examples:
        >>> from mimiqcircuits import *
        >>> # declare a gate that prepares a Bell pair, and call it
        >>> bell = GateDecl("bell", (), Circuit().push(GateH(), 0).push(GateCX(), 0, 1))
        >>> c = Circuit()
        >>> c.push(bell(), 2, 3)
        4-qubit circuit with 1 instruction:
        └── bell() @ q[2:3]
        <BLANKLINE>
        >>> # the call is replaced by its body, on qubits 2 and 3
        >>> decompose_step(c, FlattenContainers())
        4-qubit circuit with 2 instructions:
        ├── H @ q[2]
        └── CX @ q[2], q[3]
        <BLANKLINE>
    """

    def matches(self, op: Operation) -> bool:
        """Check whether an operation is a container this rule expands.

        Args:
            op: The operation to check.

        Returns:
            ``True`` for ``GateCall``, ``Block``, ``BlockCall``, and the
            ``Inverse`` of a ``GateCall``.
        """
        from mimiqcircuits.blockdecl import BlockCall
        from mimiqcircuits.gatedecl import GateCall
        from mimiqcircuits.operations.block import Block
        from mimiqcircuits.operations.inverse import Inverse

        if isinstance(op, (GateCall, Block, BlockCall)):
            return True
        if isinstance(op, Inverse):
            return isinstance(op.op, (GateCall, Block))
        return False

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Expand one level of a container operation.

        Args:
            op: The container to expand.
            qubits: Qubit indices the container acts on.
            bits: Classical bit indices the container acts on.
            zvars: Z-variable indices the container acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the contents of
            ``op`` on those indices.

        Raises:
            DecompositionError: If ``op`` is not a container this rule
                matches.
        """
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        if not self.matches(op):
            raise DecompositionError(
                f"FlattenContainers cannot decompose {type(op).__name__}"
            )

        return CanonicalRewrite().decompose_step(op, qubits, bits, zvars)


__all__ = ["FlattenContainers"]
