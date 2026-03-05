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
    from mimiqcircuits import Circuit, Operation


class FlattenContainers(RewriteRule):
    """Flatten container operations into their constituent instructions.

    This rule expands:
    - ``GateCall``
    - ``Block``
    - ``Inverse(GateCall)``
    - ``Inverse(Block)`` (if present)

    The rule does not perform gate-level decomposition; it only expands
    container structure. Decomposition behavior is delegated to
    :class:`CanonicalRewrite`.
    """

    def matches(self, op: Operation) -> bool:
        """Check if an operation is a flattenable container."""
        import mimiqcircuits as mc

        if isinstance(op, (mc.GateCall, mc.Block)):
            return True
        if isinstance(op, mc.Inverse):
            return isinstance(op.op, (mc.GateCall, mc.Block))
        return False

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Flatten one container operation."""
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        if not self.matches(op):
            raise DecompositionError(
                f"FlattenContainers cannot decompose {type(op).__name__}"
            )

        return CanonicalRewrite().decompose_step(op, qubits, bits, zvars)


__all__ = ["FlattenContainers"]
