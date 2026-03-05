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
"""Flattened decomposition basis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionBasis, DecompositionError

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


class FlattenedBasis(DecompositionBasis):
    """Flatten container operations while leaving all other operations untouched."""

    def isterminal(self, op: Operation) -> bool:
        """Containers are non-terminal; all other operations are terminal."""
        import mimiqcircuits as mc

        if isinstance(op, (mc.GateCall, mc.Block)):
            return False
        if isinstance(op, mc.Inverse):
            return not isinstance(op.op, (mc.GateCall, mc.Block))
        return True

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose one container expansion step using FlattenContainers."""
        from mimiqcircuits.decomposition.rules import FlattenContainers

        rule = FlattenContainers()
        if rule.matches(op):
            return rule.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} matches FlattenedBasis but cannot be decomposed."
        )


__all__ = ["FlattenedBasis"]
